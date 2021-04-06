import numpy as np
import torch
import time

import scipy.sparse
from sksparse import cholmod
import torch.nn.functional as F
from multiprocessing import Pool



class GridCholeskySolver(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, chols, J, w, b):
        """ Solve linear system """
        B, H, W, M, N = J.shape
        D = b.shape[-1]
        bs = b.detach().reshape(B, -1, D).cpu().numpy()
        
        xs = []
        for i in range(len(chols)):
            xs += [ chols[i](bs[i]) ]

        xs = np.stack(xs).astype(np.float32)
        xs = torch.from_numpy(xs).to(J.device)
        xs = xs.view(B, H, W, N//4, D)

        ctx.chols = chols
        ctx.save_for_backward(xs, J, w, b)
        return xs

    @staticmethod
    def backward(ctx, grad_output):
        xs, J, w, b = ctx.saved_tensors
        B, H, W, M, N = J.shape
        D = b.shape[-1]

        gs = grad_output.reshape(B, -1, D).cpu().numpy()
        chols = ctx.chols
        
        dz = []
        for i in range(len(chols)):
            dz += [ chols[i](gs[i]) ]

        dz = np.stack(dz, axis=0).astype(np.float32)
        dz = torch.from_numpy(dz).to(J.device).view(*xs.shape)

        J = GridFactor(A=J, w=w)

        grad_J = torch.matmul(-w[...,None] * J.A(dz), J._unfold(xs).transpose(-1,-2)) + \
                 torch.matmul(-w[...,None] * J.A(xs), J._unfold(dz).transpose(-1,-2))
        
        grad_w = -torch.sum(J.A(xs) * J.A(dz), -1)

        return None, grad_J, grad_w, dz


sym_factor = None
sym_shape = None

class GridFactor:
    """ Generalized grid factors """
    def __init__(self, A=None, w=None):
        self.factors = []
        self.weights = []
        self.residuals = []
        
        self.chols = None
        self.Af = A
        self.wf = w

    def _build_factors(self):
        self.Af = torch.cat(self.factors, dim=3)
        self.wf = torch.cat(self.weights, dim=3)

    def add_factor(self, Js, ws=None, rs=None, ftype='u'):
        """ Add factor to graph """

        B, H, W, M, N = Js[0].shape
        device = Js[0].device

        A = torch.zeros([B, H, W, M, N, 2, 2]).to(device)
        w = torch.zeros([B, H, W, M]).to(device)
        
        # unary factor
        if ftype == 'u':
            A[...,0,0] = Js[0]
            w[:] = ws[:]
        
        # horizontal pairwise factor
        elif ftype == 'h':
            A[...,0,0] = Js[0]
            A[...,0,1] = Js[1]
            w[:, :, :-1, :] = ws[:, :, :-1, :]

        # verticle pairwise factor 
        elif ftype == 'v':
            A[...,0,0] = Js[0]
            A[...,1,0] = Js[1]
            w[:, :-1, :, :] = ws[:, :-1, :, :]

        A = A.view(B, H, W, M, 2*2*N)

        self.factors.append(A)
        self.weights.append(w)

        if rs is not None:
            self.residuals.append(rs)

    def _fold(self, x):
        """ Transposed fold operator """
        B, H, W, M, D = x.shape
        x = x.transpose(-1,-2)
        x = x.reshape(B, H, W, M*D)
        x = F.pad(x, [0,0,1,0,1,0])
        x = x.reshape(B, (H+1)*(W+1), M*D).permute(0, 2, 1)
        x = F.fold(x, [H, W], [2,2], padding=1)
        x = x.permute(0, 2, 3, 1).reshape(B, H, W, D, M//4)
        return x.transpose(-1,-2)

    def _unfold(self, x):
        """ Transposed unfold operator """
        B, H, W, N, D = x.shape
        x = x.transpose(-1,-2)
        x = F.pad(x.view(B, H, W, N*D), [0,0,0,1,0,1])
        x = x.permute(0, 3, 1, 2)
        x = F.unfold(x, [2,2], padding=0)
        x = x.permute(0, 2, 1).reshape(B, H, W, D, 4*N)
        return x.transpose(-1, -2)

    def A(self, x, w=False):
        """ Linear operator """
        return torch.matmul(self.Af, self._unfold(x))

    def At(self, y):
        """ Adjoint operator """
        w = self.wf.unsqueeze(dim=-1)
        At = self.Af.transpose(-1,-2)
        return self._fold(torch.matmul(At, w*y))

    def to_csc(self):
        """ Convert linear operator into scipy csc matrix"""

        if self.Af is None:
            self._build_factors()

        with torch.no_grad():
            B, H, W, N, M = self.Af.shape
            dims = [torch.arange(d).cuda() for d in (H, W, N, M//4)]

            i0, j0, k0, h0 = \
                [x.reshape(-1) for x in torch.meshgrid(*dims)]

            # repeats are ok because edge weights get zeroed
            s = [W*(M//4), M//4, 1]
            i1 = i0+1
            j1 = j0+1
            i1[i1 >= H] = H-1
            j1[j1 >= W] = W-1

            col_idx = torch.stack([
                s[0]*i0 + s[1]*j0 + s[2]*h0,
                s[0]*i0 + s[1]*j1 + s[2]*h0,
                s[0]*i1 + s[1]*j0 + s[2]*h0,
                s[0]*i1 + s[1]*j1 + s[2]*h0
            ], dim=-1).view(-1)

            dense_shape = [H*W*N, H*W*(M//4)]
            col_idx = col_idx.cpu().numpy()
            row_idx = M * np.arange(0, H*W*N+1)

            A = self.Af.detach().view(B, H*W*N, M)
            wsqrt = self.wf.detach().sqrt().view(B, H*W*N, 1)
            vals = (wsqrt * A).cpu().numpy()

            sparse_matricies = []
            for batch_ix in range(B):
                data = (vals[batch_ix].reshape(-1), col_idx, row_idx)
                mat = scipy.sparse.csr_matrix(data, shape=dense_shape)
                mat.sum_duplicates()
                sparse_matricies.append(mat.T)

        return sparse_matricies

    def factorAAt(self):
        """ Peform sparse cholesky factorization """
        global sym_factor, sym_shape

        with torch.no_grad():
            self.chols = []
            start = time.time()
            As = self.to_csc()

            if sym_factor is None or As[0].shape != sym_shape:
                sym_factor = cholmod.analyze_AAt(As[0], ordering_method='best')
                sym_shape = As[0].shape

            for A in As:
                chol = sym_factor.cholesky_AAt(A)
                self.chols.append(chol)

        return self.chols

    def solveAAt(self, b=None):
        if self.chols is None:
            self.factorAAt()

        if b is None:
            r = torch.cat(self.residuals, -2)
            b = self.At(r)

        x = GridCholeskySolver.apply(self.chols, self.Af, self.wf, b)
        return x.reshape(*b.shape)