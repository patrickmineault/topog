import numpy as np
import torch
from torch import nn
import torch.utils.benchmark as benchmark
import matplotlib.pyplot as plt

def next_multiple(x, y):
    return int(np.ceil(x / y) * y)


@torch.jit.script
def jitted_loop(X, w, kernel_size: int, h: int):
    X_out = torch.zeros(X.shape[0], w.shape[0], X.shape[2], X.shape[3])
    for i in range(h):
        for j in range(h):
            # Pick a chunk of X
            rgy = slice(i, i + kernel_size)
            rgx = slice(j, j + kernel_size)

            X_sub = X[:, :, rgy, rgx]
            w_sub = w[:, :, i, j, :, :]
            X_out[:, :, i, j] = torch.einsum('mnop,qnop->mq', X_sub, w_sub)

    return X_out

class TopLayer(nn.Module):
    """This class implements a topographic layer via a naive, baseline algorithm.
    
    It is a drop-in replacement of torch.nn.Conv2d
    The input has size (N x channels_in x h x h). 
    The output has size (N x channels_out x ((h + 1) // stride) x ((h + 1) // stride)).
    Each layer is implicitly zero-padded on the sides.
    This layer only works for square fields of neurons (ny = nx). 
    The pattern of connection is everything to everything.
    kernel_size is a user parameter.
    """
    def __init__(self, channels_in, channels_out, kernel_size, stride, h, bias=True):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.h = h
        self.bias_ = bias

        self.prepare_weights()

    def extra_repr(self):
        return f"channels_in={self.channels_in}, channels_out={self.channels_out}, kernel_size={self.kernel_size}, stride={self.stride}, h={self.h}"

    def prepare_weights(self):
        # variance = 2.0 / (fan-in +fan-out)
        # stddev = sqrt(variance)
        # weight = gaussian(mean=0.0, stddev)
        variance = 2.0 / (self.kernel_size ** 2 * (self.channels_in + self.channels_out))
        stddev = np.sqrt(variance)
        self.weights = nn.Parameter(stddev * torch.randn(self.channels_out, 
                                   self.channels_in, 
                                   self.h,
                                   self.h,
                                   self.kernel_size, 
                                   self.kernel_size))
        if self.bias_:
            self.bias = nn.Parameter(torch.zeros(self.channels_out))

        self.idxs0 = None
        self.idxs1 = None

    def forward(self, X, method='tiled'):
        # Pad X
        assert self.h == X.shape[2]
        assert X.shape[1] == self.channels_in
        assert self.kernel_size % 2 == 1
        pad_size = (self.kernel_size - 1) // 2
        

        if method == 'tiled':
            xs = next_multiple(X.shape[2]+2*pad_size, self.kernel_size)
        else:
            xs = X.shape[2] + 2*pad_size

        X_ = torch.zeros(X.shape[0], X.shape[1], xs, xs, device=X.device, dtype=X.dtype)
        rg = slice(pad_size, pad_size + X.shape[2])
        X_[:, :, rg, rg] = X

        # Now let's do the multiplication.
        if method == 'naive':
            X_out = torch.zeros(X.shape[0], self.channels_out, self.h, self.h, device=X.device, dtype=X.dtype)
            for i in range(self.h):
                for j in range(self.h):
                    # Pick a chunk of X
                    rgy = slice(i, i + self.kernel_size)
                    rgx = slice(j, j + self.kernel_size)

                    X_out[:, :, i, j] = torch.einsum('mnop,qnop->mq',
                        X_[:, :, rgy, rgx], 
                        self.weights[:, :, i, j, :, :])
        elif method == 'matmul':
            X_out = torch.zeros(X.shape[0], self.channels_out, self.h, self.h, device=X.device, dtype=X.dtype)
            for i in range(self.h):
                for j in range(self.h):
                    # Pick a chunk of X
                    rgy = slice(i, i + self.kernel_size)
                    rgx = slice(j, j + self.kernel_size)

                    X_out[:, :, i, j] = (X_[:, :, rgy, rgx].reshape(X_.shape[0], -1) @ 
                                         self.weights[:, :, i, j, :, :].reshape(self.weights.shape[0], -1).T)

        elif method == 'script':
            X_out = jitted_loop(X_, self.weights, self.kernel_size, self.h)
        elif method == 'tiled':
            # The naive methods do matrix multiplication one small piece of the image at a time.
            # However, filters are small, and they act locally, so we can batch multiple filters together.
            # With 5x5 filters, and a 60 pixel wide image, we can run 144 image patches in parallel.
            # To do the whole set of filters, we nudge the image by 1 (using roll), and run the appropriate filters. 
            # Thus, a whole image can be run in 25 passes.
            weights_padded = torch.zeros(self.weights.shape[0],
                                         self.weights.shape[1],
                                         next_multiple(self.weights.shape[2] + 2 * pad_size, self.kernel_size),
                                         next_multiple(self.weights.shape[3] + 2 * pad_size, self.kernel_size),
                                         self.weights.shape[4],
                                         self.weights.shape[5], device=self.weights.device, dtype=self.weights.dtype)

            rg = slice(0, self.weights.shape[2])
            weights_padded[:, :, rg, rg, :, :] = self.weights
            
            n = 0
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    # Shift                    
                    if not (i == 0 and j == 0):
                        if j != 0:
                            # Roll left
                            X_ = torch.roll(X_, shifts=-1, dims=3)
                        else:
                            # Carriage return
                            X_ = torch.roll(X_, shifts=[-1, self.kernel_size - 1], dims=[2, 3])
                    
                    #X_copy = torch.roll(X_.clone(), shifts=[-i, -j], dims=[2, 3])

                    # Pick just the right weights.
                    w_ = weights_padded[:, :, i::self.kernel_size, j::self.kernel_size, :, :]

                    # Now do the computation
                    X__ = X_.reshape(
                        X_.shape[0], X_.shape[1], 
                        X_.shape[2] // self.kernel_size, self.kernel_size, 
                        X_.shape[3] // self.kernel_size, self.kernel_size)
                    Xr = torch.einsum('ijklmn,pjkmln->ipkm', X__, w_)
                    if n == 0:
                        X_out = torch.zeros(*Xr.shape, self.kernel_size ** 2, dtype=X.dtype, device=X.device)
                    X_out[:, :, :, :, n] = Xr
                    n += 1

            X_out = X_out.reshape(
                X.shape[0], self.channels_out, X_.shape[2] // self.kernel_size, X_.shape[3] // self.kernel_size, 
                self.kernel_size, self.kernel_size
            ).permute(0, 1, 2, 4, 3, 5).reshape(X.shape[0], self.channels_out, X_.shape[2], X_.shape[3])
            X_out = X_out[:, :, :X.shape[2], :X.shape[3]]
        elif method == 'full':
            # Use a standard fully connected matrix product
            # X has shape batch_size, ndim, height, width
            # Hence, the create a matrix product of size [batch_size x (ndim, height, width)] x [(in_dim, height, width) x (out_dim, height, width)]
            W = torch.zeros(self.channels_in, X_.shape[2], X_.shape[3], self.channels_out, self.h, self.h, dtype=X.dtype, device=X.device)
            w_ = torch.permute(self.weights, (1, 4, 5, 0, 2, 3))

            for j in range(self.h):
                for i in range(self.h):
                    yrg_in = slice(j, j + self.kernel_size)
                    xrg_in = slice(i, i + self.kernel_size)
                    W[:, yrg_in, xrg_in, :, j, i] = w_[:, :, :, :, j, i]

            W = W.reshape(W.shape[0] * W.shape[1] * W.shape[2], W.shape[3] * W.shape[4] * W.shape[5])
            X_out = torch.mm(X_.reshape(X_.shape[0], -1), W)
            X_out = X_out.reshape(X_out.shape[0], self.channels_out, self.h, self.h)
        elif method == 'sparse':
            # Similar to full, but more annoying because one must do coordinate multiplication manually.
            w_ = torch.permute(self.weights, (1, 4, 5, 0, 2, 3))

            if self.idxs0 is None:

                I = (torch.arange(0, X_.shape[1] * X_.shape[2] * X_.shape[3]).reshape(
                    X_.shape[1], X_.shape[2], X_.shape[3]).to(dtype=X.dtype, device=X.device))
                J = (torch.arange(0, w_.shape[3]*w_.shape[4]*w_.shape[5]).reshape(w_.shape[3:])).to(dtype=X.dtype, device=X.device)

                idxs0 = []
                idxs1 = []
                for j in range(self.h):
                    for i in range(self.h):
                        yrg_in = slice(j, j + self.kernel_size)
                        xrg_in = slice(i, i + self.kernel_size)
                        idx0 = I[:, yrg_in, xrg_in]
                        idx0 = torch.stack([idx0] * w_.shape[3], dim=3)
                        idxs0.append(idx0.ravel())

                        idx1 = J[:, j, i]
                        idx1 = torch.stack([idx1] * (w_.shape[0] * w_.shape[1] * w_.shape[2]), dim=0)
                        idxs1.append(idx1.ravel())

                        assert idx0.numel() == idx1.numel()

                idxs0 = torch.cat(idxs0)
                idxs1 = torch.cat(idxs1)

                self.idxs0 = idxs0
                self.idxs1 = idxs1
            else:
                idxs0 = self.idxs0
                idxs1 = self.idxs1

            # Assemble into a big W matrix
            w_size = w_.shape[3]*w_.shape[4]*w_.shape[5]
            w_ = w_.permute((4, 5, 0, 1, 2, 3))
            WT = torch.sparse_coo_tensor(torch.stack([idxs1, idxs0], dim=0), w_.ravel(), (w_size, X_.shape[1] * X_.shape[2] * X_.shape[3]))
            X_T = X_.reshape(X_.shape[0], -1).T
            X_out = torch.sparse.mm(WT, X_T).T
            X_out = X_out.reshape(X_out.shape[0], self.channels_out, self.h, self.h)
        if method == 'blocksparse':
            # Similar to full, but more annoying because one must do coordinate multiplication manually.
            w_ = torch.permute(self.weights, (1, 4, 5, 0, 2, 3))

            I = (torch.arange(0, X_.shape[1] * X_.shape[2] * X_.shape[3]).reshape(
                X_.shape[1], X_.shape[2], X_.shape[3]).to(dtype=X.dtype, device=X.device))
            J = (torch.arange(0, w_.shape[3]*w_.shape[4]*w_.shape[5]).reshape(w_.shape[3:])).to(dtype=X.dtype, device=X.device)

            idxs0 = []
            idxs1 = []
            for j in range(self.h):
                for i in range(self.h):
                    yrg_in = slice(j, j + self.kernel_size)
                    xrg_in = slice(i, i + self.kernel_size)
                    idx0 = I[:, yrg_in, xrg_in]
                    idx0 = torch.stack([idx0] * w_.shape[3], dim=3)
                    idxs0.append(idx0.ravel())

                    idx1 = J[:, j, i]
                    idx1 = torch.stack([idx1] * (w_.shape[0] * w_.shape[1] * w_.shape[2]), dim=0)
                    idxs1.append(idx1.ravel())

                    assert idx0.numel() == idx1.numel()

            idxs0 = torch.cat(idxs0)
            idxs1 = torch.cat(idxs1)

            self.idxs0 = idxs0
            self.idxs1 = idxs1

            block_size = 8
            na = int(np.ceil(I.numel() / block_size))
            nb = int(np.ceil(J.numel() / block_size))

            idx = (torch.floor(idxs0 / 8) * nb + torch.floor(idxs1 / 8)).to(dtype=torch.long)

            sparsity_pattern = torch.zeros((na, nb), dtype=X_.dtype, device=X_.device)
            sparsity_pattern[idx] = 1

            from deepspeed.ops.sparse_attention.matmul import MatMul
            mm = MatMul(sparsity_pattern, block_size, 'dds')

            # Assemble into a big W matrix
            w_size = w_.shape[3]*w_.shape[4]*w_.shape[5]
            w_ = w_.permute((4, 5, 0, 1, 2, 3)).ravel()

            # Construct sub-matrices
            unique_idx = sorted(np.unique(idx))

            blocks = []

            for uid in unique_idx:
                # Block numbers
                ba, bb = uid // nb, uid % nb

                # Deltas within the blocks
                da, db = idxs0[idx == uid] - ba * block_size, idxs1[idx == uid] - bb * block_size
                the_block = torch.zeros((block_size, block_size), device=X_.device, dtype=X_.dtype)
                the_block[da * block_size + db] = w_[idx==uid]
                blocks.append(the_block)

            X_out = mm(X_.reshape(X_.shape[0], -1), blocks)
            X_out = X_out.reshape(X_out.shape[0], self.channels_out, self.h, self.h)

        if self.bias_:
            X_out += self.bias.reshape(1, -1, 1, 1)

        return X_out

if __name__ == '__main__':
    h = 24
    X = torch.randn(512, 8, h, h)
    layer = TopLayer(8, 32, 5, 1, h)

    # This is 10 times slower than cuda, no need to beat a dead horse
    # t0 = benchmark.Timer('layer(X)', globals={'X': X, 'layer': layer})
    # print(t0.timeit(5))

    # Now do it on the GPU
    layer = layer.to(device='cuda')
    X = X.to(device='cuda')

    # Test that gradient are calculated well
    import time
    t0 = time.time()
    X_out_tiled = layer(X, 'blocksparse')
    (X_out_tiled.sum()).backward()
    the_gradient = layer.weights.grad.clone()
    print(time.time() - t0)
    
    # Call once to amortize the index manipulation
    X_out_matmul = layer(X, 'tiled')
    layer.zero_grad()
    import time
    t0 = time.time()
    X_out_matmul = layer(X, 'tiled')
    (X_out_matmul.sum()).backward()
    the_gradient_matmul = layer.weights.grad.clone()
    print(time.time() - t0)

    print("Testing values")
    np.testing.assert_allclose(
        X_out_tiled.detach().cpu().numpy(), 
        X_out_matmul.detach().cpu().numpy(), rtol=0, atol=1e-5)

    print("Testing gradient")
    np.testing.assert_allclose(
        the_gradient.detach().cpu().numpy(), 
        the_gradient_matmul.detach().cpu().numpy(), rtol=0, atol=3e-5)

    print(abs(the_gradient).max())
    print(abs(the_gradient_matmul).max())

    # Do the same, with matmul
    #import time
    #t0 = time.time()
    #X_out_tiled = layer(X, 'tiled')
    #print(time.time() - t0)
    #print(X_out_tiled[0, 1, :6, :6])
    #
    #plt.imshow(X_out_tiled.detach().cpu().numpy()[0, 1, :, :])
    #plt.savefig('tiled.png')

    #X_out_naive = layer(X, 'matmul')
    #print(X_out_naive[0, 1, :6, :6])
    #
    #plt.imshow(X_out_naive.detach().cpu().numpy()[0, 1, :, :])
    #plt.savefig('naive.png')

    #np.testing.assert_allclose(
    #    X_out_tiled.detach().cpu().numpy(), 
    #    X_out_naive.detach().cpu().numpy(), rtol=0, atol=1e-5)

    #t0 = benchmark.Timer('layer(X, "matmul")', globals={'X': X, 'layer': layer})
    #print(t0.timeit(5))
    
    #t0 = benchmark.Timer('layer(X, "tiled")', globals={'X': X, 'layer': layer})
    #print(t0.timeit(5))

    """
    t0 = benchmark.Timer('layer(X, "script")', globals={'X': X, 'layer': layer})
    print(t0.timeit(5))

    t0 = benchmark.Timer('layer(X, "naive")', globals={'X': X, 'layer': layer})
    print(t0.timeit(5))
    """

