# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import, division, print_function

import argparse
import os
import numpy as np
import timeit

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import mkldnn as mkldnn_utils

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=2,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

# Args for architecture
parser.add_argument('--ndims', type=int, default=2, help='Specifies dimensionality of convolution, e.g. ndims=3 will test 3D convoltuion')
parser.add_argument('--in-channels', type=int, default=32, help='Specifies number of input channels for the single conv layer')
parser.add_argument('--out-channels', type=int, default=32, help='Specifies number of output channels for the single conv layer')
parser.add_argument('--kernel-size', type=int, default=3, help='Specifies the kernel size for the conv layer')
parser.add_argument('--ngroups', type=int, default=1, help='Specifies the number of groups for the conv layer')

# Args for input size
parser.add_argument('--input-dim', nargs="+", type=int, help='Specifies dimensionality of the input. Takes multiple arguments, e.g. --input_dim 256 256 100 indicates a 3D input of (256, 256, 100)')

# Args for output file
parser.add_argument('--output', type=str, default='output.txt', help="Specify output file")

args = parser.parse_args()

# Make sure that only rank 0 prints if Horovod is used
def log(s, nl=True):
    print(s, end='\n' if nl else '')

# Decide on which device is used
args.cuda = not args.no_cuda
dev = 'cuda' if ( torch.cuda.is_available() and args.cuda ) else 'cpu'
if dev == 'cuda':
    torch.backends.cudnn.benchmark = True

# Process input shape
input_dim = list(args.input_dim)

# Check if IPEX should be used
if args.use_ipex:
    import intel_pytorch_extension as ipex

#if args.mixed_prec:
#    log('Running with mixed_float16 as global policy for the precision')
#    mixed_precision.set_global_policy('mixed_float16')

# Report on the key input parameters
log(f'PyTorch using device: {dev}')
log(f'Running with batch size: {args.batch_size}')
log(f"Number of input channels: {args.in_channels}")
log(f"Number of output channels: {args.out_channels}")
log(f"Kernel size: {args.kernel_size}")
log(f"Groups: {args.ngroups}")
log(f"Input size: {input_dim}")

# Setup PyTorch model
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        if args.ndims == 2:
            self.conv = nn.Conv2d(in_channels = args.in_channels, out_channels = args.out_channels, kernel_size = args.kernel_size, groups = args.ngroups, padding=args.kernel_size//2)
        elif args.ndims == 3:
	    # Custom implementation for depthwise: try to split in PyTorch
            if args.in_channels == args.ngroups and args.in_channels == args.out_channels:
                self.conv = nn.ModuleList([nn.Conv3d(in_channels = 1, out_channels = 1, kernel_size = args.kernel_size, padding=args.kernel_size//2) for i in range(args.in_channels)])
                print(f'self.conv: {self.conv}')
            else:
                self.conv = nn.Conv3d(in_channels = args.in_channels, out_channels = args.out_channels, kernel_size = args.kernel_size, groups = args.ngroups, padding=args.kernel_size//2)

    def forward(self, x):
        # Custom implementation of depthwise: try to split in PyTorch, then call conv on each channel, then merge
        if args.ndims == 3 and args.in_channels == args.ngroups and args.in_channels == args.out_channels:
            x = torch.chunk(x, args.in_channels, 1) # dim 1 = channel dim
            print(f'Chuncked x: {len(x)}, x[0].shape: {x[0].shape}')
            x = [conv(xt) for conv, xt in zip(self.conv, x) ]
            x = torch.cat(x, 1) # dim 1 = channel dim
        else:
            x = self.conv(x)
        return x

# Create PyTorch model instance
if args.use_mkl:
    model_torch = mkldnn_utils.to_mkldnn(MyNet())
elif args.use_ipex:
    model_torch = MyNet().to(ipex.DEVICE)
else:
    model_torch = MyNet().to(dev)
model_torch.train() # Set in training mode, we don't want to do this in the benchmark step
print(model_torch)

# INPUT & TARGET
log('Creating synthetic input & output data')
X_dim = [args.batch_size, args.in_channels ] + input_dim
Y_dim = X_dim
X = torch.rand(*X_dim)
Y = torch.rand(*Y_dim)

if args.use_mkl:
    X = X.to_mkldnn()
    Y = Y.to_mkldnn()
elif args.use_ipex:
    X = X.to(ipex.DEVICE)
    Y = Y.to(ipex.DEVICE)
else:
    X = X.to(dev)
    Y = Y.to(dev)

# LOSS
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-6

# OPTIMIZER
optimizer = torch.optim.RMSprop(model_torch.parameters(), lr = learning_rate)

def benchmark_step():
#    print('Run benchmark step')
    out = model_torch(X)
    loss = loss_fn(out, Y)
#    print(f"Loss: {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if dev == 'cuda':
        torch.cuda.synchronize()
    return out


# Warm-up
log('Running warmup...')

timeit.timeit(lambda: benchmark_step(),
              number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
for x in range(args.num_iters):
    if dev == 'cuda':
       torch.cuda.synchronize()
    time = timeit.timeit(lambda: benchmark_step(),
                         number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time
    log(f'Iter #{x}: {img_sec:.1f} img/sec per rank')
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log(f'Img/sec per rank: {img_sec_mean:.1f} +- {img_sec_conf:.1f}')
if args.use_horovod:
    ndevices = hvd.size()
else:
    ndevices = 1
log(f'Total img/sec on {ndevices} ranks: {ndevices*img_sec_mean:.1f} +- {ndevices * img_sec_conf:.1f}')

# Writing output
log(f'Writing output file: {args.output}')
output = f"{ndevices} {args.batch_size} {args.in_channels} {args.out_channels} {args.kernel_size} {args.ngroups}"
for i in range(len(input_dim)):
    output = output + f" {input_dim[i]}"
output = output + f" {img_sec_mean}\n"

log("Writing output line:")
log(output)

with open(args.output, "a") as f:
    f.write(output)
    f.close()

log('Benchmark completed')
