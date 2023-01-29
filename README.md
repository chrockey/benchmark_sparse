# Benchmark of sparse convolution libraries
Benchmarking various sparse convolution libraries: MinkowskiEngine, SpConv, TorchSparse, and Open3D.


### Environments
- Ubuntu 20.04
- CUDA 11.3
- PyTorch 1.12.1
- SpConv v2.3.2
- MinkowskiEngine v0.5.4
- TorchSparse v1.4.0

### Comparsion
- tf32 support
- mixed precision support

### Benchmark results


### todos
- [ ] CUDA 11.4: According to [SpConv news](https://github.com/traveller59/spconv), SpConv with CUDA >=11.4 is much faster than CUDA <11.4.
- [ ] Benchmark results on real datasets (e.g. ScanNet, S3DIS, ShapeNet, ModelNet).