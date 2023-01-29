## Benchmark of sparse convolution libraries
Benchmarking various sparse convolution libraries: MinkowskiEngine, SpConv, TorchSparse, and Open3D.


### Environments
- A6000 GPU
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
- [ ] Benchmark the actual training time of ResNet, using each library.
- [ ] Add Open3D.


### Acknowlegement
This repo heavily borrowed codes from [Chris Choy's benchmark code](https://gist.github.com/chrischoy/d8e971daf8308aa1dcba1524bf1fd91a).