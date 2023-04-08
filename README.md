## Benchmark of sparse convolution libraries
Benchmarking various sparse convolution libraries: MinkowskiEngine, SpConv, TorchSparse, and Open3D.


### Environments
- A6000 GPU
- Ubuntu 22.04
- CUDA 11.7
- PyTorch 2.0.0
- SpConv v2.3.5
- MinkowskiEngine v0.5.4
- TorchSparse v1.4.0
- libsparsehash-dev # apt-get install libsparsehash-dev


### Installation
```bash
~/benchmark_sparse$ conda create -n bench python=3.8 -y
~/benchmark_sparse$ conda activate bench
(bench) ~/benchmark_sparse$ pip install torch ninja open3d
(bench) ~/benchmark_sparse$ pip install spconv-cu117
(bench) ~/benchmark_sparse$ pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                                                                                            --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
                                                                                            --install-option="--blas=openblas"
(bench) ~/benchmark_sparse$ pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

### Comparsion
- tf32 support
- mixed precision support

### Benchmark results


### Todos
- [ ] Benchmark results with a more complex networks (e.g., UNet).
- [ ] Benchmark the actual training time of ResNet, using each library.
- [ ] Add Open3D's sparse convolution.


### Acknowlegement
This repo heavily borrowed codes from [Chris Choy's benchmark code](https://gist.github.com/chrischoy/d8e971daf8308aa1dcba1524bf1fd91a).