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
| Library         | Voxelization | Mixed Precision Training |
|:----------------|:------------:|:------------------------:|
| MinkowskiEngine | :white_check_mark: |  |
| SpConv          | :white_check_mark: | :white_check_mark: |
| TorchSparse     | :white_check_mark: | :white_check_mark: |


### Benchmark results
Each forward/backward time is the minimum time of 10 trials.
#### 1. Sample data (1,296,440 points) & Single convolution layer
| Library         | Forward Time (ms) | Backward Time (ms) |
|:----------------|------------------:|-------------------:|
| MinkowskiEngine | 14.879 | 10.764 |
| SpConv          |  8.764 |  0.492 |
| TorchSparse     | 29.397 |  7.705 |

#### 2. Sample data (1,296,440 points) & Cylindrical network (no down/up sampling)
The cylindrical network is a stack of eight conv-bn-relu blocks.
| Library         | Forward Time (ms) | Backward Time (ms) |
|:----------------|------------------:|-------------------:|
| MinkowskiEngine | 49.448 | 67.557 |
| SpConv          | 24.142 |  3.119 |
| TorchSparse     | 146.081 | 132.985 |

#### 3. ScanNetV2 & [MinkowskiNet42](https://openaccess.thecvf.com/content_CVPR_2019/papers/Choy_4D_Spatio-Temporal_ConvNets_Minkowski_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)
I measured the average time of training iterations (forward + backward) of MinkowskiNet42 on ScanNetV2 with voxel size of 2cm and batch size of 8.
| Library         | Average time (sec/iter) |
|:----------------|------------------------:|
| MinkowskiEngine | 2.51 |
| SpConv          | 2.40 |


### Conclusion
For now, the simple benchmark results show that SpConv is the fastest sparse conovlution library among MinkowskiEngine, SpConv, and TorchSparse.
However, the speed difference between MinkowskiEngine and SpConv becomes marginal when it comes to a more complex benchmark scenario (3D semantic segmentation on ScanNetV2).
Furthermore, MinkowskiEngine with [TensorField](https://github.com/NVIDIA/MinkowskiEngine/blob/master/MinkowskiEngine/MinkowskiTensorField.py) support is more flexible than SpConv and TorchSparse, which is a big advantage for researchers.
Therefore, I recommend using MinkowskiEngine for research purposes and SpConv for production purposes.
Any feedback is welcome!

### Todos
- [x] Benchmark results with a more complex network (e.g., UNet).
- [x] Benchmark the actual training time of the network on 3D semantic segmentation task.
- [ ] Add Open3D's sparse convolution.


### Acknowlegement
This repo heavily borrowed codes from [Chris Choy's benchmark code](https://gist.github.com/chrischoy/d8e971daf8308aa1dcba1524bf1fd91a).