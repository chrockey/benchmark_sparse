import os
import time
import argparse
import numpy as np
from urllib.request import urlretrieve

import open3d as o3d
import torch

import MinkowskiEngine as ME
import spconv.pytorch as spconv
import torchsparse as tsp
import torchsparse.nn as nn


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_channels", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--mode", type=str, default="mink", choices=["mink", "spconv", "tsp", "o3d"])

    return parser.parse_args()


def get_conv(inc, outc, kernel_size, stride, mode="mink"):
    if mode == "mink":
        return ME.MinkowskiConvolution(
            inc, outc, kernel_size=kernel_size, stride=stride, dimension=3
        )
    elif mode == "spconv":
        return spconv.SparseConv3d(
            inc, outc, kernel_size=kernel_size, stride=stride
        ) # TODO: check out other algorithms
    elif mode == "tsp":
        return nn.Conv3d(inc, outc, kernel_size=kernel_size, stride=stride)
    else:
        raise NotImplementedError("Open3D is not supported yet.")


def get_sparse_tensor(coords, feats, batch_size, mode="mink"):
    if mode == "mink":
        return ME.SparseTensor(
            feats, coords, minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
        )
    elif mode == "spconv":
        spatial_shape = coords[:, 1:].max(dim=0)[0] - coords[:, 1:].min(dim=0)[0] + 1
        return spconv.SparseConvTensor(
            feats, coords, spatial_shape, batch_size
        )
    elif mode == "tsp":
        return tsp.SparseTensor(feats, coords, stride=1)
    else:
        raise NotImplementedError("Open3D is not supported yet.")


# Check if the weights and file exist and download
if not os.path.isfile("1.ply"):
    print("Downloading an example pointcloud...")
    urlretrieve("https://bit.ly/3c2iLhg", "1.ply")


def load_file(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    coords -= np.min(coords, axis=0, keepdims=True) # ME does not need this.
    colors = np.array(pcd.colors)
    return coords, colors, pcd


if __name__ == "__main__":
    cfg = get_config()
    batch_size = cfg.batch_size
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    coords, colors, pcd = load_file("1.ply")

    dcoords, dcolors = ME.utils.sparse_quantize(coords, colors, quantization_size=cfg.voxel_size)
    bcoords = ME.utils.batched_coordinates([dcoords for _ in range(batch_size)]).to(device)
    bcolors = torch.cat(
        [torch.from_numpy(dcolors).float() for i in range(batch_size)], 0
    ).to(device)

    # conv = get_conv(3, cfg.out_channels, kernel_size=cfg.kernel_size, stride=cfg.stride, mode=cfg.mode).to(device)
    conv = torch.nn.Sequential(
        get_conv(3, cfg.out_channels, kernel_size=cfg.kernel_size, stride=cfg.stride, mode=cfg.mode),
        get_conv(cfg.out_channels, cfg.out_channels, kernel_size=cfg.kernel_size, stride=cfg.stride, mode=cfg.mode),
        get_conv(cfg.out_channels, cfg.out_channels, kernel_size=cfg.kernel_size, stride=cfg.stride, mode=cfg.mode),
        get_conv(cfg.out_channels, cfg.out_channels, kernel_size=cfg.kernel_size, stride=cfg.stride, mode=cfg.mode),
        get_conv(cfg.out_channels, cfg.out_channels, kernel_size=cfg.kernel_size, stride=cfg.stride, mode=cfg.mode)
    ).to(device)

    ts = np.zeros(10)
    for i in range(10):
        c = time.time()
        stensor = get_sparse_tensor(bcoords, bcolors, batch_size=batch_size, mode=cfg.mode)
        output = conv(stensor)
        ts[i] = time.time() - c
    print(f"Forward Min time for {cfg.mode}: {np.min(ts)} for size {len(bcoords)} sparse tensor")

    # loss = output.features.sum() if hasattr(output, "features") else output.F.sum()
    # for i in range(10):
    #     c = time.time()
    #     loss.backward(retain_graph=True)
    #     ts[i] = time.time() - c
    # print(f"Backward Min time for {cfg.mode} {conv}: {np.min(ts)} for size {len(bcoords)} sparse tensor")