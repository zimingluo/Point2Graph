## [Point2Graph: An End-to-end Point Cloud-based 3D Open-Vocabulary Scene Graph for Robot Navigation](https://arxiv.org/abs/2409.10350)

This is the implementation of **Object Detection and Classification** module of the paper "Point2Graph: An End-to-end Point Cloud-based 3D Open-Vocabulary Scene Graph for Robot Navigation". 

Authors: [Yifan Xu](https://www.linkedin.com/in/yifan-xu-43876120b/), [Ziming Luo](https://zimingluo.github.io/), [Qianwei Wang](https://www.linkedin.com/in/qianwei-wang-945bb9292/), [Vineet Kamat](https://live.engin.umich.edu/), [Carol Menassa](https://cee.engin.umich.edu/people/menassa-carol-c/)

## News:

[2025/02] Our paper is accepted by ICRA2025 🎉🎉🎉

## Object Detection and Classification Pipeline

This module consists of two stages: (1) detection and localization using class-agnostic bounding boxes and DBSCAN filtering for object refinement, and (2) classification via cross-modal retrieval, connecting 3D point cloud data with textual descriptions, without requiring annotations or RGB-D alignment.

![Pipeline Image](https://point2graph.github.io/static/figure/object_pipeline.png)



## Getting Started

### Installation

**Step 1.** Create a conda environment and activate it.

```shell
conda env create -f point2graph.yaml
conda activate point2graph
```

**Step 2.** install **Minkowski Engine**.

```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

**Step 3.** install **mmcv**.

```bash
pip install openmim
mim install mmcv-full==1.6.1
```

**Step 4.** install third party support.

```bash
cd third_party/pointnet2/ && python setup.py install --user
cd ../..
cd utils && python cython_compile.py build_ext --inplace
cd ..
```

### Dataset preparation

**Scannet Data**

1. Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Move/link the `scans` folder such that under `scans` there should be folders with names such as `scene0001_01`.
2. Open the 'scannet' folder. Extract point clouds and annotations (semantic seg, instance seg etc.) by running `python batch_load_scannet_data.py`, which will create a folder named `scannet_train_detection_data` here.

### Model preparation

You should 

* download the 3D Object Detection  pre-trained model [V-DETR](https://huggingface.co/byshen/vdetr/blob/main/scannet_540ep.pth), and put it in `./models/` folder.
* download the 3D Object Classification pre-trained model  [Uni-3D](https://github.com/baaivision/Uni3D#model-zoo) and the [clip model](https://huggingface.co/timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k/blob/main/open_clip_pytorch_model.bin), and put them in `./Uni3D/downloads/` folder.


## Testing

The test script is in the `run.sh` file. Once you have the datasets and model prepared, you can test this models as

```shell
bash run.sh
```

The script performs two functions:

1. Get a set of point cloud of objects with unknown class and store them at `./results/objects/`
2. Retrieve and visualize the 3D object point cloud most relevant to the user's query

## Acknowledgement

Point2Graph is built on the [V-DETR](https://github.com/V-DETR/V-DETR), and [Uni3D](https://github.com/baaivision/Uni3D).


## Citation

If you find this code useful in your research, please consider citing:

```
@misc{xu2024point2graphendtoendpointcloudbased,
      title={Point2Graph: An End-to-end Point Cloud-based 3D Open-Vocabulary Scene Graph for Robot Navigation}, 
      author={Yifan Xu and Ziming Luo and Qianwei Wang and Vineet Kamat and Carol Menassa},
      year={2024},
      eprint={2409.10350},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.10350}, 
}
```
