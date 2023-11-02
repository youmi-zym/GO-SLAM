

<h1 align="center"> GO-SLAM <br>Global Optimization for Consistent 3D Instant Reconstruction <br> (ICCV 2023) </h1> 


:rotating_light: This repository contains the code and trained models of our work  "**GO-SLAM: Global Optimization for Consistent 3D Instant Reconstruction**",  [ICCV 2023](https://iccv2023.thecvf.com/)

by [Youmin Zhang](https://youmi-zym.github.io/), [Fabio Tosi](https://fabiotosi92.github.io/), [Stefano Mattoccia](http://www.vision.disi.unibo.it/smatt/) and [Matteo Poggi](https://mattpoggi.github.io/)

Department of Computer Science and Engineering (DISI),
University of Bologna


**Note**: 🚧 Kindly note that this repository is currently in the development phase.

<h4 align="center">
<ins>Code is available now, enjoy! </ins>
</h4>

<div class="alert alert-info">

<h2 align="center"> 

[Project Page](https://youmi-zym.github.io/projects/GO-SLAM/) | [Paper & Supplementary](https://arxiv.org/pdf/2309.02436.pdf) 
</h2>



<p align="center">
  <img src="./images/comparison.png" alt="3D Reconstruction Comparison" width="800" />
</p>

**3D Reconstruction and Trajectory Error**. From left to right: RGB-D methods ([iMAP](https://arxiv.org/abs/2103.12352), [NICE-SLAM](https://github.com/cvg/nice-slam), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), and ours), ground truth scan, and monocular methods (DROID-SLAM and ours).


## :bookmark_tabs: Table of Contents

1. [Introduction](#clapper-introduction)
2. [Code](#memo-code)
3. [Qualitative Results](#art-qualitative-results)
4. [Contacts](#envelope-contacts)

</div>


## :clapper: Introduction


We introduce **GO-SLAM**, a deep-learning-based dense visual SLAM framework that achieves **real-time global optimization of poses and 3D reconstruction**. By integrating robust pose estimation, efficient loop closing, and continuous surface representation updates, GO-SLAM effectively addresses the error accumulation and distortion challenges associated with neural implicit representations. Through the utilization of learned global geometry from input history, GO-SLAM sets new benchmarks in tracking robustness and reconstruction accuracy across synthetic and real-world datasets. Notably, its versatility encompasses **monocular**, **stereo**, and **RGB-D** inputs..

**Contributions:** 

* A novel deep-learning-based, **real-time global pose optimization system** that considers the complete history of input frames and continuously aligns all poses.

* An **efficient alignment strategy** that enables instantaneous loop closures and correction of global structure, being both memory and time efficient.

* An **instant 3D implicit reconstruction** approach, enabling on-the-fly and continuous 3D model update with the latest global pose estimates. This strategy facilitates real-time 3D reconstructions.

* The first deep-learning architecture for joint robust pose estimation and dense 3D reconstruction suited for any setup: **monocular**, **stereo**, or **RGB-D cameras**.

**Architecture Overview** 

GO-SLAM consists of three parallel threads: **front-end tracking**, **back-end tracking**, and **instant mapping**. It can run with monocular, stereo, and RGB-D input.

<img src="./images/framework.png" alt="Alt text" style="width: 800px;" title="architecture">





:fountain_pen: If you find this code useful in your research, please cite:

```bibtex
@inproceedings{zhang2023goslam,
    author    = {Zhang, Youmin and Tosi, Fabio and Mattoccia, Stefano and Poggi, Matteo},
    title     = {GO-SLAM: Global Optimization for Consistent 3D Instant Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
}
```

## :memo: Code

You can create an anaconda environment called `go-slam`. For linux, you need to install **libopenexr-dev** before creating the environment.
```bash

git clone --recursive https://github.com/youmi-zym/GO-SLAM

sudo apt-get install libopenexr-dev
    
conda env create -f environment.yaml
conda activate go-slam

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install evo --upgrade --no-binary evo

python setup.py install

```

### Replica

Download the data from [Google Drive](https://drive.google.com/drive/folders/1RJr38jvmuIV717PCEcBkzV2qkqUua-Fx?usp=sharing), and then you can run:

```bash
# please modify the OUT_DIR firstly in the script, and also DATA_ROOT in the config file
# MODE can be [rgbd, mono], EXP_NAME is the experimental name you want

./evaluate_on_replica.sh MODE EXP_NAME

# for example

./evaluate_on_replica.sh rgbd first_try

```

**Mesh and corresponding evaluated metrics are available in OUT_DIR.**

We also upload our predicted mesh on [Google Drive](https://drive.google.com/drive/folders/1RJr38jvmuIV717PCEcBkzV2qkqUua-Fx?usp=sharing). Enjoy!


### ScanNet
Please follow the data downloading procedure on [ScanNet](http://www.scan-net.org/) website, and extract color/depth frames from the `.sens` file using this [code](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).

<details>
  <summary>[Directory structure of ScanNet (click to expand)]</summary>
  
  DATAROOT is `./Datasets` by default. If a sequence (`sceneXXXX_XX`) is stored in other places, please change the `input_folder` path in the config file or in the command line.

```
  DATAROOT
  └── ScanNet
      └── scans
          └── scene0000_00
              └── frames
                  ├── color
                  │   ├── 0.jpg
                  │   ├── 1.jpg
                  │   ├── ...
                  │   └── ...
                  ├── depth
                  │   ├── 0.png
                  │   ├── 1.png
                  │   ├── ...
                  │   └── ...
                  ├── intrinsic
                  └── pose
                      ├── 0.txt
                      ├── 1.txt
                      ├── ...
                      └── ...

```
</details>

Once the data is downloaded and set up properly, you can run:
```bash
# please modify the OUT_DIR firstly in the script, and also DATA_ROOT in the config file
# MODE can be [rgbd, mono], EXP_NAME is the experimental name you want

./evaluate_on_scannet.sh MODE EXP_NAME

# for example

./evaluate_on_scannet.sh rgbd first_try

# besides, you can generate video as shown in our project page by:

./generate_video_on_scannet.sh rgbd first_try_on_video
```

We also upload our predicted mesh on [Google Drive](https://drive.google.com/drive/folders/1RJr38jvmuIV717PCEcBkzV2qkqUua-Fx?usp=sharing). Enjoy!

### EuRoC

Please use the following [script](https://github.com/youmi-zym/GO-SLAM/blob/main/scripts/download_euroc.sh) to download the EuRoC dataset. The GT trajectory can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1RJr38jvmuIV717PCEcBkzV2qkqUua-Fx?usp=sharing). 

Please put the GT trajectory of each scene to the corresponding folder, as shown below:


<details>
  <summary>[Directory structure of EuRoC (click to expand)]</summary>

DATAROOT is `./Datasets` by default. If a sequence (e.g., `MH_01_easy`) is stored in other places, please change the `input_folder` path in the config file or in the command line.

```
  DATAROOT
  └── EuRoC
     └── MH_01_easy
         └── mav0
             ├── cam0
             ├── cam1
             ├── imu0
             ├── leica0
             ├── state_groundtruth_estimate0
             └── body.yaml
         └── MH_01_easy.txt

```
</details>

Then you can run:

```bash
# for data downloading:

DATA_ROOT=path/to/folder
mkdir $DATA_ROOT
./scripts/download_euroc.sh $DATA_ROOT

# please modify the OUT_DIR firstly in the script, and also DATA_ROOT in the config file
# MODE can be [stereo, mono], EXP_NAME is the experimental name you want

./evaluate_on_euroc.sh MODE EXP_NAME

# for example

./evaluate_on_euroc.sh stereo first_try
```


## :art: Qualitative Results

In this section, we present illustrative examples that demonstrate the effectiveness of our proposal.


**Qualitative results on ScanNet dataset**. We evaluate our RGB-D mode SLAM using the ScanNet dataset and benchmark it against state-of-the-art techniques. Our method showcases improved global-consistency in reconstruction results.

<p float="left">
  <img src="./images/ScanNet.png" width="800" />
</p>

**Qualitative results on Replica dataset**. Supporting both Monocular and RGB-D modes, our GO-SLAM is evaluated on the Replica dataset. It achieves real-time, high-quality 3D reconstruction from monocular or RGB-D input. This stands in contrast to NICE-SLAM, designed solely for depth input, which operates at a frame rate of less than 1 per second and requires hours to achieve comparable outcomes.

<p float="left">
  <img src="./images/Replica.png" width="800" />
</p>

**Qualitatives examples of LC and full BA on scene0054 00 (ScanNet) with a total of 6629 frames.** . In (a), a significant error accumulates when no global optimization is available. With loop closing (b), the system is able to eliminate the trajectory error using global geometry. Additionally, online full BA optimizes (c) the poses of all existing keyframes. The final model (d), which integrates both loop closing and full BA, achieves a more complete and accurate 3D model prediction.

<p float="left">
  <img src="./images/LC_and_full_BA.png" width="800" />
</p>



## :envelope: Contacts

For questions, please send an email to youmin.zhang2@unibo.it, fabio.tosi5@unibo.it or m.poggi@unibo.it


## :pray: Acknowledgements

We sincerely thank the scholarship supported by China Scholarship Council (CSC). 

We adapted some codes from some awesome repositories including [NICE-SLAM](https://github.com/cvg/nice-slam), [NeuS](https://github.com/Totoro97/NeuS) and [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM).