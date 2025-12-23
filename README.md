# Data collection & Processing tools

This repository contains complete pipeline for processing raw data collected from Jackal mobile robots using ROS2 bag package.

We use this tool to collect Expert/Negative data of LiDAR (Livox Mid-360) and other from sensors in warehouse sites. Ultimately, we aim to utilize the dataset to recognize abnormal behavior of robots in warehouse environments.


### Scenario Taxonomy



### Methodology



### Definition of state-action

![state-action](https://latex.codecogs.com/svg.image?\dpi{120}\mathbf{s}_t:=\begin{bmatrix}v_{[t-k:t]}\\a_{[t-k:t]}\\\omega_{[t-k:t]}\\p_{[t-k:t]}^{\mathrm{obj}}\\d_{[t-k:t]}\end{bmatrix}\qquad\mathbf{a}_t:=\begin{bmatrix}a_{[t]}\\\omega_{[t]}\end{bmatrix})



### How to use

processing raw data

    python data_parser.py --src_dir ./Data --interp_method linear   # or, 'poly' 'spline'
    python data_processor.py --src_dir ./Data

training

    python main.py --frame 1 --id 1   # MDN
    python main.py --frame 1 --mode vae --h_dim 128 --z_dim 32 --id 1   # VAE