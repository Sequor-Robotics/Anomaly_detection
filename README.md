# Anomaly detection

This repository contains complete pipeline for processing raw data collected from Jackal mobile robots using ROS2 bag package.

We use this tool to collect Expert/Negative data of LiDAR (Livox Mid-360) and other from sensors in warehouse sites. Ultimately, we aim to utilize the dataset to recognize abnormal behavior of robots in warehouse environments.

### How to use

Clone repository

    git clone https://github.com/Sequor-Robotics/Anomaly_detection.git

Setup environment (Python=3.10.19)

    cd ~/Anomaly_detection_Sequor_TIPS
    pip install -r requirements.txt

Download dataset

    git clone https://github.com/Sequor-Robotics/Anomaly_detection_dataset.git

processing raw data (if needed)

    python processor.py --src ./Data/{scenario_id}                                             # or
    python processor.py --src ./Data/{scenario_id}/{trial_no}                                  # or
    python processor.py --src ./Data/{scenario_id}/{trial_no}/{scenario_id}_{trial_no}_0.mcap

labelling (if needed)

    python label_tool.py

training & show results

    python main.py --frame 10 --mode vae --h_dim 256 --z_dim 64    # VAE
    python main.py --frame 10 --mode mdn                           # MDN