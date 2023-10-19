# Offline Knowledge Distillation 
A 3DCNN-Based Knowledge Distillation Framework for Human Activity Recognition [paper](https://www.mdpi.com/2313-433X/9/4/82)

<img src="readme_images/framework.gif" width="800"/>

## Introduction
This repo contains the implementation of our proposed 3DCNN based knowledge distillation approach for human activity recognition, the prerequisite libraries, and the obtained quantiative results across different human activity recognition datasets.  

## Installation
This code is written in python 3.8. Install Anaconda python 3.8 and clone the repo using the following command:
```
git clone https://github.com/hayatkhan8660-maker/Response-based-offline-knowledge-distillation-for-HAR
```
## Prerequisites
### Recommended Environment
- Python 3.8
- Tensorflow 2.7
- Keras 2.7
  
### Dependencies
The following libraries need to be installed
- numpy == 1.24.4
- sklearn == 1.3.0
- scikitplot == 0.3.7
- matplotlib == 3.7.3
- tensorboard == 2.14.0
- scipy == 1.10.1

run ```pip install -r requirements.txt``` to install all the dependencies.

### Data Preparation
We have conducted experiments on four HAR datasets, including UCF11, UCF50, HMDB51, and UCF101 datasets. 
- UCF11 [dataset link](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php)
- UCF50 [dataset link](https://www.crcv.ucf.edu/data/UCF50.php)
- HMDB51 [dataset link](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#dataset)
- UCF101 [dataset link](https://www.crcv.ucf.edu/data/UCF101.php)

Before training, we have converted each video dataset to frames data using ```video_to_frames.py``` file as follows: 
```
python video_to_frames.py --input_path Datasets/dataset_name/ --sequence_length 16 --frame_height 224 --frame_width 224 --output_path Output_dataset/dataset_name/
``` 
The above ```video_to_frames.py``` file will generate five files. 1) ```Format_Time.csv``` listing amount of time and memory taken for frames extraction processes per class. 2) ```Format_Time_Summary.txt``` provide the runtime history (in terms of time and memoray allocation) of frames extraction process. 3) ```frames.npy``` contains video frames for the entire dataset (with a skip_frames_window = total_num_frames/sequence_length). 4) ```labels.npy``` contains numerical labels of each video. 5) ```labels.npy``` contains the path of all videos of a given dataset.
```
Frames Dataset
├── UCF11
│   ├── Format_Time.csv
│   ├── Format_Time_Summary
|   ├── frames.npy
|   ├── labels.npy
|   ├── video_file_paths.npy
├── UCF50
│   ├── Format_Time.csv
│   ├── Format_Time_Summary
│   ├── frames.npy
│   ├── labels.npy
│   ├── video_file_paths.npy
├── HMDB51
│   ├── Format_Time.csv
│   ├── Format_Time_Summary
│   ├── frames.npy
│   ├── labels.npy
│   ├── video_file_paths.npy
├── UCF101
│   ├── Format_Time.csv
│   ├── Format_Time_Summary
│   ├── frames.npy
│   ├── labels.npy
│   ├── video_file_paths.npy
```
Note: Sequence length, frames width, and frame height are subjective, one can choose different values for these arguements. In our work we used (sequence length = 16), (frame height = 224), and (frame width = 224). 
## Citation
Please cite our paper, if you want to reproduce the results using this code.
```
@article{ullah20233dcnn,
  title={A 3DCNN-Based Knowledge Distillation Framework for Human Activity Recognition},
  author={Ullah, Hayat and Munir, Arslan},
  journal={Journal of Imaging},
  volume={9},
  number={4},
  pages={82},
  year={2023},
  publisher={MDPI}
}
```

```
Ullah, H., & Munir, A. (2023). A 3DCNN-Based Knowledge Distillation Framework for Human Activity Recognition.
Journal of Imaging, 9(4), 82.
```
