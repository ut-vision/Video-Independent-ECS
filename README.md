# Learning Video-independent Eye Contact Segmentation from In-the-wild Videos

This repository contains the official implementation of our [paper](https://arxiv.org/abs/2210.02033) *Learning Video-independent Eye Contact Segmentation from In-the-wild Videos* (ACCV2022) and also the evaluation annotation on 52 selected videos from VoxCeleb2 dataset.

If you have some requests or questions, please contact the first author.

Results on our annotated videos from VoxCeleb2 dataset:
![result_on_VoxCeleb2_test_tracklets](./example/result.gif)

Results on random street interview videos from Youtube:
![result_on_street_interview](./example/result2.gif)

## Environment Configuration

Please refer to requirements.txt. Otherwise for conda virtual environments, please see conda_env.yml.

We also provide dockerfile used for running the code.

## Preparations

### Appearance-based Gaze Eestimation

Please first download the pretrained [ETH-XGaze model](https://github.com/xucong-zhang/ETH-XGaze) and place the ckpt into ./modules

### Face Detection and Facial Landmark Detection

Following ETH-XGaze, we rely on dlib for face detection and landmark detection.

Please download the following from [dlib model library](https://github.com/davisking/dlib-models) and place them into ./modules
- mmod_human_face_detector.dat
- shape_predictor_68_face_landmarks_GTX.dat

### Face Recognition and Pose Estimation

Our tracking algorithm is bulit on face recognition and pose estimation. You can choose any pretrained face recognition and pose estimation models.

You need to implement encode_face() and compute_pose() in the tools.py and modify 

- generate_face_embedding.py L83
- trackletFormation.py L158, L207-L220

In our case, we used [ArcFace_r100_fp16](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) for face recognition and [here](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation) for pose estimation. Note that face recognition and pose estimation are only required for training but not inference.

## Random VoxCeleb2 Dataset

We provide code to download the raw videos we randomly sampled from VoxCeleb2 Dataset in **randomVox** folder.

Note that the above docker image is not designed for downloading dataset here. Please install yt-dlb and ffmpeg.

To download the raw videos, run

`cd randomVox`

`python download.py`


After downloading as much videos as you can, you should convert all videos to 25FPS in mkv by


```
python preprocess.py --convert
python preprocess.py --clean
```

Note that the actual videos we use for training is a subset of the downloaded videos. (only uses videos from 2 mins to 12 mins)

The final dataset should have a folder structure:

```
RootFolderName
| - randomVox
    | - id00016 (Each folder here contains xxx_25fps.mkv videos)
    | - id00018
    | ...
    | ...
    | ...
    | - id09272
| - camera
| - modules
| ...
| ...
| ...
| demo.py
```

## Evaluation dataset

We annotated 52 videos from VoxCeleb2 Dataset for one-way eye contact segmentation. 

Video-level annotations and tracklets used for evaulation are available under ./Annotation.

## Training

After downloading the dataset, you also need the face images from VGGCeleb2 dataset. Please download VGGFace2 dataset and place it in the root. You will also be able to obtain a csv that maps VoxCeleb2 celebrity IDs into VGGFace2 celebritiy IDs. Please rename this csv file to vox_face_conversion.csv and place it into the root folder.

Then, run 

`python generate_face_embedding.py`. 

This generates face embeddings for all celebrity IDs, which will be used for face tracking.

`python trackletFormation.py` 

(This is going to take long. Roughly 2 months depending on the hardware (both CPU and GPU).
 
This generates face tracklets in .npz format. Each tracklet contain: start_frame, bounding_box locations, facial landmarks, and gaze features. It also records framewise gaze points in the csv file used later for gaze target discovery.

`python training/pseudo_label_generator.py --run_train --m_eps 8 --associate`

This clusters the gaze points in the cylinder plane, results in framewise pseudo-labels, and associate pseudo-labels with each tracklet in .npz

`python training/create_dataset.py --make_training`

This generates a tracklet meta file for the dataloader.

Please then go to TCN folder and adjust the dataloader path accordingly, and run

```
python training/ECSTrainer.py --mse (This gives iteration 0 model)
python training/ECSTrainer.py --stack --fintune --mse (This gives iteration >= 1 models)
```

or simply 

`python training/ECSTrainer.py --stack --mse`

The first way allows you to manually pick the intermediate model.

But before training, one might want to first convert test set segmentation labels for videos to test set tracklets framewise labels.

Please do following, 

```
python training/pseudo_label_generator.py --gt
python training/create_dataset.py --make_testing
```

## Inference

Note that inference does not require the implementation of functions in the tools.py.

You simply need the dlib pretrained weights, ETH-XGaze pretrained weights on ResNet50 and our pretrained weights.

During inference, our model does not require face recognition and pose estimation. However, for videos containing multiple people, it is essential to track faces. Depending on the use case, tracking algorithm of different complexity should be used. In the most simple cases, a tracking algorithm simply based on IoU of face bounding boxes can be adopted.

We provide an inference demo for a single-person video from our test tracklets.

Please first download our pretrained weights and place it to ./modules.

Then run

`python demo.py`

You will get a processed video under ./examples

If you see the following messages, you are doing it correctly.

```
Loading resources...
load gaze estimator
load the pre-trained model:  ./modules/epoch_24_ckpt.pth.tar
Video is 720p
Extracting Gaze Features...
Predicted:
[0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0
 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1
 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0
 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1]
writing as a video
Done.
```
We also provide a demo with a simple face tracker based on IoU of bounding boxes in demo_with_tracking.py. You can simply run by specifying input and output path

`python demo_with_tracking.py --input examples/multiperson.mkv --output examples/multiperson_out.mkv --iou 0.4`

Note that our pretrained model is trained using unsupervised approach on large-scale conversation videos.

It works best on
- conversation videos (interview, speaking to the camera)
- frontal faces with most of the attention drawn to the gaze target

It fails severely on
- short profile face tracklets 


