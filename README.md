# Everybody Dance Now

### [[website]](https://carolineec.github.io/everybody_dance_now/) [[paper]](https://arxiv.org/pdf/1808.07371.pdf) [[youtube]](https://www.youtube.com/watch?v=PCBTZh41Ris)

Colab Version by Stanley Shen, and minor bug fixes plus port for openpose preprocessing scripts to Python 3

Original Implementation:  
Everybody Dance Now  
Caroline Chan, Shiry Ginosar, Tinghui Zhou, Alexei A. Efros  
UC Berkeley  
hosted on arXiv

## Prerequisites
1. [PyTorch](https://pytorch.org/)
2. Python Library [Dominate](https://github.com/Knio/dominate)
```
pip install dominate
```
3. Clone this repository
```
git clone https://github.com/carolineec/EverybodyDanceNow
```

I ran my code on Colab, and the only the T4 Tesla Accelerator has enough Vram to run this project. The below command are copied from the original author's repo, but I recommend running the cells in the Notebooks instead. Please don't just run all the cells, instead, please check with what they do first before running.

First, open the Openpose Notebook and install openpose, then convert the video to frames with ffmpeg and use the openpose demo to create json keypoint files(This is around 2 cells).
Then, use either the graph_train and graph_avesmooth cell to convert the frames and keypoints to images and labels for pix2pixHD.
Then train in the EverybodyDanceNow notebook with either the local/local+face_gan model, and stop when you are satisifed with the ouputs.
Then convert the test data with the Openpose Notebook, and go back and inference with the EveryBodyDanceNow Notebook. 
## Training

#### Global Stage
We follow similar stage training as in [pix2pixHD](https://github.com/NVIDIA/pix2pixHD). We first train a "global" stage model at 512x256 resolution
```
# train a model at 512x256 resolution
python train_fullts.py \
--name MY_MODEL_NAME_global \
--dataroot MY_TRAINING_DATASET \
--checkpoints_dir WHERE_TO_SAVE_CHECKPOINTS \
--loadSize 512 \
--no_instance \
--no_flip \
--tf_log \
--label_nc 6
```

#### Local Stage
Followed by a "local" stage model with 1024x512 resolution.
```
# train a model at 1024x512 resolution
python train_fullts.py \
--name MY_MODEL_NAME_local \
--dataroot MY_TRAINING_DATASET \
--checkpoints_dir WHERE_TO_SAVE_CHECKPOINTS \
--load_pretrain MY_MODEL_NAME_global \
--netG local \
--ngf 32 \
--num_D 3 \
--resize_or_crop none \
--no_instance \
--no_flip \
--tf_log \
--label_nc 6
```

#### Face GAN stage
We then can apply another stage with a separate GAN focused on the face region.
```
# train a model specialized to the face region
python train_fullts.py \
--name MY_MODEL_NAME_face \
--dataroot MY_TRAINING_DATASET \
--load_pretrain MY_MODEL_NAME_local \
--checkpoints_dir WHERE_TO_SAVE_CHECKPOINTS \
--face_discrim \
--face_generator \
--faceGtype global \
--niter_fix_main 10 \
--netG local \
--ngf 32 \
--num_D 3 \
--resize_or_crop none \
--no_instance \
--no_flip \
--tf_log \
--label_nc 6
```

## Testing

The full checkpoint will be loaded from --checkpoints_dir/--name (i.e. if flags: "--name foo \ ... --checkpoints_dir bar \"" are included, checkpoints will be loaded from foo/bar)
Replace --howmany flag with an upper bound on how many test examples you have

#### Global Stage
```
# test model at 512x256 resolution
python test_fullts.py \
--name MY_MODEL_NAME_global \
--dataroot MY_TEST_DATASET \
--checkpoints_dir CHECKPOINT_FILE_LOCATION \
--results_dir WHERE_TO_SAVE_RESULTS \
--loadSize 512 \
--no_instance \
--how_many 10000 \
--label_nc 6
```

#### Local Stage
```
# test model at 1024x512 resolution
python test_fullts.py \
--name MY_MODEL_NAME_local \
--dataroot MY_TEST_DATASET \
--checkpoints_dir CHECKPOINT_FILE_LOCATION \
--results_dir WHERE_TO_SAVE_RESULTS \
--netG local \
--ngf 32 \
--resize_or_crop none \
--no_instance \
--how_many 10000 \
--label_nc 6
```

#### Face GAN stage
```
# test model at 1024x512 resolution with face GAN
python test_fullts.py \
--name MY_MODEL_NAME_face \
--dataroot MY_TEST_DATASET \
--checkpoints_dir CHECKPOINT_FILE_LOCATION \
--results_dir WHERE_TO_SAVE_RESULTS \
--face_generator \
--faceGtype global \
--netG local \
--ngf 32 \
--resize_or_crop none \
--no_instance \
--how_many 10000 \
--label_nc 6
```

## Dataset preparation
We also provide code for creating both training and testing datasets (including global pose normalization) in the **data_prep** folder. See the **sample_data** folder for examples on how to prepare the code for training. Note the original_img is not necessary at test time and is provided only for reference.

Our dataset preparation code is based on output formats from [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and currently supports the COCO, BODY_23, and BODY_25 pose output format as well as hand and face keypoints. To install and run OpenPose please follow the directions at the [OpenPose repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

### graph_train.py
will prepare a train dataset with subfolders
- train_label (contains 1024x512 inputs)
- train_img (contains 1024x512 targets)
- train_facetexts128 (contains face 128x128 bounding box coordinates in .txt files)
No smoothing
```
python3 graph_train.py \
--keypoints_dir /data/scratch/caroline/keypoints/jason_keys \
--frames_dir /data/scratch/caroline/frames/jason_frames \
--save_dir /data/scratch/caroline/savefolder \
--spread 4000 25631 1 \
--facetexts
```

### graph_avesmooth.py
will prepare a dataset with averaged smoothed keypoints with subfolders (usually for validation)
- test_label (contains 1024x512 inputs)
- test_img (contains 1024x512 targets)
- test_factexts128 (contains face 128x128 bounding box coordinates in .txt files)
```
python3 graph_avesmooth.py \
--keypoints_dir /data/scratch/caroline/keypoints/wholedance_keys \
--frames_dir /data/scratch/caroline/frames/wholedance \
--save_dir /data/scratch/caroline/savefolder \
--spread 500 29999 4 \
--facetexts
```

### graph_posenorm.py
will prepare a dataset with global pose normalization + median smoothing
- test_label (contains 1024x512 inputs)
- test_img (contains 1024x512 targets)
- test_factexts128 (contains face 128x128 bounding box coordinates in .txt files)
```
python3 graph_posenorm.py \
--target_keypoints /data/scratch/caroline/keypoints/wholedance_keys \
--source_keypoints /data/scratch/caroline/keypoints/dubstep_keypointsFOOT \
--target_shape 1080 1920 3 \
--source_shape 1080 1920 3 \
--source_frames /data/scratch/caroline/frames/dubstep_frames \
--results /data/scratch/caroline/savefolder \
--target_spread 30003 178780 \
--source_spread 200 4800 \
--calculate_scale_translation
--facetexts
```

## Citation

If you find this work useful please use the following citation:

```
@inproceedings{chan2019dance,
 title={Everybody Dance Now},
 author={Chan, Caroline and Ginosar, Shiry and Zhou, Tinghui and Efros, Alexei A},
 booktitle={IEEE International Conference on Computer Vision (ICCV)},
 year={2019}
}
```

## Acknowledgements

Model code adapted from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 

Original code from [Caroline Chan's Repo](https://github.com/carolineec/EverybodyDanceNow}

Data Preparation code adapted from [Realtime_Multi-Person_Pose_Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

Data Preparation code based on outputs from [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
