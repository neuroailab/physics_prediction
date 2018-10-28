# Flexible neural representation for physics prediction

Codes for reproducing main results in paper "Flexible Neural Representation for Physics Prediction".

## Prerequisites

We have tested this repo under Ubuntu 16.04 with tensorflow version 1.9.0.

## Prepare data

For now, we provide one dataset including two rigid cubes each of which contains 64 particles hitting each other on a static plane described by 5000 particles.
You can download the dataset through this [link](http://physicspredictiondata.s3.amazonaws.com/physics_dataset.tar).
After downloading, untar this file.
Please check README.md in `data` folder for more explanations about this dataset.

## Start the training!

Go to folder `scripts`. And run the following command:

```
sh train_physics.sh --gpu your_gpu_number --dataset path_to_your_dataset
```

By default, the models and log will be saved to `~/.model_cache/physics_pred/`. 
Besides, multi-gpu training is not supported for now.
You can restore your training by setting `restore_path` parameter.

## Validate your training

We will soon release a script for using a trained model to generate predictions through unrolling.
Along training, the reported losses including `loss`, `preserve_distance_loss`, `un_velocity_loss`, and `velocity_loss` should usually keep decreasing.
At the end of the training, `velocity_loss` should be around `0.0030`.
