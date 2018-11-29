# Flexible neural representation for physics prediction

Codes for reproducing main results in paper "Flexible Neural Representation for Physics Prediction".

## Prerequisites

We have tested this repo under Ubuntu 16.04 with tensorflow version 1.9.0.

## Prepare data

For now, we provide one dataset including two rigid cubes each of which contains 64 particles hitting each other on a static plane described by 5000 particles.
You can download the dataset through this [link](http://physicspredictiondata.s3.amazonaws.com/physics_dataset_new.tar).
For those who downloaded the dataset before Nov. 29th, please download it again as we updated the file to include pretrained model and validation files.
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

During training, the reported losses including `loss`, `preserve_distance_loss`, `un_velocity_loss`, and `velocity_loss` should usually keep decreasing.
At the end of the training, `velocity_loss` should be around `0.0006`.

Additionally, we provide two example bash scripts for quantatively and qualitatively verifying the trained model. 
Besides, we also provide a pretrained model trained using the training bash script in the dataset.

Now if you want to run a qualitative test, you can go to folder `scripts`. And run the following command:

```
sh test_physics_qual.sh --gpu your_gpu_number --dataset path_to_your_dataset --restore_path path_to_your_dataset/pretrained_model/checkpoint-384000
```

By default, this will generate two pickle files named as `true_results_physics_pred_better_8.pkl` and `results_physics_pred_better_8.pkl` in your home directory.
You can change saving directly by setting `SAVE_DIR` parameter. Please check examples of visualizing these results in `visualize` folder.

Similarly for a quantitative test, you can just replace `test_physics_qual.sh` using `test_physics_quan.sh`. 
Then you will see the metrics also reported in our paper as outputs of this command.
And the test result will be stored as `quant_results_physics_pred_better_8.pkl` in your home directory by default.
