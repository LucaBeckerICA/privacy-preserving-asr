# Dataset installation

## General
Download the [LRS3]{https://mmai.io/datasets/lip_reading/} and [VoxCeleb2]{https://huggingface.co/datasets/ProgramComputer/voxceleb/tree/main/vox2} datasets.

## Pre-processing the data
The video files in LRS3 doe not have to be processed specifically. The video files in VoxCeleb2 have to be downscaled from 224 x 224 to 96 x 96 pixel.

## Required Metadata files
In the example config some text files are necessary to select labels, signals, etc. All of these files are either in [`data/LRS3`](data/LRS3) or in [`data/voxceleb`](data/voxceleb), depending on the task in the config.