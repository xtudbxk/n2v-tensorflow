### Introduction

This is a project which just move the [N2V](https://github.com/juglab/n2v) to tensorflow. The N2V is referring to the approach for image denoising in the paper ["Noise2Void - Learning Denoising from Single Noisy Images"](https://arxiv.org/abs/1811.10980). And here, I just use the tensorflow to implement the approach.

### Preparation
for using this code, you have to do something else:

##### 1. Install tensorflow-2.2.0
please refer to [tensorflow](https://tensorflow.google.com/) for details.

##### 2. Download the data
this project only implement the code to load the BSD68 dataset. And you can follow the two steps to prepare the dataset.
>1. open [BSD68_reproducibility_data.zip](https://cloud.mpi-cbg.de/index.php/s/pbj89sV6n6SyM29/download/) in web viewer and download it to the folder "data".
>2. uncompress the "BSD68_reproducibility_data.zip" in "data" folder.

### Training

then, you just input the following sentence to train it.

> python train.py <gpu_id>

### Testing
> python predict.py <gpu_id> saved_weight_path

### Result
final psnr on BSD68 is 27.48 dB while it is 27.71 dB.
