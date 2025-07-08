# Image Quality Manipulation using Deep Autoencoders

<image src="./screenshots/a.png">
    
## Mentors

- Aakarsh Bansal
- Abhishek Srinivas
- Raajan Wankhade

## Mentees

- Ananya A. K
- Raunak Nayak
- Sai Akhilesh Donga
- Sanga Balanarsimha
- Sarth Shah
- Utkarsh Shukla
- Vaibhavi Nagaraja Nayak
- Vanshika Mittal
- Vedh Adla

## Aim

- Perform denoising of images and image super resolution using deep autoencoders.
- To create a simple frontend(Streamlit) to deploy the model.

## Overview

In our project, we use the capabilities of autoencoder architectures to enhance image quality. By using super-resolution and noise removal techniques, our project aims to tackle two of the most important problems with image quality- denoising of noisy images, and super resolution of low quality images. Deep autoencoders unveil intricate details within images, making them significant tools in applications requiring image quality preservation and restoration such as medical diagnostics, surveillance, and satellite imagery.

During the course of the project, we were able to gain knowledge in the fields of Machine Learning, Deep Learning, Convolutional Neural Networks. We also completed Kaggle tasks during the learning phase of the project.

## Technologies used

1. Python
1. Streamlit
1. Pytorch


## Datasets

We made use of 2 datasets that were publicly available on Kaggle for performing denoising and super resolution. For the denoising of images, we used a dataset on Kaggle that contained 120 black-white images of Teeth X-Ray. For the super resolution component, we used a dataset on Kaggle that had 685 low and corresponding high resolution images. The links for the datasets are provided below :

[Super Resolution Dataset](https://www.kaggle.com/datasets/adityachandrasekhar/image-super-resolution)

[Denoising Dataset](https://www.kaggle.com/datasets/parthplc/medical-image-dataset)


## Model and Architecture

### 1. Denoising of Images

Presence of noise in an image makes it difficult to interpret. As our dataset had noise free images of Teeth X-Ray, we first artificially induce noise into these images. We chose randomly between adding noise from a Gaussian Distribution (Gaussian Noise) and noise from a Uniform Distribution (Uniform Noise). 

<br>
<image src="./screenshots/1.png"><br>

We then used an architecture that would model a UNET here. We use an autoencoder with skip connections (non-conventional U-Net) to obtain the denoised image from the noisy image. The autoencoder architecture comprises an encoder, consisting of two convolutional layers with ReLU activation, facilitating the compression of input images into a lower-dimensional representation. Following encoding, the decoder utilizes upsampling layers to reconstruct the original image dimensions. Skip connections are also employed to preserve spatial information during reconstruction. We finally use the sigmoid activation function to get the pixel values in the range [0,1].<br>

<image src="./screenshots/2.png"><br>

Our architecture consisted of 2 encoder layers followed by 2 decoder layers and we also employed the use of skip connections for garnering global context.<br>

<image src="./screenshots/3.png"><br>

We then used Cross Entropy Loss as the loss function and Adam as the optimiser. The model was trianed for 200 epochs and the learning rate to be around 1e-5. We were able to converge to a loss of around 0.000290 from 0.007233.<br>


### 2. Super Resolution of Images

Super resolution of images involves transforming a low quality image into a higher quality image while maintaining the content, color and details as much as possible. <br>


<image src="./screenshots/4.png"><br>

Again, we used an autoencoder to perform this task. We model a U-Net to do this where the encoder layer has a series of downsampling blocks, each consisting of two convolutional layers, after which we do batch normalization and activation by ReLU. This is for feature extraction and dimensionality reduction. Following this we have a bottleneck layer that acts as a bridge between the encoder and the decoder. The decoder consists of transpose convolutional layers that are used for upsampling. Moreover, at each upsampling step, we have skip connections from the encoder to help in understanding spatial context. Lastly, we have a 1*1 convolutional layer to adjust the number of output channels. <br>

<image src="./screenshots/5.png"><br>

We however, used a more complex architecture as compared to the denoising model with 5 encoder layers, also making use of skip connections.<br>

<image src="./screenshots/6.png"><br>

We tried 2 different types of loss - VGG Loss and MSE Loss. The model was trained for first 10 epochs using a combination of VGG Loss annd MSE loss to ensure semantic accuracy, and then using only MSE Loss for 10 more. Adam optimiser with learning rate of 0.0002 was used, the loss converged to about 0.001.<br>

## Results

### 1. X-Ray Image Denoising:
For denoising of images, on running for 200 epochs, we were able to achieve a loss of 0.00029. Some results obtained are as follows:<br>

<image src="./screenshots/7.png"><br>

<image src="./screenshots/8.png"><br>

Here are the calculated Median SSIM and PSNR on the entire dataset:<br>
```
SSIM (Noisy): 0.00298811656483968
SSIM (Denoised): 0.04422990805663252
PSNR (Noisy): 4.836596727387118
PSNR (Denoised): 5.8340839697113385
```
As you can see, the SSIM and PSNR have increased after denoising, which was the aim of the project.<br>



### 2. Image Super-Resolution:

For the super resolution of images, we were able to obtain a training loss of around 0.001 and some of the results on the test images are here:<br>

<image src="./screenshots/9.png"><br>

<image src="./screenshots/10.png"><br>

<image src="./screenshots/11.png"><br>

<image src="./screenshots/12.png"><br>

<image src="./screenshots/13.png"><br>

## Streamlit Interface

1. We were also able to create a simple user interface using Streamlit. After training the model for both tasks, we saved their weights, which allows us to use the model for any image without having to train again.<br>
1. For both tasks, we have a simple frontend where the user can choose to insert an image for either denoising or super resolution. The frontend would then display the new image - after denoising on increasing the resolution.<br>

## Conclusion

We were able to build 2 models that could successfully perform denoising and super resolution with great accuracy, and deploy models for both tasks using Streamlit, where users can select an input image to either denoise or get a higher resolution image. <br>

<image src="./screenshots/14.png"><br>

<image src="./screenshots/15.jpg"><br>

## Running the Streamlit App

1. Ensure that you have installed Git on your system.
You can check the installation using: 

```
git --version
```

2. Install streamlit
```
pip install streamlit
```
3. To run the app, please follow the given instructions:

    - Clone the repository onto your local system
    ```
    git clone https://github.com/raajanwankhade/autoencoder-image-quality-manipulation 
    ```
    - After this, for super resolution:
    ```
    cd autoencoder-image-quality-manipulation/super-resolution/app
    ```
    - For denoising,
    ```
    cd autoencoder-image-quality-manipulation/xray-denoising/app
    ```
    - Ensure that path to model weights is set correctly in the ```weights_pth ``` variable inside ```app.py```
    - After this:
    ```
    streamlit run app.py
    ```
## References

1. <a href="https://arxiv.org/abs/1505.04597"><u>U-Net: Convolutional Networks for Biomedical Image Segmentation</u></a>
2. <a href="https://www.science.org/doi/abs/10.1126/science.1127647"><u>Reducing the Dimensionality of Data with Neural Networks | Science</u></a>
3. <a href="https://www.kaggle.com/datasets/parthplc/medical-image-dataset"><u>https://www.kaggle.com/datasets/parthplc/medical-image-dataset</u></a>
4. <a href="https://www.kaggle.com/datasets/adityachandrasekhar/image-super-resolution"><u>https://www.kaggle.com/datasets/adityachandrasekhar/image-super-resolution</u></a>
5. <a href="https://arxiv.org/pdf/1808.03344"><u>Deep Learning for Single Image Super-Resolution: A Brief Review</u></a>
6. <a href="https://arxiv.org/abs/2301.03362"><u>Image Denoising: The Deep Learning Revolution and Beyond -- A Survey Paper –</u></a>


