# Instance Segmentation-based Angle Detection

This project implements an instance segmentation model for detecting the wire(heating element), camera and body of a robot using **Detectron2** and then determines the relative angles between them. 

## Overview
During endoscopic surgeries continuum robots often utilize heating elements to cut and remove unwanted material. Howerever, due to material properties of the heating element (Nitinol) there is a difference in motion when the element is straight to when it is bent. Thus there is a need to find a compensation factor. This is done by determining the element angles using this pipeline and then comparing them to the provided input.  

## Features
- Instance segmentation of objects in an image.
- Calculation and classification of angles between segmented objects.
- Outputs video/images with overlayed angle for the element.
- Docker-based environment setup for reproducibility and ease of deployment.

## Installation

### Prerequisites:
- Python 3.10.12
- Pytorch 2.1.0 (CUDA 11.8)
- Docker (optional)

### Steps to Install:
1. Clone the repository:
   ```bash
   git clone https://github.com/FujinNiazi/instance-segmentation-based-angle-detection.git
   ```

2. Clone and install detectron2:
   ```bash
   git clone https://github.com/facebookresearch/detectron2.git
   ```
   
   This will create a folder called detectron2 in your current directory, and you can then install it by running:
   ```bash
   cd detectron2
   pip install .
   cd ..
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

&nbsp;

![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white) 
  
***Optional*** If you want to use Docker:
1. Build the Docker image:
   ```bash
   docker build -t angle-detection .
   ```
2. Run the Docker container:
   ```bash
   docker run -it angle-detection
   ```
## Usage
Once the installation is complete, you can run the model using the pretrained weights to detect the wire/line angle in the images.
To run the angle detection model on a video or set of images, use the following command:
  ```bash
  python main.py \
  --config path/to/config.yaml \
  --weights path/to/model_final.pth \
  --input path/to/video/or/image_folder \
  --output path/to/output_dir
  ```
**OR** you can place the required files in the relevant directories.


## Metrics & Example Output
We utilized the mask_rcnn_R_FPN_3x model for training as compared to the other models from the model zoo, it gave the best performance for the shortest training time. 
The model was trained on 600 images (70/30 split) of the face of the continuum robot with both the face and the heating element in different orientations. 

And the model after 3000 iterations was able to achieve about **90%** for both bounding box AP and segmentation AP. 
On a per category basis the segmentation performance of the ***line*** or heating element was lacking a bit at about 76% but that can attributed to the its complex shapes especially during its motion.


Overall the algorithm performs reasonably well as shown in the following images with the bounding boxes overlayed.
<p float="left">
  <img src="https://github.com/user-attachments/assets/c0c1b1d6-15e2-4c36-9023-49e1bdd86997" width="32%" />
  <img src="https://github.com/user-attachments/assets/cd910a6e-9934-4e33-b933-ce80841cad21" width="32%" />
  <img src="https://github.com/user-attachments/assets/991a9a20-e524-448b-8086-b7197a7aabb8" width="32%" />
</p>

Or in the following with the segmentation masks
<p float="left">
  <img src="https://github.com/user-attachments/assets/ce3500af-3c9b-461f-bddb-854c8298f57c" width="32%" />
  <img src="https://github.com/user-attachments/assets/af98b6b8-749f-4878-861c-c9ec6ec0a3d1" width="32%" />
  <img src="https://github.com/user-attachments/assets/b79e10cb-a2c3-4e82-9e69-fb9b2f6715be" width="32%" />
</p>

Video samples
<p float="left">
  <img src="https://github.com/user-attachments/assets/924b7127-df24-4141-be2a-273a67351cb6" width="45%" />
  <img src="https://github.com/user-attachments/assets/6aa9a5e5-a361-4406-97b2-7e637613cb90" width="45%" />
</p>



## License
This program is licensed under the Apache 2.0 License.
