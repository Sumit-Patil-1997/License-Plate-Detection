# 🚗 License Plate Detection using Faster R-CNN + MobileNet V2

This project implements a license plate detection system using a custom object detection pipeline with **Faster R-CNN** and a **MobileNet V2 backbone** with **Feature Pyramid Network (FPN)** in **PyTorch**. The training data is sourced from **Roboflow Universe**, and the model is trained to accurately detect license plates in images.


## 📂 Project Overview

- 🔍 **Model**: Faster R-CNN with MobileNetV2 + FPN
- 🧠 **Framework**: PyTorch
- 📦 **Dataset**: COCO format (downloaded via Roboflow API)
- 🎯 **Task**: License Plate Detection (Object Detection)
- 🖼️ **Inference**: Bounding box prediction on images with OpenCV visualization


## 📁 Folder Structure

- license-plate-detection/
  
├── dataset/   # Roboflow COCO-format dataset

├── License_Plate_Detection.ipynb    # Training script

└── README.md    # Project documentation


=====================================================================================

📥 Dataset - Roboflow
https://public.roboflow.com/object-detection/license-plates-us-eu?utm_source

And 

Dataset is downloaded directly from Roboflow Universe using their API:
--------------------------------------------------------------------------------------



from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")

project = rf.workspace("your-workspace").project("your-project")

dataset = project.version(1).download("coco")
----------------------------------------------------------------------------------------
✅ Be sure to export your dataset in COCO format for compatibility with Faster R-CNN.

----------------------------------------------------------------------------------------

Key components:

Converts COCO format to PyTorch's expected input

MobileNetV2 backbone + Feature Pyramid Network

GPU training supported (uses torch.cuda.is_available())

-------------------------------------------------------------------------------------------

🧠 Acknowledgements

Roboflow Universe – for open datasets

PyTorch – for the deep learning framework

TorchVision – for Faster R-CNN & MobileNetV2

