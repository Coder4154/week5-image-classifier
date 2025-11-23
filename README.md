# Week 5: Smart Image Classifier

## Overview
This project is a **Smart Image Classifier** built using Python and a model trained with **Google’s Teachable Machine**. The program can classify images from:

- Uploaded image files in the `test_images/` folder
- Live webcam feed in real-time

It identifies objects based on the custom classes I created (Dog or Cat).  

---

## Biggest Struggle
The most challenging part of this project was **dependency management in Python**.  

- Initially, running `load_test.py` failed due to **incompatible versions of NumPy, TensorFlow, and Matplotlib**.  
- After carefully uninstalling conflicting packages and manually installing compatible versions, the environment finally worked.  

This taught me the importance of **controlled virtual environments** and **matching package versions for deep learning projects**.

