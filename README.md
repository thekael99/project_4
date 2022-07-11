# Capstone Project: Dog breed classifier

## Table of Contents

1. [Description](#description)
2. [Contents](#contents)
3. [Requirements](#requirements)
4. [Acknowledgement](#acknowledgement)

## Description

In this project, I uses Convolutional Neural Networks (CNNs) and transfer learning in order to build a pipeline to process real-world image classification task. CNNs are commonly used to analyse image data. Transfer learning is a technique that allows to reuse a model across different tasks. The objective is that given an image of a dog, model will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed. If the model can't identify the image as a human or dog, it will inform to user.

You can use [online_app](demo_link) to demo project

Or you can build app in local with command:

`streamlit run app.py`

and access link [local_app](http://localhost:8501)

## Contents Notebook

The project is divided into 8 steps:

- Intro
- Step 0: Import Datasets
- Step 1: Detect Humans
- Step 2: Detect Dog
- Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
- Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
- Step 5: Write Your Algorithm
- Step 6: Test Your Algorithm
- Step 7: Demo app

## Acknowledgements

- Good Capstone Project
- You can visit [Udacity](https://www.udacity.com/) for providing an amazing Data Science Nanodegree Program
