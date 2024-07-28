# Fake Tweet Detection

This repository contains a model for detecting fake tweets using a diverse dataset. The model is built using BERT and TensorFlow, with both serial and parallel training approaches implemented to enhance performance. Additionally, a Flask-based web application is provided for user interaction.

## Table of Contents
- [Overview](#overview)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Results](#results)
- [Website](#website)


## Overview
This project aims to detect fake tweets using an image classification model with high accuracy and efficiency.

## Model Details
- **Frameworks Used**: BERT, TensorFlow
- **Training Approaches**:
  - Serial Training: Accuracy of 0.94
  - Parallel Training: Accuracy of 0.96 with reduced training time

## Dataset
The dataset used for training and testing consists of diverse fake and real tweets.

## Results
The results of the model include accuracy metrics for each training approach:
- **Serial Training**: 0.94
- **Parallel Training**: 0.96

## Website
A Flask-based web application is integrated with the trained model. Users can upload an image of a tweet, and the model will predict its authenticity. The web app is located in the `webapp` directory.

## Installation
To set up the project locally:
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/fake-tweet-detection.git
   cd fake-tweet-detection
