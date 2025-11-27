# Wind Farm Power Prediction

A machine learning project for predicting wind farm power output using deep learning with TensorFlow/Keras and MLflow for experiment tracking.

## Overview

This project implements a neural network model to predict power output from wind farm data. The model uses historical wind farm data to learn patterns and make predictions about future power generation.

## Features

- **Deep Learning Model**: Multi-layer neural network with 3 hidden layers (100 neurons each)
- **MLflow Integration**: Experiment tracking and model versioning
- **Data Processing**: Automated data loading and preprocessing from remote dataset
- **Model Evaluation**: Validation metrics including RMSE and loss tracking

## Requirements

- Python 3.12+
- TensorFlow 2.20.0+
- MLflow
- pandas
- numpy

## Installation

Install the required dependencies:

```bash
pip install tensorflow mlflow pandas numpy
```

## Usage

1. Open the Jupyter notebook `windpark_26_11_25.ipynb`
2. Run all cells sequentially to:
   - Install dependencies
   - Load wind farm data
   - Train the neural network model
   - Log the model with MLflow
   - Make predictions on sample data

## Model Architecture

The model consists of:
- **Input Layer**: Accepts features from wind farm data
- **Hidden Layers**: 3 dense layers with 100 neurons each, ReLU activation
- **Output Layer**: Single neuron for power prediction
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)

## Training Configuration

- **Epochs**: 200
- **Batch Size**: 64
- **Validation Split**: 20%
- **Training Period**: 2014-01-01 to 2018-01-01

## Data Source

The wind farm dataset is loaded from:
```
https://github.com/dbczumar/model-registry-demo-notebook/raw/master/dataset/windfarm_data.csv
```

## MLflow Tracking

The project uses MLflow to track:
- Model parameters (epochs, batch size, validation split, number of hidden layers)
- Training metrics (validation loss, RMSE)
- Model artifacts and signatures

## Project Structure

```
windpark/
├── README.md
└── windpark_26_11_25.ipynb
```

## License

This project is part of a course exercise (26_11_25).
