# Transformer - Language Translation Speedup

This repository contains following files:

- `config.py`: Configuration data for Transformer model training
- `dataset.py`: Contains dataset class to create data for English to Italian translation 
- `LITTransformer.py`: Contains lightning module to train the transformer model
- `PyLight_Model.py`: Contains Pytorch Lightening injected building blocks for transformer model
- `train.py`: Contain utilities for model trainingP
- `PyLight_S18.ipynb`: Notebook with model training details

# Transformer - Language Translation Speedup

## Overview
This project aims to speed up language translation by training a Transformer model for translating English to Italian. The model is trained using the PyTorch Lightning framework, which provides a high-level interface for PyTorch, enabling efficient and scalable deep learning model training.

## Training Details

### Data Preparation
- **Dataset**: The dataset used for training includes parallel English-Italian sentences.
- **Preprocessing**: Sentences are tokenized and converted into tensors suitable for input to the Transformer model.

### Model Configuration
- **Model Architecture**: The Transformer model is based on the standard architecture with an encoder-decoder structure.
- **Configuration**: Key hyperparameters include batch size of 6, 18 epochs, and mixed precision training with Lion Optimizer.

### Training Process
- **Training Framework**: PyTorch Lightning is used to streamline the training process, enabling efficient GPU utilization and easy integration of callbacks.
- **Epochs**: The model is trained for 18 epochs.
- **Validation**: Validation checks are performed once every 5 epochs to monitor the model's performance and prevent overfitting.
- **Precision**: Mixed precision training (16-bit) is used to speed up the training process and reduce memory usage without compromising model accuracy.

![image](https://github.com/Shivdutta/ERA2-Session18-Transformer_SpeedUp/assets/15068266/2f2413dc-b9b9-4106-bbba-977f28ed72bc)

- Below is the sample output after training

```commandline
    SOURCE: How often am I to say the same thing?
    TARGET: quante volte dovrò ripetere la stessa cosa?
    PREDICTED: quante sono ripetere di fare ?
```
