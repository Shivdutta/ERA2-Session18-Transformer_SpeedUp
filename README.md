# Transformer - Language Translation

This repository contains following files:

- `config.py`: Configuration data for Transformer model training
- `dataset.py`: Contains dataset class to create data for English to Italian translation 
- `LITTransformer.py`: Contains lightning module to train the transformer model
- `PyLight_Model.py`: Contains Pytorch Lightening injected building blocks for transformer model
- `train.py`: Contain utilities for model trainingP
- `PyLight_S16_17.ipynb.ipynb`: Notebook with model training details


## Training Details

- The model was trained on dataset from HuggingFace on Google Colab Notebook

- The transformer model is trained to translate the English to Italian

- The model is trained for 10 epochs
  ![Training Log](Data/training.jpg)

- Below is the sample output after training

```commandline
    SOURCE: How often am I to say the same thing?
    TARGET: quante volte dovrò ripetere la stessa cosa?
    PREDICTED: Come sono contenta di fare ?
```
