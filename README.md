# VeryDCN-for-Text-Classification

Implementation of Very Deep Convolutional Networks for Text Classification

PreProcessor.py - to be used only once for converting text data to a sequence of chars, where each char is represented by an integer (0-68).
This code is good for the Amazon reviews dataset; small modifications might need to be done for other datasets. 

TextDataset.py - a dataset implementation in order to initialize a torch data loader.

DataLoaderExample.py - an example of how to use the data loader with a simple model.