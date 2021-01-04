#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 10:54:51 2020

@author: jordy
"""

'''
This script consists in a sequence of commands used to train the InceptionV3
model which predicts the ingredients on pizza.

Launching the script consists in the training of the model over a fixed
number of epochs. I used numbers between 1-3 epochs over different days
in order to not overheat my computer but this causes the train/validation sets
to change over time, leading to more irregural results in accuracy and loss.
This is not a big problem because the same set of 8045 images is always used
so overall the training is always on the same data.

The total number of images in folder ./images is set as 8045.
The training is processed over the 8045 labeled images in this script, but 
we can clearly train on less images changing the value of tot_images

The model (class Model) is loaded from folder ./model. This means we can split
the training in different parts, loading and saving the model after every
run of this piece of code.

Then a Preprocess object
containing the 8045 labeled images is created and the train method is used.
See class Model for more information about the training.
'''

#Importing the 2 main classes
from preprocess import Preprocess
from model import Model

#Setting the number of images to train and the number of epochs
tot_images = 8045
epochs_num = 1

#Loading the model from folder and creating the preprocessed dataset to train
md = Model('./model')
data = Preprocess(tot_images)

#Facultative: Plotting the quantity of ingredients which appears on pizzas
data.bar()

#Actually training the model, see Model class for more information
md.train(data, epochs = epochs_num, save_model = True)
