#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:35:28 2020

@author: jordy
"""

import time
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from cv2 import imread, resize
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input


class Preprocess:
    """Used to preprocess a dataset of images in order to train a model.
    
    The preprocessing consists in getting a pre-determined number
    of images from a pre-determined folder in order to resize them to the
    same pre-determined dimension. These images are assumed to be 3-channel
    RGB images with the channel number as 3rd dimension.
    The preprocessing also consists in associating to every image its 
    vector containing binary labels. These binary labels are 7-dim vectors
    which tells us wheter an ingredient is on the pizza image associated
    or not.
    These preprocessed images and labels will then be stored into a tensor
    and will be the input to train a neural network.
    For further information about the neural network refer to model.py.
    FIXME: For a matter of comfort (mine), attributes act like constants and 
    should be changed in this script.
    For further informations about how these images and labels are provided
    refer to http://pizzagan.csail.mit.edu
    
    Attributes
    ----------
    LABELS_PATH : str
        A string containing the path where lables are stored. The .txt file
        containing labels can be found in the repository of this file under
        the name imageLabels.txt.
    IMAGES_PATH : str
        A string containing the path (folder) where images are stored. Images
        used for the training can't be put into a GitHub repository due to
        high dimensions so basically just pretend an ./images folder full of 
        pizza images is present in the repository.
    TARGET_IMAGES_SIZE : 2-dimensional tuple
        This is the tuple containing the dimensions in pixel (length,height)
        at which images should be resized by the preprocessing. These
        dimensions should match the ones for the input of the class model.py
        To be clear the images also have 3 channels so their dimension as
        input of the model is (length,height,3) while the preprocess
        assumes 3-channeled images (with channel-last) are provided.
    TOT_IMAGES : int
        Total number of images in IMAGES_PATH folder.
    images : NumPy 4-dimensional tensor
        Tensor containing the images. The first dimension of this tensor is
        the index associated to an image or a counter of the images. The
        second and third dimensions represent width and lenght of the images.
        The fourth dimension is 3 and represents the channel of the images.
        Example: images.shape = (10,299,299,3) means we have 10 images with
        size 299x299 with 3 colour channels.
    labels : NumPy 2-dimensional tensors
        Matrix containing the labels associated to every of the images. The
        first dimension is the index of the image associated while the second
        dimension is 7, which is the number of ingredients that can be
        detected on pizzas by the neural network.
        
    Methods
    -------
    __init__(n_images = 128, elapsed_time = True)
        Instantiate an object of this class.
    show_image(index = None, print_lables = True)
        Show a random image (and its label) taken from preprocessed images.
        This method was used to visualize images and check if labels were
        correctly associated to images.
    bar()
        Show a bar graph which counts every ingredients that appears in the
        preprocessed images.
    """
    LABELS_PATH = './imageLabels.txt'
    IMAGES_PATH = './images'
    TARGET_IMAGES_SIZE = (299,299)
    TOT_IMAGES = 8045
    
    def __init__(self, n_images = 128, elapsed_time = True):
        """Initzialize an object containing preprocessed images and labels.
        
        The inizialization of a Preprocessing object consists in setting
        the parameters self.images and self.labels.
        A pool of n_images indexes is randomly created to preprocess random
        images and labels instead of fixed ones.
        In the case of self.images, n_images associated to the random pool of 
        indexes are taken from IMAGES_PATH folder, resized to 
        TARGET_IMAGES_SIZE and stacked into self.images.
        In the case of self.labels, n_images labels are extracted
        from LABELS_PATH, appropriately parsed in order to get binary vectors
        with size 7 (number of ingredients) and stacked into self.labels.
        Everything is done wrt the fact that x image in self.images is
        associated with x label in self.labels for every x form 0 to n_images.
        If the number of images is set to be TOT_IMAGES, then the images are
        not randomly picked and every image from IMAGES_PATH is used.
        
        Parameters
        ----------
        n_images : int, optional
            The number of images (and associated labels) to be processed.
            These images are the first n_images of IMAGES_PATH, so in order
            to get the largest pool of images this value should be 8045, which
            is the total number of images in the folder. The default is 128.
            128 has been small enough to preprocess small batches of images
            to test the code.
        elapsed_time : bool, optional
            Optionally print the duration in time of the preprocess. Used
            mainly for debugging purpose. The default is True.

        Returns
        -------
        None.

        """
        
        
        if elapsed_time == True:
            start = time.time()
        
        #Opening raw labels from a .txt files which is organized as follows:
        #- 8045 rows
        #- Every row contains 7 elements which are 0 or 1, separated by '  '
            
        raw_labels = open(self.LABELS_PATH, 'r').read()
        
        #Split the whole file into array which elements are the rows
        raw_labels = raw_labels.split('\n')
        #Deleting the last item, which is '' (empty)
        del raw_labels[-1]
        
        #Creates the pool of indexes associated to the images to preprocess
        if n_images != self.TOT_IMAGES:
            index_pool = random.sample(range(len(raw_labels)), n_images)
        else:
            index_pool = range(self.TOT_IMAGES)
        
        #Creating the labels object as a numpy array
        self.labels = []
        for i in index_pool:
            self.labels.append(list(map(int,raw_labels[i].split(' '))))
            
        self.labels = np.array(self.labels)
        
        #Preprocessing the images starting from their paths
        #
        #- files contained in IMAGES_PATH folder are listed 
        #- n_images images are loaded from their path and processed
        #- Processing on images consists in a resize to (299x299x3)
       
        paths = os.listdir(self.IMAGES_PATH)
        paths.sort()
        
        #Delete hidden filenames/dir which are sorted at the beginning of paths
        #This is not necessary if you are 100% sure the only paths are images
        #In this case images namefiles start with a '0' so we delete the others
        while paths[0][0] != '0':
            del paths[0]
        
        #Adding the complete path to the images (not sure if necessary)
        paths = [self.IMAGES_PATH + '/' + path for path in paths]
        
        #Check if len(paths) corresponds to len(raw_labels). It should.
        if len(paths) != len(raw_labels):
            print('Mismatch between the number of images and labels [1]')
            
        #Actually preprocessing the images, straight-forward.
        #Images are read, resized and appended to the final array of images
        self.images = []
        for i in index_pool:
            image = imread(paths[i])
            image = resize(image, self.TARGET_IMAGES_SIZE)  
            image = img_to_array(image) 
            self.images.append(image)
        self.images = np.array(self.images, np.float32)#aaaaaaaaaaaaaa
        
        #keras Inceptionv3 preprocess. Turning pixels into (-1,1) values.
        preprocess_input(self.images)
        
        #Check if len(self.images) corresponds to len(self.labels). It should.
        if len(self.images) != len(self.labels):
            print('Mismatch between the number of images and labels [2]')
        
        
        if elapsed_time == True:
            end = time.time()
            print("Preprocessing elapsed time: ", end - start," seconds" )
        
    
    

    def show_image(self, index = None, print_lables = True):
        """
        This methods shows an image and its associated label in order to
        check whether the label corresponds.

        Parameters
        ----------
        index : int, optional
            The index associated to the image to show, This index should be
            equal or more than 0 and < length of self.images. If not provided
            it's chosen randomly. The default is None.
        print_lables : bool, optional
            Also print the labels associated to the images into the console.
            To check whether the associated label is correct, refer to the 
            file categories.txt. The default is True.

        Returns
        -------
        None.

        """
        
        if index is None:
            index = random.randint(0, len(self.labels) - 1)
            
        #BGR (used by cv2) to RGB (used by matplotlib) converting
        #img is in cv2 format, img2 is in matplotlib format
        img = self.images[index]
        img2 = img[...,::-1].copy()
        
        plt.imshow(img2)
        
        if print_lables == True:
            print(self.labels[index])
            
            
            
            
    def bar(self):
       """
       This is to check how the ingredients are distributed over the 
       preprocessed images. From self.labels the ingredients are counted
       and plotted into a bar graph.

       Returns
       -------
       None.

       """ 
       #This sums every column of self.labels 
       ingredients_sums = np.zeros(7) 
       for i in range(7):
           ingredients_sums[i] = np.sum(self.labels[:,i], dtype = int)
         
       #Then the sums are plotted as a bar graph.    
       plt.figure()
       plt.title('Ingredients counter')
       plt.bar(x = np.arange(7), height = ingredients_sums)
       plt.xticks(np.arange(7), 
                 ('Pepr','Mush','On','Pepp','Oliv','Tom','Bas'))
       plt.show()
    
          

        