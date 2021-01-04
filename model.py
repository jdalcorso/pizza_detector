#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:26:19 2020

@author: jordy
"""

import numpy as np
import matplotlib.pyplot as plt
from keras import Model as Md
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.applications import InceptionV3


class Model:
    
    """Used to create the neural network which predicts ingredients on pizza.
    
    This class consists of the model used to make predictions about the
    ingredients over a pizza image provided.
    The problem of guessing/predicting the ingredients on a pizza image can
    be described as a multilabel classification problem. In a multilabel
    classification an image can belong to different classes (labels)
    simultaneously. In our setting labels are represented by ingredients so
    if an ingredient/label is set to 1, it belongs to the image.
    As other Machine Learning models, this needs to be trained over labeled 
    data. In this case the dataset used to train the model is a Preprocess 
    object (see preprocess.py for more information about the dataset).
    The model consists in a neural network called InceptionV3 which is briefly
    described in the documentation of the initzializer of this class.
    
    
    Attributes
    ----------
    INPUT_DIM: 3-dimension tuple
        Dimension of images taken as input by the model. InceptionV3 works
        with 299x299x3 image, which contains pixels values from -1 to 1. This
        is achieved during the preprocess of the images.
    OUTPUT_DIM: int
        Labels predicted by the model.  Values near 1 means an ingredient is
        on the pizza, while 0 means not.
    model: Functional Keras object
        InceptionV3 model, which is the CNN used to make predictions. More 
        details in the __init__ docstring.
        
    Methods
    -------
    __init__(model_path = None)
        Creates the model to accomplish the task of predicting ingredients.
    train(data, 
          return_history = False, 
          plot_loss = True, 
          val_loss_callback = True, 
          save_model = False, 
          batch_size = 32, 
          epochs = 5)
        Function used to train the model, this is also the core of train.py
    test(data, batch_size = 32)
        Test the model on a Preprocess object.
    predict(data, show_images = True)
        Test the model over a smaller Preprocess object.
    save(save_path = './model'):
        Save the model.
    show()
        Show the structure of the model.
    
    
    """
    
    INPUT_DIM = (299,299,3) #256 X 256
    OUTPUT_DIM = 7
    
    def __init__(self, model_path = None, weights_path = None):
        """
        The Model class initzializer consists in iniztializing the model
        attribute, which is actually the model used to make predictions.
        The model used here is InceptionV3, a CNN on which some information 
        are provided at https://arxiv.org/abs/1512.00567. In general, this 
        model is one of the best CNN model for image classification. It 
        is provided by keras so I basically used it with pretrained weights
        obtained from imagenet classification. I modified the top layers of
        the net in order to accomplish my task, which is a multi-label
        classification problem (while the InceptionV3 provided by keras is 
        for multiclass). The modification consists in changing the activation
        function to sigmoid (which basically doesn't penalize other classes
        while giving high weight for the most suitable one). Then the
        canonical loss and optimizer for a multilabel classification are used,
        in particular the loss function is binary crossentropy.
        (see https://keras.io/api/applications/inceptionv3/ for more) 
        
        Other attributes could be added, for example a Preprocessing object
        or the past history of trainings. Another big improvement should be
        turning this class into a child of the tensorflow.keras Model class,
        which is the "canon" way to deal with models.

        Parameters
        ----------
        model_path : string, optional
            If provided, the model is taken from this path instead of being
            created from zero. The default is None.

        Returns
        -------
        None.

        """
        
        if model_path is None:
            
            #InceptionV3 works on 299x299x3 images if no input shape is provided
            #Also, imagenet weights are used by default
            inception = InceptionV3(include_top = False)

            #Creation of "our" top layers, to get a multilabel classifier
            last_layers = inception.output
            last_layers = GlobalAveragePooling2D()(last_layers)
            last_layers = Dense(self.OUTPUT_DIM, activation='sigmoid')(last_layers)
            
            #Adding the new layers to InceptionV3
            self.model = Md(inputs = inception.inputs, outputs =  last_layers)

            #Compiling the model
            self.model.compile(loss='binary_crossentropy',
                               optimizer='adam', 
                               metrics = ['accuracy'])
            
            #Load pretrained weights
            if weights_path is not None:
                self.model.load_weights(weights_path)
            
            
        else:
            #Load a (already trained) model from model_path directory
            #Note that loading in this way means the model is already compiled
            self.model = load_model(model_path)






    #The train function take as input a Preprocessing object (data parameter). See class Preprocessing for further detail
    #Then the training of the CNN is computed on X = images and y = labels
    def train(self, 
              data, 
              plot_loss = False, 
              val_loss_callback = True, 
              save_model = False, 
              batch_size = 32, 
              epochs = 5):
        """
        This function characterize the training of the model.
        The training is computed over a Preprocess object where images are
        inputs for the model while labels are outputs.
        An images augmentator is used in order to make the model see different
        versions of every image during different epochs.
        A function from sklearn is then used to divide the dataset in a train
        and a validation parts. The function fit, from keras, is the core of
        the training process: it apply the model over the images provided
        and trains the net using backpropagation. The idea of training is the
        same as many other models, we want to minimize the loss function (in
        this case binary crossentropy) changing the weights of the network.
        In this case there are around 24 million trainable weights.
        Using my Macbook Pro 2019 (Intelcore i5 quad-core, 1.4 GHz, 16 GB RAM)
        every epoch lasted around 20-25 minutes if 8045 images were used.
        I also tried for a GPU approach but didn't manage to make it work. A
        GPU training would have allowed me to train even more epochs and that
        could be a big improvement to this model.
        

        Parameters
        ----------
        data : Preprocess object
            Images with associated labels used to train the model. For more
            information see preprocess.py
        plot_loss : boolean, optional
            Whether to plot the value of the loss function after every epoch
            or not. The default is True.
        val_loss_callback : boolean, optional
            Whether to use ore not a callback over validation loss.
            A callback stops the training process if a metric has an unusual
            behaviour (in this case loss growing).
            Mainly used  for debugging the model and being sure loss didn't grow.
            The default is True.
        save_model : boolean, optional
            Wheter to save the model or not. This can be useful because allows
            us to train the model, save it and load it in the ui.py or for
            another piece of training. The default is False.
        batch_size : int, optional
            Batch size of every step of the training. Basically during
            every step of the training 32 labeled images are evaluated by the
            model. The default is 32.
        epochs : int, optional
            Number of times the model sees all the dataset provided.
            The default is 5.

        Returns
        -------
        None

        """
        #Data generator object that augment the dataset provided.
        #Several transformations are applied to obtain the augmentation
        #In every epoch a transformed version of every image is used
        images_augmentator = ImageDataGenerator(rotation_range=25,
                                                width_shift_range=0.1, 
                                                height_shift_range=0.1,
                                                shear_range=0.2, 
                                                zoom_range=0.2,
                                                horizontal_flip = 1,
                                                vertical_flip = 1)
        
        #Splitting the dataset.
        #Note that test in this case means validation
        (images_train, images_test, labels_train, labels_test) = train_test_split(data.images, data.labels, test_size = 0.2, random_state = 69)
        
        
        #Callback to monitor the val_loss function, to overcome overfitting
        #The train is stopped if the val_loss gets higher in 2 consecutive epochs
        if val_loss_callback == True:
            early_stopping = EarlyStopping(monitor='val_loss',
                                       mode='min', 
                                       patience = 2,
                                       verbose=1)
        else:
            early_stopping = None
            
            
        
        
        #Actually the training of the model. 
        history = self.model.fit(
                    x = images_augmentator.flow(images_train, labels_train, batch_size = batch_size),
                    validation_data = (images_test, labels_test),
                    epochs = epochs,
                    callbacks = [early_stopping],
                    )
        
        if save_model == True:
            self.save('./model')
        
        
        #Plot the history of loss and val_loss during training
        if plot_loss == True:
            plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.xlabel('epochs')
            plt.legend(['loss','val_loss','accuracy','val_accuracy'])
            plt.show()
        
        
    
    
    
    def test(self, data, batch_size = 32):
        """
        Test the model on the data Preprocess object provided. This method
        tests the model appying it to the images and checking wether the 
        predictions are similar to data.labels true values.

        Parameters
        ----------
        data : Preprocess object
            Images to predict and labels to check.
        batch_size : int, optional
            Batch size used by keras.evaluate(). The default is 32.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.model.evaluate(data.images, data.labels, batch_size = batch_size)
    
    


    def predict(self, data, show_images = True):
        """
        This method predict the labels of one or more images provided.
        Images need to be part of a Preprocess object. In order to predict
        an image which is not part of a Preprocess object, refer to 
        predict_this_image

        Parameters
        ----------
        data : Preprocess object
            Images to predict.
        show_images : TYPE, optional
            Wether to show predicted images or not. The default is True.

        Returns
        -------
        float array
            Return an array which rows are the predicted labels for the images
            provided.

        """
        
        if show_images == True:
            for i in range(data.images.shape[0]):
                img = data.images[i]
                img2 = img[...,::-1].copy()
                plt.figure()
                plt.imshow(img2)
        
        if data.images.shape[0] > 1:
            #Predicting on more than 1 image we use keras predict function
            #keras predict function should be used for a consistent number of images
            prediction = self.model.predict(data.images,
                                            batch_size = data.images.shape[0])
        else:
            #While predicting on 1 image it's more convenient to use directly the model
            prediction =  self.model(data.images, train = False)
        
        return np.array(prediction)
       
    
    
    
    def predict_this_image(self, image):
        '''
        Predict the ingredients over the pizza image provided.
        ThE InceptionV3 model is directly applied over the standard input 
        provided, the output is then converted from EagerTensor to Numpy.
        This is the function used in ui.py to get predictions.

        Parameters
        ----------
        image : numpy array
            Image needs to be (1x299x299x3) and pixel values between -1 and 1
            in order to match the standard input of InceptionV3.

        Returns
        -------
        predictions : numpy vector
            1x7 numpy vector containing predictions. If a value of this vector
            is near 1, the corresponding ingredient is probabily on the pizza.

        '''
        #Apply the model to the image
        predictions = self.model(image)
            
        #Convert prediction from EagerTensor to Numpy
        predictions = np.array(predictions)
        
        return predictions
        
    
    
    def evaluation(self, data, t = [0.5,0.5,0.5,0.5,0.5,0.5,0.5]):
        """
        Homemade evaluation function. Images from data parameter are
        predicted by the model. In particular, predictions are turned into
        binary labels using the threshold vector t (which could be refined
        in order to get better predictions). Then these binary labels are
        compared with the true ones from data.labels.

        Parameters
        ----------
        data : Preprocess object
            Images used for the predictions and labels to check wheter the
            predictions are right.
        t : float vector, optional
            Vector containing the threshold which tells us whether an
            ingredient is on the pizza or not. 
            The default is [0.5,0.5,0.5,0.5,0.5,0.5,0.5].

        Returns
        -------
        None.

        """
        correct_predictions = 0
        total_predictions = 0
        
        predictions = self.model(data.images)
        
        for i in range(len(data.labels)):
            
            print('True:',data.labels[i])
            
            prediction = predictions[i]
            prediction = np.array(prediction)
            
            print('Pred:',prediction)
            
            #discretize the prediction
            for l in range(len(prediction)):
                if prediction[l] < t[l]:
                    prediction[l] = 0
                else:
                    prediction[l] = 1
            
            
            #comparison prediction/true value
            correctness_flag = True
            ingredient = 0
            while correctness_flag and  ingredient < 7:
                if prediction[ingredient] != data.labels[i][ingredient]:
                    correctness_flag = False 
                ingredient += 1
                
            if correctness_flag == True:
                correct_predictions += 1
            
            total_predictions += 1
        
        print('Correct predictions percentage: ', correct_predictions/total_predictions)
                
            
        
    
    
    #Save the (trained) model to local
    def save(self, save_path = './model'):
        """
        In my case the save_path is ./model but it can be changed.
        The saving is useful because __init__ can load a saved model instead
        of creating a new untrained one.
        As below, an ineheritance from keras Model class would made this 
        function deprecated and only keras.model.save would be used.

        Parameters
        ----------
        save_path : string, optional
            Where to save the model. The default is './model'.

        Returns
        -------
        None.

        """
        self.model.save(save_path)
    
    
    
    
    
    
    def show(self):
        """
        Prints the structure of the model. No need of parameters. Such a 
        function shows how an ineheritance from the keras Model class would
        have made benefits.

        Returns
        -------
        None.

        """
        self.model.summary()
        
        
        
        