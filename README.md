# Project Food

## Content
In Project Food, I co-designed a convolutional neural network to classify triplets of pictures of different foods according to their similarity in taste. The concrete task was to predict whether the food on the first image of a given triplet tastes more similar to the food on the second image than to the food on the third image. The approach relies on transfer learning and a logistic regression model built on top of the augmented convolutional neural network. We used ideas from "Facenet: A unified embedding for face recognition and clustering" by Schroff et al. in 2015 for the design of the custom loss function.	

## Requirements
The user needs to save the images in a folder called "images" in the working directory. Also, the user needs to save the training and test set in separate .txt files with both files being a sequence of three numbers separated by a white space. These numbers specify the images (i.e. need to be identical to the image names). In the training set, the food on the first image always tastes more similar to the food on the second image.

## Scripts
main.py

## Notes
If possible install a GPU driver for Tensorflow, as it speeds up computations considerably. The output will be a sequence of zeros and ones indicating whether the food on the first image tastes more similar to the food on the second image than to the food on the third image. 

## Co-authors
David Wissel, Pascal Kündig
