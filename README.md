# Implementation-of-Logistic-Regression-Using-Newtons-Method

## Objective:
This project implements the logistic regression optimized by Newton's Method from scratch (without using any existing machine learning libraries e.g. sklearn) for optical recognition of handwritten digits dataset. 

The hyperparameter lambda (the weight of penalty) is fine-tuned using 10-Fold Cross Validataion which is also implemented from scratch.

Note: sklearn packages are used in this project for verification and comparision purposes.

## Dataset:
Link: http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits  

The data used for this project is a modified version of the Optical Recognition of Handwritten Digits Dataset from the UCI repository. 
It contains pre-processed black and white images of the digits 5 and 6. Each attribute indicates how many pixels are black in a patch of 4 x 4 pixels.

### Format: 
There is one row per image and one column per attribute. The class labels are 5 and 6. The training set is already divided into 10 subsets for 10-fold cross validation.

## Classification Accuracy Results from 10-Fold Cross Validation:

![Capture](https://user-images.githubusercontent.com/29167705/63802851-8836cb80-c8e1-11e9-86c4-31eac879d937.JPG)

## Visualization:

![Capture](https://user-images.githubusercontent.com/29167705/63802902-a8668a80-c8e1-11e9-82d5-7ec3176e815e.JPG)

## Advantages of Newton's Method over Gradient Descend:
At each iteration, Newton's method estimates a quadratic Q(w) at current w_i. w_(i+1) = argmax[Q(w)]. Therefore, the local maximum Q(w) provides the next w.

Comparing to linear converging rate from gradient descend. Newton's method has a quadratic converging rate. 

In addition, it's very hard to determing the learning rate from the gradient descend. Small learning rate will slow down the learning, but large learning rate lead to divergence. Newton's method mitigates this problem since its step size is determined at each iteration.
