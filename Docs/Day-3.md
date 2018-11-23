# Day - 3:

## Summary So Far:

<p align="center">
  <img width="460" height="400" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/w3_1.png">
</p>


## Exploring the Iris Data-Set:

One of the most common datasets that I came across in my journey into the data science world is called Iris. The dataset is a record of feature measurements (petal lengths and widths, sepal lengths and widths) of different species of Iris flowers. It is often used to demonstrate simple machine learning techniques.


<p align="center">
  <img width="460" height="400" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/W3_3.jpeg">
</p>

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.


<p align="center">
  <img width="460" height="400" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/W3_2.gif">
</p>


# Check out the Iris Data Set here: 
> [Iris Dataset](https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/tree/master/Iris%20Dataset)


## K-Fold Cross Validation:

Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.

The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. 
As such, the procedure is often called k-fold cross-validation. When a specific value for k is chosen, it may be used in place of k in the reference to the model, such as k=10 becoming 10-fold cross-validation.

Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. 
That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.

It is a popular method because it is simple to understand and because it generally results in a less biased or less optimistic estimate of the model skill than other methods, such as a simple train/test split.

> Read more about k-Fold Cross Validation [here](https://machinelearningmastery.com/k-fold-cross-validation/)


## Stratify 
Makes sure that each fold has the same proportion of observations.

## Feature Engineering

Datasets used to train classification and regression algorithms are high dimensional in nature — this means that they contain many features or attributes. In textual datasets each feature is a word and as you can imagine the vocabulary used in the dataset can be very large. Not all features however, contribute to the prediction variable. Removing features of low importance can improve accuracy, and reduce both model complexity and overfitting. Training time can also be reduced for very large datasets. In this lecture we will be  performing Recursive Feature Elimination (RFE) with Scikit Learn.

# Recursive Feature Elimination: 

Recursive Feature Elimination (RFE) as its title suggests recursively removes features, builds a model using the remaining attributes and calculates model accuracy. RFE is able to work out the combination of attributes that contribute to the prediction on the target variable (or class). Scikit Learn does most of the heavy lifting just import RFE from sklearn.feature_selection and pass any classifier model to the RFE() method with the number of features to select. Using familiar Scikit Learn syntax, the .fit() method must then be called.


# Q/A Time

<p align="center">
  <img width="460" height="400" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/W3_4.png">
</p>


# Thank You
Thanks for being so awesome and completing this 3 part Machine Learning Workshop. 
I hope all of you have a slight idea about how ML works and how to apply ML in real life data now. 
You can now take off from here, Remember the sky is the limit. We wish you all the best and Good Luck. 



