# Day - 2:

## Linear Regression using Python:

Linear Regression is usually the first machine learning algorithm that every data scientist comes across. It is a simple model but everyone needs to master it as it lays the foundation for other machine learning algorithms.

## Where can Linear Regression be used? 
It is a very powerful technique and can be used to understand the factors that influence profitability. It can be used to forecast sales in the coming months by analyzing the sales data for previous months. It can also be used to gain various insights about customer behaviour. By the end of the blog we will build a model which looks like the below picture i.e, determine a line which best fits the data.



## What is Linear Regression
The objective of a linear regression model is to find a relationship between one or more features(independent variables) and a continuous target variable(dependent variable). When there is only feature it is called `Uni-variate Linear Regression` and if there are multiple features, it is called `Multivariate Linear Regression`.

<p align="center">
  <img width="460" height="450" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/LR.png">
</p>



## Hypothesis of Linear Regression

<p align="center">
  <img width="460" height="450" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/eq.png">
</p>

## How do we determine the best fit line? 

The line for which the the error between the predicted values and the observed values is minimum is called the best fit line or the regression line. These errors are also called as residuals. The `residuals` can be visualized by the vertical lines from the observed data value to the `regression` line.

<p align="center">
  <img width="460" height="400" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/LR2.png">
</p>

<p align="center">
  <img width="460" height="400" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/eq2.png">
</p>


 ## **Classification** 

## Classification Algorithms:



<p align="center">
  <img width="460" height="400" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/Regression_vs_Classification.png">
</p>
                                     
## What is a Classification Problem?
We identify problem as classification problem when independent variables are continuous in nature and dependent variable is in categorical form i.e. in classes like positive class and negative class. The real life example of classification example would be, to categorize the mail as spam or not spam, to categorize the tumor as malignant or benign and to categorize the transaction as fraudulent or genuine. All these problem’s answers are in categorical form i.e. Yes or No. and that is why they are two class classification problems.


<p align="center">
  <img width="460" height="400" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/1_-a_J9I0cr0BoJRc_6MhJog.png">
</p>


Although, sometime we come across more than 2 classes and still it is a classification problem. These types of problems are known as multi class classification problems.



## Logistic Regression: 

Logistic Regression is one of the basic and popular algorithm to solve a classification problem. It is named as ‘Logistic Regression’, because it’s underlying technique is quite the same as Linear Regression. The term “Logistic” is taken from the Logit function that is used in this method of classification.



## Why not use Linear Regression?
Suppose we have a data of tumor size vs its malignancy. As it is a classification problem, if we plot, we can see, all the values will lie on 0 and 1. And if we fit best found regression line, by assuming the threshold at 0.5, we can do line pretty reasonable job.


<p align="center">
  <img width="460" height="400" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/22.png">
</p>

We can decide the point on the x axis from where all the values lie to its left side are considered as negative class and all the values lie to its right side are positive class.

<p align="center">
  <img width="460" height="400" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/33.jpeg">
</p>

But what if there is an outlier in the data. Things would get pretty messy. For example, for 0.5 threshold,


<p align="center">
  <img width="460" height="400" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/44.png">
</p>


If we fit best found regression line, it still won’t be enough to decide any point by which we can differentiate classes. It will put some positive class examples into negative class. The green dotted line (Decision Boundary) is dividing malignant tumors from benign tumors but the line should have been at a yellow line which is clearly dividing the positive and negative examples. So just a single outlier is disturbing the whole linear regression predictions. And that is where logistic regression comes into a picture.


## Logistic Regression Algorithm
As discussed earlier, to deal with outliers, Logistic Regression uses Sigmoid function.
An explanation of logistic regression can begin with an explanation of the standard logistic function. The logistic function is a Sigmoid function, which takes any real value between zero and one. It is defined as

<p align="center">
  <img width="260" height="200" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/55.png">
</p>



And if we plot it, the graph will be S curve,

<p align="center">
  <img width="460" height="300" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/66.png">
</p>

Let’s consider t as linear function in a univariate regression model.

<p align="center">
  <img width="260" height="200" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/77.png">
</p>

So the Logistic Equation will become:

<p align="center">
  <img width="260" height="200" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/88.png">
</p>

Now, when logistic regression model come across an outlier, it will take care of it.

But sometime it will shift its y axis to left or right depending on outliers positions.

## What is Decision Boundary?

Decision boundary helps to differentiate probabilities into positive class and negative class.

## Non Linear Decision Boundary
<p align="center">
  <img width="460" height="300" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/db.png">
</p>

## Linear Decision Boundary

<p align="center">
  <img width="460" height="300" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/db2.png">
</p>

## Classification and Regression Trees
Decision Trees are an important type of algorithm for predictive modeling machine learning.

The representation of the decision tree model is a `binary tree`. This is your binary tree from algorithms and data structures, nothing too fancy. Each node represents a single input variable (x) and a split point on that variable (assuming the variable is numeric).

<p align="center">
  <img width="460" height="300" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/tree.png">
</p>

The leaf nodes of the tree contain an output variable (y) which is used to make a prediction. Predictions are made by walking the splits of the tree until arriving at a leaf node and output the class value at that leaf node.

Trees are fast to learn and very fast for making predictions. They are also often accurate for a broad range of problems and do not require any special preparation for your data.


## Naive Bayes

Naive Bayes is a simple but surprisingly powerful algorithm for predictive modeling.

The model is comprised of two types of probabilities that can be calculated directly from your training data: 
```
1) The probability of each class; and 
2) The conditional probability for each class given each x value. 

```
Once calculated, the probability model can be used to make predictions for new data using Bayes Theorem. When your data is real-valued it is common to assume a Gaussian distribution (bell curve) so that you can easily estimate these probabilities.

<p align="center">
  <img width="460" height="300" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/bayes.png">
</p>


Naive Bayes is called naive because it assumes that each input variable is `independent`. This is a strong assumption and unrealistic for real data, nevertheless, the technique is very effective on a large range of complex problems.

## K-Nearest Neighbors

The KNN algorithm is very simple and very effective. The model representation for KNN is the entire training dataset. 

```
Simple right?

```

Predictions are made for a new data point by searching through the entire training set for the K most similar instances (the neighbors) and summarizing the output variable for those K instances. For regression problems, this might be the mean output variable, for classification problems this might be the mode (or most common) class value.

The trick is in how to determine the similarity between the data instances. The simplest technique if your attributes are all of the same scale (all in inches for example) is to use the Euclidean distance, a number you can calculate directly based on the differences between each input variable.

<p align="center">
  <img width="460" height="300" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/knn.png">
</p>

KNN can require a lot of memory or space to store all of the data, but only performs a calculation (or learn) when a prediction is needed, just in time. You can also update and curate your training instances over time to keep predictions accurate.

The idea of distance or closeness can break down in very high dimensions (lots of input variables) which can negatively affect the performance of the algorithm on your problem. This is called the curse of dimensionality. It suggests you only use those input variables that are most relevant to predicting the output variable.

## Learning Vector Quantization

A downside of K-Nearest Neighbors is that you need to hang on to your entire training dataset. The Learning Vector Quantization algorithm (or LVQ for short) is an artificial neural network algorithm that allows you to choose how many training instances to hang onto and learns exactly what those instances should look like.

<p align="center">
  <img width="460" height="300" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/vector.png">
</p>

The representation for LVQ is a collection of codebook vectors. These are selected randomly in the beginning and adapted to best summarize the training dataset over a number of iterations of the learning algorithm. After learned, the codebook vectors can be used to make predictions just like K-Nearest Neighbors. The most similar neighbor (best matching codebook vector) is found by calculating the distance between each codebook vector and the new data instance. The class value or (real value in the case of regression) for the best matching unit is then returned as the prediction. Best results are achieved if you rescale your data to have the same range, such as between 0 and 1.

If you discover that KNN gives good results on your dataset try using LVQ to reduce the memory requirements of storing the entire training dataset.

## Support Vector Machines

Support Vector Machines are perhaps one of the most popular and talked about machine learning algorithms.

A hyperplane is a line that splits the input variable space. In SVM, a hyperplane is selected to best separate the points in the input variable space by their class, either class 0 or class 1. In two-dimensions, you can visualize this as a line and let’s assume that all of our input points can be completely separated by this line. The SVM learning algorithm finds the coefficients that results in the best separation of the classes by the hyperplane.

<p align="center">
  <img width="460" height="300" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/svm.jpeg">
</p>

The distance between the hyperplane and the closest data points is referred to as the margin. The best or optimal hyperplane that can separate the two classes is the line that has the largest margin. Only these points are relevant in defining the hyperplane and in the construction of the classifier. These points are called the support vectors. They support or define the hyperplane. In practice, an optimization algorithm is used to find the values for the coefficients that maximizes the margin.

SVM might be one of the most powerful `out-of-the-box` classifiers and worth trying on your dataset.

## Bagging and Random Forest

Random Forest is one of the most popular and most powerful machine learning algorithms. It is a type of ensemble machine learning algorithm called Bootstrap Aggregation or bagging.

The bootstrap is a powerful statistical method for estimating a quantity from a data sample. Such as a mean. You take lots of samples of your data, calculate the mean, then average all of your mean values to give you a better estimation of the true mean value.

In bagging, the same approach is used, but instead for estimating entire statistical models, most commonly decision trees. Multiple samples of your training data are taken then models are constructed for each data sample. When you need to make a prediction for new data, each model makes a prediction and the predictions are averaged to give a better estimate of the true output value.


<p align="center">
  <img width="460" height="300" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/forest.png">
</p>

Random forest is a tweak on this approach where decision trees are created so that rather than selecting optimal split points, suboptimal splits are made by introducing randomness.

The models created for each sample of the data are therefore more different than they otherwise would be, but still accurate in their unique and different ways. Combining their predictions results in a better estimate of the true underlying output value.

If you get good results with an algorithm with high variance (like decision trees), you can often get better results by bagging that algorithm.


## Boosting and AdaBoost

Boosting is an ensemble technique that attempts to create a strong classifier from a number of weak classifiers. This is done by building a model from the training data, then creating a second model that attempts to correct the errors from the first model. Models are added until the training set is predicted perfectly or a maximum number of models are added.

AdaBoost was the first really successful boosting algorithm developed for binary classification. It is the best starting point for understanding boosting. Modern boosting methods build on AdaBoost, most notably stochastic gradient boosting machines.

<p align="center">
  <img width="460" height="300" src="https://github.com/NeloyNSU/Workshop-on-Machine-Learning-2018/blob/master/Images/boosting.jpeg">
</p>

AdaBoost is used with short decision trees. After the first tree is created, the performance of the tree on each training instance is used to weight how much attention the next tree that is created should pay attention to each training instance. Training data that is hard to predict is given more weight, whereas easy to predict instances are given less weight. Models are created sequentially one after the other, each updating the weights on the training instances that affect the learning performed by the next tree in the sequence. After all the trees are built, predictions are made for new data, and the performance of each tree is weighted by how accurate it was on training data.

Because so much attention is put on correcting mistakes by the algorithm it is important that you have clean data with outliers removed.


## Some Important Methodologies:
- **DataSets**
- **DataPreprocessing**
- **Data Visualization**
- **Prediction**
- **Model Validation**


## DataSets:

There are three types of data sets – Training, Dev and Test that are used at various stage of development. Training dataset is the largest of three of them, while test data functions as seal of approval and you don’t need to use till the end of the development.

## What is Test Dataset in ML?

This is the data typically used to provide an unbiased evaluation of the final that are completed and fit on the training dataset. Actually, such data is used for testing the model whether it is responding or working appropriately or not.

## Data Preprocessing
Before Training the dataset, regularization needs to apply to remove noisy data and transform categorical values to binary values.

- **Check Null values**
- **Removing NULL Values and Empty Rows**
- **Dummy all the categorical variables used for modeling** 

## Data Visualization
Visualize the data with features to get a clear idea.

- **Finding Correlation**
- **Feature Engineering**
- **Dataset Splitting**


## Prediction
### What does Prediction mean in Machine Learning?
“Prediction” refers to the output of an algorithm after it has been trained on a historical dataset and applied to new data when you’re trying to forecast the likelihood of a particular outcome, such as whether or not a customer will churn in 30 days. The algorithm will generate probable values for an unknown variable for each record in the new data, allowing the model builder to identify what that value will most likely be.

The word “prediction” can be misleading. In some cases, it really does mean that you are predicting a future outcome, such as when you’re using machine learning to determine the next best action in a marketing campaign. Other times, though, the “prediction” has to do with, for example, whether or not a transaction that already occurred was fraud. In that case, the transaction already happened, but you’re making an educated guess about whether or not it was legitimate, allowing you to take the appropriate action.

### Why are Predictions important?

Machine learning model predictions allow businesses to make highly accurate guesses as to the likely outcomes of a question based on historical data, which can be about all kinds of things – customer churn likelihood, possible fraudulent activity, and more. These provide the business with insights that result in tangible business value. For example, if a model predicts a customer is likely to churn, the business can target them with specific communications and outreach that will prevent the loss of that customer.


## Model Validation

### Metrics

Choice of metrics influences how the performance of machine learning algorithms is measured and compared. They influence how you weight the importance of different characteristics in the results and your ultimate choice of which algorithm to choose. In this project, as both Regression and Classification problem was handled so both types of matrices were used. 

### Regression Problem Metrices:
- **Mean Absolute Error.**
- **Mean Squared Error.**
- **R^2**

### Mean Absolute Error
The Mean Absolute Error (or MAE) is the sum of the absolute differences between predictions and actual values. It gives an idea of how wrong the predictions were.
The measure gives an idea of the magnitude of the error, but no idea of the direction (e.g. over or under predicting).

### Mean Squared Error
The Mean Squared Error (or MSE) is much like the mean absolute error in that it provides a gross idea of the magnitude of the error.
Taking the square root of the mean squared error converts the units back to the original units of the output variable and can be meaningful for description and presentation. This is called the Root Mean Squared Error (or RMSE).

### R^2 Metric
The R^2 (or R Squared) metric provides an indication of the goodness of fit of a set of predictions to the actual values. In statistical literature, this measure is called the coefficient of determination.

## Classification Problem Metrices:
- **Classification Accuracy.**
- **Area Under ROC Curve.**

### Classification Accuracy
Classification accuracy is the number of correct predictions made as a ratio of all predictions made.
This is the most common evaluation metric for classification problems, it is also the most misused. It is really only suitable when there is an equal number of observations in each class (which is rarely the case) and that all predictions and prediction errors are equally important, which is often not the case.


### Area Under ROC Curve
The area under ROC Curve (or AUC for short) is a performance metric for binary classification problems.
The AUC represents a model’s ability to discriminate between positive and negative classes. An area of 1.0 represents a model that made all predictions perfectly. An area of 0.5 represents a model as good as random. 
ROC can be broken down into sensitivity and specificity. A binary classification problem is really a trade-off between sensitivity and specificity.

- Sensitivity is the true positive rate also called the recall. It is the number of instances from the positive (first) class that actually predicted correctly.
- Specificity is also called the true negative rate. Is the number of instances from the negative class (second) class that was actually predicted correctly.
