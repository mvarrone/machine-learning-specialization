## The problem of overfitting

Sometimes in an application, Logistic Regression can run into a problem called *overfitting*, which can cause it to perform poorly.

Our goal when creating a model is to be able to use the model to predict outcomes correctly for new examples. A model which does this is said to generalize well. 

When a model fits the training data well but does not work well with new examples that are not in the training set we say this is an example of Overfitting (high variance)

## Overfitting (high variance)

Overfitting occurs when a model learns the training data too well, capturing noise or random fluctuations that are not representative of the true underlying pattern in the data. This can lead to poor performance on new, unseen data because the model's learned behavior is too specific to the training examples.

The intuition behind overfitting or high-variance is that the algorithm is trying very hard to fit every single training example. It turns out that if your training set were just even a little bit different, then the function that the algorithm fits could end up being totally different.

If two different machine learning engineers were to fit this fourth-order polynomial model, to just slightly different datasets, they couldn't end up with totally different predictions or highly variable predictions. That's why we say the algorithm has high variance.

## Underfitting (high bias)

Underfitting, on the other hand, happens when a model is too simple to capture the underlying structure of the data. It fails to learn the patterns in the training data and performs poorly both on the training set and on new data. This often occurs when the model is too basic or lacks the capacity to represent the complexity of the data.

## Generalization (just right: there is neither underfit nor overfit)

Generalization refers to the ability of a model to perform well on new, unseen data that it hasn't been trained on. A model that generalizes well is able to make accurate predictions or classifications on examples it hasn't encountered before. Generalization is a desirable property in machine learning, as it indicates that the model has learned the underlying patterns in the data rather than just memorizing the training examples.

## Underfitting, Generalization and Overfitting for Regression

![alt text](./images_for_07/image1.png)

## Underfitting, Generalization and Overfitting for Classification

![alt text](./images_for_07/image2.png)

## Addressing overfitting

There are some ways in order to addressing overfitting:

## 1. Collect more data

One way to address this problem is to collect more training data: If the training set is larger, then the learning algorithm will learn to fit a function that is less wiggly.

Disadvantage: The problem may be that getting more data is not always an option

## 2. Feature selection

A second option for addressing overfitting is to see if you can use fewer features:

It turns out that if you have a lot of features but don't have enough training data, then your learning algorithm may also overfit to your training set.

Now, instead of using all features, if could pick just a subset of the most useful and relevant ones. By doing it,you may find that your model no longer overfits as badly. Choosing the most appropriate set of features to use is sometimes also called Feature selection. 

Disadvantage: One disadvantage of Feature Selection is that by using only a subset of the features, the algorithm is throwing away some of the information that you have about the houses

Note: Later in Course 2, you will also see some algorithms for automatically choosing the most appropriate set of features to use for our prediction task

## 3. Regularization

Regularization is a way to more gently reduce the impacts of some of the features without doing something as harsh as eliminating it outright. What regularization does is encourage the learning algorithm to shrink the values of the parameters without necessarily demanding that the parameter is set to exactly 0.

So what regularization does is it lets you keep all of your features, but they just prevents the features from having an overly large effect, which is what sometimes can cause overfitting. By the way, by convention, we normally just reduce the size of the $w_{j}$ parameters, that is $w_{1}$ through $w_{n}$. It doesn't make a huge difference whether you regularize the parameter *b* as well, you could do so if you want or not if you don't. In practice, it should make very little difference whether you also regularize b or not

This is a very useful technique for training learning algorithms including neural networks specifically.

Example:

It turns out that even if you fit a higher order polynomial the shown in the image below, so long as you can get the algorithm to use smaller parameter values: $w_{1}$, $w_{2}$, $w_{3}$, $w_{4}$ you end up with a curve that ends up fitting the training data much better.

![alt text](./images_for_07/image3.png)

## Optional lab 17: Overfitting

## Cost function with regularization

