## Feature scaling (Part 1)

- Let's take a look at some techniques that make Gradient Descent work much better:

    In this case, we will see a technique called **Feature scaling** that will enable Gradient Descent to run much faster. 
    
    Let's start by taking a look at the relationship between the size of a feature that is how big are the numbers for that feature and the size of its associated parameter. 
    
    ## Example:
    
    As a concrete example, let's predict the price of a house using 2 features: $x_{1}$ (the size of the house) and $x_{2}$ (the number of bedrooms). Let's say that $x_{1}$ typically ranges from 300 to 2000 square feet and $x_{2}$ in the dataset ranges from 0 to 5 bedrooms. 
    
    So, for this example, $x_{1}$ takes on a relatively **large range of values** and $x_{2}$ takes on a relatively **small range of values**.
    
    Now, let's take an example of a house that has a size of 2000 square feet has 5 bedrooms and a price of 500k or $500,000 dollars. For this one training example, what do you think are reasonable values for the size of parameters $w_{1}$ and $w_{2}$? 
    
    Well, let's look at 2 possible set of parameters. 
    
    a) Say $w_{1}$ is 50 and $w_{2}$ is 0.1 and *b* is 50 for the purposes of discussion.

    - So, in this case, the estimated price, in thousands of dollars, is 100k or $100,000 dollars plus 0.5 k plus 50 k which is slightly over $100,000 dollars. So, that's clearly very far from the actual price of $500,000 dollars and so **this is not a very good set of parameter choices for $w_{1}$ and $w_{2}$**
    
    Now, let's take a look at the another possibility: 
    
    b) Say $w_{1}$ and $w_{2}$ were the other way around: $w_{1}$ is 0.1 and $w_{2}$ is 50 and *b* is still also 50. 
    
    - In this choice of $w_{1}$ and $w_{2}$, $w_{1}$ is relatively **small** and $w_{2}$ is relatively **large**: 50 is much bigger than 0.1. 
    
    So, here the predicted price is 0.1 times 2000 plus 50 times 5 plus 50: The first term becomes 200k, the second term becomes 250k and then plus 50. 
    
    **So, this version of the model predicts a price of $500,000 dollars which is a much more reasonable estimate and happens to be the same price as the true price of the house.**

    ## Conclusions

    1) So, hopefully you might notice that when a possible range of values of a feature is **large**, like the size in square feet (which goes all the way up to 2000), it is more likely that a good model will learn to choose a relatively **small** parameter value, like 0.1. 
    
    2) Likewise, when the possible values of the feature are **small**, like the number of bedrooms, then a reasonable value for its parameters will be relatively **large**, like 50. 

    ![alt text](./images_for_3/image1.png)
    
## Relating to Gradient Descent

- So how does this relate to Gradient Descent? Well, let's take a look at the *scatter plot* of the features where the size square feet is the horizontal axis $x_{1}$ and the number of bedrooms $x_{2}$ is on the vertical axis. If you plot the training data, you notice that the horizontal axis is on a much larger scale or much larger range of values compared to the vertical axis.

    Next, let's look at how the cost function J(w, b) might look in a *contour plot*. You might see a *contour plot* where the horizontal axis has a much narrower range, say between 0 and 1, whereas the vertical axis takes on much larger values, say between 10 and 100. So, the contours form ovals or ellipses and they are short on one side and longer on the other. And this is because a very small change to $w_{1}$ can have a very large impact on the estimated price and that's a very large impact on the cost function J(w, b) because $w_{1}$ tends to be multiplied by a very large number, the size in square feet. 

    ![alt text](./images_for_3/image2.png)
    
    In contrast, it takes a much larger change in $w_{2}$ in order to change the predictions much. And thus, small changes to $w_{2}$, don't change the cost function J(w, b) nearly as much. 

    ![alt text](./images_for_3/image3.png)
    
    So, where does this leave us? This is what might end up happening if you were to run Gradient Descent, if you were to use your training data as is. Because the contours are so tall and skinny, Gradient Descent may end up **bouncing back and forth** for a long time before it can finally find **its way to the global minimum**. 
    
    ![alt text](./images_for_3/image4.png) 

    ## Scaling the features

    In situations like this, a useful thing to do is to **scale the features**: This means performing some transformation of your training data so that $x_{1}$ say might now range from 0 to 1 and $x_{2}$ might also range from 0 to 1. So, the data points now look more like the next image shown below and you might notice that the scale of the plot on the bottom is now quite different than the one on top. 

    ![alt text](./images_for_3/image5.png) 
    
    The key point is that the rescaled **$x_{1}$ and $x_{2}$ are both now taking comparable ranges of values to each other**. 
    
    And if you run Gradient Descent on a cost function J(w, b) to find on this rescaled $x_{1}$ and $x_{2}$ using this transformed data, then the contours will look more like the next image shown below: more like circles and less tall and skinny and now, Gradient Descent can find a **much more direct path to the global minimum**.

    ![alt text](./images_for_3/image6.png) 

## Recap

- So, to recap, when you have different features that take on very different ranges of values, it can cause Gradient Descent to run slowly but re scaling the different features so they all take on comparable range of values because speed up Gradient Descent significantly. How do you actually do this? Let's take a look at that in the next Part.

## Feature scaling (Part 2): How to implement it

- Let's look at how you can implement Feature Scaling to take features that take on very different ranges of values and how to skill them to have comparable ranges of values to each other.

## Implementing Feature Scaling: 1. Dividing by the maximum value of the range

* Well, if $x_{1}$ ranges from 3 to 2000, one way to get a scale version of $x_{1}$ is to take each original $x_{1}$ value and divide by 2000, the **maximum of the range**. The scaled $x_{1}$ will range from 0.15 up to 1. Similarly, since $x_{2}$ ranges from 0 to 5, you can calculate a scaled version of $x_{2}$ by taking each original $x_{2}$ and dividing by five, which is again the maximum. So the scaled is $x_{2}$ will now range from 0 to 1. If you plot the scale to $x_{1}$ and $x_{2}$ on a graph, it might look like the following image:

# INSERTAR FOTO

## Implementing Feature Scaling: 2. Using the mean normalization

* You can also do what is called **mean normalization**: 

What this looks like is, you start with the original features and then you re-scale them so that both of them are **centered around zero**. 

Whereas before they only had values greater than zero, now they have both negative and positive values that may be usually between -1 and +1. 

To calculate the mean normalization of $x_{1}$, first find the average, also called the mean of $x_{1}$ on your training set, and let's call this mean Mu_1, with this being the Greek alphabets Mu. For example, you may find that the average of feature 1, Mu_1 is 600 square feet. Let's take each x_1, subtract the mean Mu_1, and then let's divide by the difference 2,000 minus 300, where 2,000 is the maximum and 300 the minimum, and if you do this, you get the normalized x_1 to range from negative 0.18-0.82. Similarly, to mean normalized x_2, you can calculate the average of feature 2. For instance, Mu_2 may be 2.3. Then you can take each x_2, subtract Mu_2 and divide by 5 minus 0. Again, the max 5 minus the mean, which is 0. The mean normalized x_2 now ranges from negative 0.46-0 54. If you plot the training data using the mean normalized x_1 and x_2, it might look like this. There's one last common re-scaling method call Z-score normalization. To implement Z-score normalization, you need to calculate something called the standard deviation of each feature. If you don't know what the standard deviation is, don't worry about it, you won't need to know it for this course. Or if you've heard of the normal distribution or the bell-shaped curve, sometimes also called the Gaussian distribution, this is what the standard deviation for the normal distribution looks like. But if you haven't heard of this, you don't need to worry about that either. But if you do know what is the standard deviation, then to implement a Z-score normalization, you first calculate the mean Mu, as well as the standard deviation, which is often denoted by the lowercase Greek alphabet Sigma of each feature. For instance, maybe feature 1 has a standard deviation of 450 and mean 600, then to Z-score normalize x_1, take each x_1, subtract Mu_1, and then divide by the standard deviation, which I'm going to denote as Sigma 1. What you may find is that the Z-score normalized x_1 now ranges from negative 0.67-3.1.

- Similarly, if you calculate the second features standard deviation to be 1.4 and mean to be 2.3, then you can compute x_2 minus Mu_2 divided by Sigma_2, and in this case, the Z-score normalized by x_2 might now range from negative 1.6-1.9. If you plot the training data on the normalized x_1 and x_2 on a graph, it might look like this. As a rule of thumb, when performing feature scaling, you might want to aim for getting the features to range from maybe anywhere around negative one to somewhere around plus one for each feature x. But these values, negative one and plus one can be a little bit loose. If the features range from negative three to plus three or negative 0.3 to plus 0.3, all of these are completely okay. If you have a feature x_1 that winds up being between zero and three, that's not a problem. You can re-scale it if you want, but if you don't re-scale it, it should work okay too. Or if you have a different feature, x_2, whose values are between negative 2 and plus 0.5, again, that's okay, no harm re-scaling it, but it might be okay if you leave it alone as well. But if another feature, like x_3 here, ranges from negative 100 to plus 100, then this takes on a very different range of values, say something from around negative one to plus one. You're probably better off re-scaling this feature x_3 so that it ranges from something closer to negative one to plus one. Similarly, if you have a feature x_4 that takes on really small values, say between negative 0.001 and plus 0.001, then these values are so small. That means you may want to re-scale it as well. Finally, what if your feature x_5, such as measurements of a hospital patients by the temperature ranges from 98.6-105 degrees Fahrenheit? In this case, these values are around 100, which is actually pretty large compared to other scale features, and this will actually cause gradient descent to run more slowly. In this case, feature re-scaling will likely help. There's almost never any harm to carrying out feature re-scaling. When in doubt, I encourage you to just carry it out. That's it for feature scaling. With this little technique, you'll often be able to get gradient descent to run much faster. That's features scaling. With or without feature scaling, when you run gradient descent, how can you know, how can you check if gradient descent is really working? If it is finding you the global minimum or something close to it. In the next video, let's take a look at how to recognize if gradient descent is converging, and then in the video after that, this will lead to discussion of how to choose a good learning rate for gradient descent.