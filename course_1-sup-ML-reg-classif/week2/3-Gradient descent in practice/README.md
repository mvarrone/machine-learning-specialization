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

## Feature scaling (Part 2)


