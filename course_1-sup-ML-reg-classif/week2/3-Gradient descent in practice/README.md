## Feature scaling (Part 1/2)

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

## Feature scaling (Part 2/2): 3 possible implementations

- Let's look at how you can implement Feature Scaling to take features that take on very different ranges of values and skill them to have comparable ranges of values to each other.

## 1. Using the maximum value of the range

- Basically, this implementation divides every value of a feature by the maximum value of the range

- It can be calculated applying the following formula:

$$
x_{1, scaled} = \frac{x_{1}}{max(x_{1})}
$$

## Example

- If a feature $x_{1}$ ranges from 3 to 2000

$$
3 \leq x_{1} \leq 2000
$$

One way to get a scaled version of $x_{1}$ is to take each original $x_{1}$ value and divide by 2000, the **maximum of the range**.

So, the scaled $x_{1, scaled}$ will range, in this case, from 0.15 up to 1

$$
0.15 \leq x_{1, scaled} \leq 1
$$

- Similarly, for another feature $x_{2}$, since $x_{2}$ ranges from 0 to 5

$$
0 \leq x_{2} \leq 5
$$

You can calculate a scaled version of $x_{2}$, denoted as $x_{2, scaled}$, by taking each original $x_{2}$ and dividing by 5, which is again the maximum value of this feature. 

So, the scaled feature $x_{2, scaled}$ will now range from 0 to 1. 

$$
0 \leq x_{2, scaled} \leq 1
$$

Now, if you plot the scaled versions, $x_{1, scaled}$ and $x_{2, scaled}$ on a graph it might look like the following image:

![alt text](./images_for_3/image7.png) 

## 2. Using the mean normalization

- What this implementation means is to perform a re-scale of all of the features so that all of them are **centered around zero**

So, whereas in the previous implementation (dividing by the maximium value) they only had values greater than 0, now they have both negative and positive values that may be usually between -1 and +1. 

- So, to calculate the mean normalization of a feature $x_{1}$ we must find the average of it, also called the mean of $x_{1}$, on the training set. This is denoted as $\mu_{1}$, with this being the Greek alphabets $\mu$. 

## Example

You may find that the average of feature 1, $\mu_{1} = 600$ square feet. 

So, let's take each $x_{1}$, subtract the mean $\mu_{1}$, and then let's divide by the difference **maximum** minus **minimum**, which is 2000 minus 300, in this case. 

By doing this, you get the normalized $x_{1}$ to range from -0.18 to 0.82

$$
-0.18 \leq x_{1} \leq 0.82
$$

Similarly, to mean normalized $x_{2}$, you can calculate the average of feature 2. 

For instance, let say $\mu_{2} = 2.3$. Then, you can take each $x_{2}$, subtract $\mu_{2}$ and divide by 5 minus 0. Again, the max is 5 and 0 is the min. The mean normalized $x_{2}$ now ranges from -0.46 to 0.54

$$
-0.46 \leq x_{2} \leq 0.54
$$

If you plot the training data using the mean normalized $x_{1}$ and $x_{2}$, it might look like the image shown below:

![alt text](./images_for_3/image8.png) 

## 3. Using the Z-score normalization

- To implement a Z-score normalization, you first need to calculate the mean $\mu$ as well as the standard deviation $\sigma$ of each feature.

- Then, to Z-score normalize a feature $x_{1}$, we can apply the next formula:

$$
x_{1} = \frac{x_{1} - \mu_{1}}{\sigma_{1}}
$$

## Example

### Calculating Z-score normalization for feature 1

For instance, maybe feature 1 $x_{1}$ has a standard deviation $\sigma_{1} = 450$ and mean $\mu_{1} = 600$

What you may find is that the Z-score normalized $x_{1}$ now ranges from -0.67 to 3.1

$$
-0.67 \leq x_{1} \leq 3.1
$$

### Calculating Z-score normalization for feature 2

Similarly, if you calculate the second feature's standard deviation $\sigma_{2} = 1.4$ and $\mu_{2} = 2.3$, then the Z-score normalized by $x_{2}$ might now range from -1.6 to 1.9

$$
-1.6 \leq x_{2} \leq 1.9
$$

If you plot the training data on the normalized $x_{1}$ and $x_{2}$ on a graph, it might look like the following image:

![alt text](./images_for_3/image9.png) 

## Aim for Feature Scaling

As a rule of thumb, when performing Feature Scaling, you might want to aim for getting the features to range from maybe anywhere around -1 to somewhere around +1 for each feature x but these values, -1 and +1, can be a little bit loose. 

$$
-1 \leq x_{j} \leq 1 \; \text{for each feature}\; x_{j}
$$

## When to re-scale and when not

### No need for re-scaling

- If the features range from -3 to +3 or -0.3 to +0.3, all of these are completely OK.
- If you have a feature $x_{1}$ that winds up being between 0 and 3, that's not a problem: You can re-scale it if you want, but if you don't want to re-scale it, it should work okay too. 
- If you have a different feature, $x_{2}$, whose values are between -2 and +0.5: again, that's okay, no harm re-scaling it, but it might be okay if you leave it alone as well. 

### Re-scale needed

- If another feature, like $x_{3}$, ranges from -100 and +100, then this takes on a very different range of **too large values**, say something from around -1 to +1, you are probably better off re-scaling this feature $x_{3}$ so that it ranges from something closer to -1 to +1.
- Similarly, if you have a feature $x_{4}$ that takes on really small values, say between -0.001 and +0.001, then these **values are so small** which means you may want to re-scale it as well. 

- Finally, what if your feature $x_{5}$, such as measurements of a hospital patients by the temperature ranges from 98.6 to 105 degrees Fahrenheit? In this case, these values are around 100, which is actually **pretty large compared to other scale features**, and this will actually cause gradient descent to run more slowly. In this case, feature re-scaling will likely help. 

![alt text](./images_for_3/image10.png)

## Some advice on rescaling

There's almost never any harm to carrying out feature Re-scaling: When in doubt, I encourage you to just carry it out. 
With this little technique, you'll often be able to get gradient descent to run much faster.

## Next video

With or without Feature Scaling, when you run Gradient Descent, how can you check if Gradient Descent is really working? If it is finding you the global minimum or something close to it. In the next video, let's take a look at how to recognize if Gradient Descent is converging, and then in the video after that, this will lead to discussion of how to choose a good learning rate $\alpha$ for Gradient Descent.