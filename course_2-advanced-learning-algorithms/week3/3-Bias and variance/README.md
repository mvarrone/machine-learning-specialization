## Diagnosing bias and variance

![alt text](./img/image1.png)

The typical workflow of developing a machine learning system is that you have an idea and you train the model, and you almost always find that it doesn't work as well as you wish yet. 

A key to the process of building a machine learning system is how to decide what to do next in order to improve his performance. I've found across many different applications that **looking at the bias and variance** of a learning algorithm gives you very good guidance on what to try next.

Let's take a look at what this means. You might remember this example from the first course on Linear Regression where given this dataset, if you were to fit a straight line to it, it doesn't do that well. 

We said that this algorithm has high bias or that it underfits this dataset. If you were to fit a fourth-order polynomial, then it has high-variance or it overfits. In the middle if you fit a quadratic polynomial, then it looks pretty good. Then I said that was just right. Because this is a problem with just a single feature $x$, we could plot the function $f$ and look at it like this but if you had more features, you can't plot $f$ and visualize whether it's doing well as easily. 

## Systematic way of diagnose 
Instead of trying to look at plots like this, a more systematic way to diagnose or to find out if your algorithm has high bias or high variance will be **to look at the performance of your algorithm on the training set and on the cross validation set**. 

### 1st case: High bias - Underfitting

In particular, let's look at the example on the left. 

#### Performance on the training set

If you were to compute $J_{train}$, how well does the algorithm do on the training set? Not that well. I'd say  $J_{train}$ here would be high because there are actually pretty large errors between the examples and the actual predictions of the model. 

#### Performance on the cross validation set

How about $J_{cv}$? 

$J_{cv}$ would be if we had a few new examples that the algorithm had not previously seen. Here the algorithm also doesn't do that well on examples that it had not previously seen, so $J_{cv}$ will also be high. 

#### Description

One characteristic of an algorithm with high bias, something that is under fitting, is that it's not even doing that well on the training set. 
When $J_{train}$ is high, that is your strong indicator that this algorithm has high bias. 

### 2nd case: High variance - Overfitting

Let's now look at the example on the right. 

#### Performance on the training set

If you were to compute $J_{train}$ we could ask how well is this doing on the training set? Well, it's actually doing great on the training set. It is fitting the training data really well. So, $J_{train}$ here will be low

#### Performance on the cross validation set

But if you were to evaluate this model on other houses not in the training set, then you find that $J_{cv}$, the cross-validation error, will be quite high. 

#### Description

A characteristic signature or a characteristic Q that your algorithm has high variance will be of $J_{cv}$ is much higher than $J_{train}$. 
In other words, it does much better on data it has seen than on data it has not seen. 

This turns out to be a strong indicator that your algorithm has high variance. 

Again, the point of what we're doing is that I'm computing $J_{train}$ and $J_{cv}$ and seeing 

- if $J_{train}$ is high 

    or

- if $J_{cv}$ is much higher than $J_{train}$

This gives you a sense, even if you can't plot the function $f$, of whether your algorithm has high bias or high variance. 

### 3rd case: Just right

Finally, the case in the middle

#### Performance on the training set

If you look at $J_{train}$, it's pretty low, so this is doing quite well on the training set. 

#### Performance on the cross validation set

If you were to look at a few new examples, like those from, say, your cross-validation set, you find that $J_{cv}$ is also pretty low. 

The fact of $J_{train}$ not being too high indicates this doesn't have a high bias problem and also the fact of $J_{cv}$ not being much worse than $J_{train}$ indicates that it doesn't have a high variance problem either which is why the quadratic model seems to be a pretty good one for this application. 

### Summary 

To summarize 

- when $d=1$, for a linear polynomial, $J_{train}$ was high and $J_{cv}$ was also high
- when $d=4$, $J_{train}$ was low but $J_{cv}$ is high
- when $d=2$, both were pretty low

![alt text](./img/image2.png)

Let's now take a different view on bias and variance. 

In particular, based on this new slide I'd like to show you how $J_{train}$ and $J_{cv}$ vary as a function of the degree of the polynomial you're fitting

So, let me draw a figure where the horizontal axis, $d$, will be the degree of polynomial that we're fitting to the data.

Over on the left we'll correspond to a small value of $d$, like $d=1$, which corresponds to fitting a straight line. 

Over to the right we'll correspond to, say, $d=4$ or even higher values of d. where fitting this high order polynomial.

So if you were to plot $J_{train}(w, b)$ as a function of $d$, the degree of polynomial, ($J_{train}(w, b)$ vs $d$) what you find is that as you fit a higher and higher degree polynomial, here I'm assuming we're not using regularization, but as you fit a higher and higher order polynomial, the training error will tend to go down because when you have a very simple linear function, it doesn't fit the training data that well, when you fit a quadratic function or third order polynomial or fourth-order polynomial, it fits the training data better and better.
So, as the degree of polynomial increases, $J_{train}$ will typically go down. 

Next, let's look at $J_{cv}$, which is how well does it do on data that it did not get to fit to? What we saw was when $d=1$, when the degree of polynomial was very low, $J_{cv}$ was pretty high because it underfits, so it didn't do well on the cross validation set. 

Here on the right as well, when the degree of polynomial is very large, say $d=4$, it doesn't do well on the cross-validation set either, and so it's also high. 

But if $d$ was in-between say, a second-order polynomial ($d=2$) then it actually did much better. 

And so if you were to vary the degree of polynomial, you'd actually get a curve that looks like this, which comes down and then goes back up where 

- if the degree of polynomial is too low then it underfits and so doesn't do on the cross validation set

    and 

- if the degree of polynomial is too high then it overfits and also doesn't do well on the cross validation set. 

Is only if it's somewhere in the middle, that is just right, which is why the second-order polynomial in our example ends up with a lower cross-validation error and neither high bias nor high-variance problems.

![alt text](./img/image3.png)

To summarize, how do you diagnose bias and variance in your learning algorithm? 

## Diagnosing high bias

If your learning algorithm has high bias or it has underfitted data, the key indicator will be if $J_{train}$ is high. And so that corresponds to this leftmost portion of the curve, which is where $J_{train}$ is high. 
And usually you have $J_{train}$ and $J_{cv}$ will be close to each other. 

## Diagnosing high variance

How do you diagnose if you have high variance? Well, the key indicator for high-variance will be if $J_{cv}$ is much greater than $J_{train}$.
This rightmost portion of the plot is where $J_{cv}$ is much greater than $J_{train}$. And usually $J_{train}$ will be pretty low but the key indicator is whether $J_{cv}$ is much greater than $J_{train}$. That's what happens when we had fit a very high order polynomial to this small dataset. 

## Diagnosing the fact of having high bias and high variance at the same time

And even though we've just seen bias and variance, it turns out, in some cases, is possible to simultaneously have high bias and have high-variance. You won't see this happen that much for Linear Regression, but it turns out that if you're training a neural network, there are some applications where unfortunately you have high bias and high variance. 

One way to recognize that situation will be if $J_{train}$ is high, so you're not doing that well on the training set, but even worse, the cross-validation error is again, even much larger than the training set. 

The notion of high bias and high variance, it doesn't really happen for Linear models applied to 1D but to give intuition about what it looks like, it would be as if for part of the input you had a very complicated model that overfit, so it overfits to part of the inputs. But then for some reason, for other parts of the input, it doesn't even fit the training data well, and so it underfits for part of the input. 

In this example, which looks artificial because it's a single feature input, we fit the training set really well and we overfit in part of the input, and we don't even fit the training data well, and we underfit the part of the input and that's how in some applications you can unfortunate end up with both high bias and high variance. 

The indicator for that will be if the algorithm does poorly on the training set, and it even does much worse than on the training set. 

For most learning applications, you probably have primarily a high bias or high variance problem rather than having both at the same time but it is possible sometimes they're both at the same time. 

I know that there's a lot of process, there are a lot of concepts on the slides, but the key takeaways are:

- high bias means is not even doing well on the training set

    and

- high variance means it does much worse on the cross validation set than on the training set

Whenever I'm training a machine learning algorithm, I will almost always try to figure out to what extent the algorithm has a high bias or underfitting vs a high-variance when overfitting problem. 

This will give good guidance, as we'll see later this week, on how you can improve the performance of the algorithm. But first, let's take a look at how regularization effects the bias and variance of a learning algorithm because that will help you better understand when you should use regularization. Let's take a look at that in the next video.

## Regularization and bias/variance

![alt text](./img/image4.png)

You saw in the last video how different choices of the degree of polynomial $d$ affects the bias in variance of your learning algorithm and therefore its overall performance. 

In this video, let's take a look at how regularization, specifically the choice of the regularization parameter lambda $\lambda$ affects the bias and variance and therefore the overall performance of the algorithm. 

This, it turns out, will be helpful for when you want to choose a good value of lambda $\lambda$ of the regularization parameter for your algorithm. 

Let's take a look. In this example, I'm going to use a fourth-order polynomial, but we're going to fit this model using regularization where here the value of lambda $\lambda$ is the regularization parameter that controls how much you trade-off keeping the parameters $w$ small vs fitting the training data well. 

## Case 1: when $\lambda$ is very large
Let's start with the example of setting lambda $\lambda$ to be a very large value. Say $\lambda=10 000$. 

If you were to do so, you would end up fitting a model that looks roughly like this (a horizontal line). Because if lambda $\lambda$ were very large, then the algorithm is highly motivated to keep these parameters $w$ very small and so you end up with $w_1$, $w_2$, etc really all of these parameters will be very close to zero. 

And so the model ends up being $f(x)\approx b$, a constant value, which is why you end up with a model like this. This model clearly has high bias and it underfits the training data because it doesn't even do well on the training set and $J_{train}$ is large. 

## Case 2: when $\lambda$ is very small or 0
Let's take a look at the other extreme. Let's say you set Lambda $\lambda$ to be a very small value. With a small value of Lambda $\lambda$, in fact, let's go to extreme of setting Lambda $\lambda$ equals zero. 

With that choice of Lambda $\lambda$, there is no regularization, so we're just fitting a fourth-order polynomial with no regularization and you end up with that curve that you saw previously that overfits the data. 

What we saw previously was when you have a model like this, $J_{train}$ is small, but $J_{cv}$ is much larger than $J_{train}$. For $J_{cv}$ is large this indicates we have high variance and it overfits this data. 

## Case 3: Some intermediate value for $\lambda$
It would be if you have some intermediate value of Lambda $\lambda$, not really large as 10,000 but not so small as zero that hopefully you get a model that looks like this, that is just right and fits the data well with small $J_{train}$ and small $J_{cv}$.

## Cross validation set helps to decide

If you are trying to decide what is a good value of Lambda $\lambda$ to use for the regularization parameter, cross-validation gives you a way to do so as well. 

Let's take a look at how we could do so. Just as a reminder, the problem we're addressing is if you're fitting a fourth-order polynomial, so that's the model and you're using regularization, how can you choose a good value of Lambda $\lambda$?

![alt text](./img/image5.png)

This would be procedure similar to what you had seen for choosing the degree of polynomial $d$ using cross-validation. Specifically, let's say we try to fit a model using $\lambda = 0$ and so we would minimize the cost function using $\lambda = 0$ and end up with some parameters $w_1, b_1$ and you can then compute the cross-validation error, $J_{cv}(w^{<1>}, b^{<1>})$. 

Now let's try a different value of Lambda $\lambda$. Let's say you try Lambda $\lambda$ equals $\lambda = 0.01$. Then again, minimizing the cost function gives you a second set of parameters $w_2, b_2$ and you can also see how well that does on the cross-validation set, and so on. 

Let's keep trying other values of Lambda $\lambda$ and in this example, I'm going to try doubling it to $\lambda = 0.02$ and so that will give you $J_{cv}(w^{<3>}, b^{<3>})$, and so on. 

Then let's double again and double again. After doubling a number of times, you end up with Lambda $\lambda$ approximately equal to $\lambda = 10$, and that will give you parameters $w_{12}, b_{12}$, and $J_{cv}(w^{<12>}, b^{<12>})$.

And by trying out a large range of possible values for Lambda $\lambda$, fitting parameters using those different regularization parameters, and then evaluating the performance on the cross-validation set, you can then try to pick what is the best value for the regularization parameter lambda.

Quickly, if in this example, you find that $J_{cv}(w^{<5>}, b^{<5>})$ has the lowest value of all of these different cross-validation errors, you might then decide to pick this value for Lambda $\lambda$, and so use $w_5, b_5$ as the chosen parameters. And finally, if you want to report out an estimate of the generalization error, you would then report out the test set error, $J_{test}(w^{<5>}, b^{<5>})$

## How the training and cross validation error vary as function of Lambda $\lambda$

![alt text](./img/image6.png)

To further hone intuition about what this algorithm is doing, let's take a look at how training error and cross validation error vary as a function of the parameter Lambda $\lambda$. 

In this figure, I've changed the x-axis again. Notice that the x-axis here is annotated with the value of the regularization parameter Lambda $\lambda$

### Case: $\lambda=0$

![alt text](./img/image7.png)

If we look at the extreme of $\lambda=0$ here on the left, that corresponds to not using any regularization, and so that's where we wound up with this very wiggly curve with Lambda $\lambda$ was small or it was even zero.

And in that case, we have a high variance model, and so $J_{train}$ is going to be small and $J_{cv}$ is going to be large because it does great on the training data but does much worse on the cross validation data. 

### Case: $\lambda=10 000$, a large value

![alt text](./img/image8.png)

This other extreme on the right with very large values of Lambda $\lambda$, lets say Lambda $\lambda = 10000$ ends up with fitting a model that looks like the right bottom graph with a horizontal blue line, $f(x) \approx b$). This has high bias, it underfits the data, and it turns out $J_{train}$ will be high and $J_{cv}$ will be high as well. 

## How $J_{train}$ varies as a function of Lambda $\lambda$

Spoiler: As $\lambda$ increases, $J_{train}$ increases as well

![alt text](./img/image9.png)

In fact, if you were to look at how $J_{train}$ varies as a function of Lambda $\lambda$, you find that $J_{train}$ will go up like the image shown above because in the optimization cost function, the larger Lambda $\lambda$ is the more the algorithm is trying to keep $w_{j}^2$ small that is, the more weight is given to this regularization term, and thus the less attention is paying to actually doing well on the training set. This term on the left is $J_{train}$, so the most trying to keep the parameters small, the less good a job it does on minimizing the training error. 

So, that's why as Lambda $\lambda$ increases, the training error $J_{train}$ will tend to increase like so. 

## How $J_{cv}$ varies as a function of Lambda $\lambda$

Spoiler: There is some intermediate value for $\lambda$ which makes $J_{cv}$ the lowest value you can obtain

![alt text](./img/image10.png)

Now, how about the cross-validation error? It turns out the cross-validation error will look like the image above

![alt text](./img/image11.png)

Because we've seen that if Lambda $\lambda$ is too small or too large, then it doesn't do well on the cross-validation set. It either overfits here on the left or underfits here on the right. There'll be some intermediate value of Lambda $\lambda$ that causes the algorithm to perform best. 

And what cross-validation is doing is, it's trying out a lot of different values of Lambda $\lambda$. This is what we saw on the last slide: try Lambda $\lambda = 0$, Lambda $\lambda = 0.01$, $\lambda = 0.02$. Try a lot of different values of Lambda $\lambda$ and evaluate the cross-validation error in a lot of these different points, and then hopefully pick a value that has low cross validation error, and this will hopefully correspond to a good model for your application. 

## Comparing diagrams

![alt text](./img/image12.png)

We are comparing the diagram discussed from the last lecture vs this new one:

* Previous lecture: $J_{train}$ and $J_{cv}$ vs $d$ (the degree of polynomial)

* Current lecture: $J_{train}$ and $J_{cv}$ vs $\lambda$ (the regularization parameter)

If you compare this diagram to the one that we had in the previous video, where the horizontal axis was the degree of polynomial $d$, these two diagrams look a little bit not mathematically and not in any formal way, but they look a little bit like mirror images of each other, and that's because when you're fitting a degree of polynomial, the left part of this curve corresponded to underfitting and high bias, the right part corresponded to overfitting and high variance whereas in this one, high-variance was on the left and high bias was on the right. 

But that's why these two images are a little bit like mirror images of each other. But in both cases, cross-validation, evaluating different values can help you choose a good value of $d$ or a good value of Lambda $\lambda$. 

That's how the choice of regularization parameter Lambda $\lambda$ affects the bias and variance and overall performance of your algorithm, and you've also seen how you can use cross-validation to make a good choice for the regularization parameter Lambda $\lambda$. 

### Summary

Now, so far, we've talked about how having a high training set error, high $J_{train}$ is indicative of high bias and how having a high cross-validation error of $J_{cv}$, specifically if it's much higher than $J_{train}$, how that's indicative of variance problem. 

But what does these words "high" or "much higher" actually mean? Let's take a look at that in the next video where we'll look at how you can look at the numbers $J_{train}$ and $J_{cv}$ and judge if it's high or low, and it turns out that one further refinement of these ideas, that is, establishing a baseline level of performance we're learning algorithm will make it much easier for you to look at these numbers, $J_{train}$, $J_{cv}$, and judge if they are high or low. Let's take a look at what all this means in the next video.

## Establishing a baseline level of performance

