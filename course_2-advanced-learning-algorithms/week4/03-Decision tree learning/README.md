## Measuring purity

![alt text](./img/image1.png)

In this video, we'll look at the way of measuring the purity of a set of examples. 

If the examples are all cats of a single class then that's very pure, if it's all not cats that's also very pure, but if it's somewhere in between how do you quantify how pure is the set of examples? 

### Entropy definition

Let's take a look at the definition of entropy which is a measure of the impurity of a set of data. 

Given a set of six examples like this, we have three cats and three dogs, let's define $p_1$ to be the fraction of examples that are cats, 

$$ p_1 \text{ = fraction of examples that are cats}$$

that is, the fraction of examples with label one, that's what the subscript one indicates. And so, in this example $p_1=\frac{3}{6}=0.5$

### Entropy function *H(p)*

We're going to measure the impurity of a set of examples using a function called *The entropy function* which looks like this. The entropy function is conventionally denoted as capital H of this number $p_1$, this is $H(p_1)$ and the function looks like this curve over here where the horizontal axis is $p_1$, the fraction of cats in the sample, and the vertical axis is the value of the entropy.

![alt text](./img/image3.png)

#### Example 1
In this example, where $p_1=\frac{3}{6}=0.5$, the value of the entropy of $p_1 = 1$. 

So, you notice that this curve is highest when your set of examples is 50-50, so it's most impure as an impurity of one or with an entropy of one when your set of examples is 50-50, whereas in contrast if your set of examples was either all cats or all not cats then the entropy is zero. 

$\text{If } p_1 = 0.5, \text{then } H(p_1) = 1$

$\text{If } p_1 = 0, \text{then } H(p_1) = 0$

$\text{If } p_1 = 1, \text{then } H(p_1) = 0$

#### Example 2

Let's just go through a few more examples to gain further intuition about entropy and how it works. Here's a different set of examples with five cats and one dog, so $p_1$, the fraction of positive examples, a fraction of examples labeled one is 5/6 and so $p_1$ is about 0.83 and if you read off that value at about 0.83 we find that the entropy of $p_1$ is about 0.65. And here I'm writing it only to two significant digits. 

#### Example 3

Here's one more example. This sample of six images has all cats so $p_1$ is six out of six because all six are cats and the entropy of $p_1$ is this point over here which is zero. 

We see that as you go from 3 out of 6 cats, $p_1 = \frac{3}{6} = 0.5$, to 6 out of 6 cats, $p_1 = \frac{6}{6} = 1$, the impurity decreases from one to zero or in other words, the purity increases as you go from a 50-50 mix of cats and dogs to all cats. 

#### Example 4

Let's look at a few more examples. Here's another sample with two cats and four dogs, so $p_1$ here is 2/6 which is 1/3, and if you read off the entropy at 0.33 it turns out to be about 0.92. This is actually quite impure and in particular this set is more impure than this set because it's closer to a 50-50 mix, which is why the impurity here is 0.92 as opposed to 0.65. 

#### Example 5

Finally, one last example, if we have a set of all six dogs then $p_1$ is equal to 0 and the entropy of $p_1$ is just this number down here which is equal to 0 so there's zero impurity or this would be a completely pure set of all not cats or all dogs. 

### Equation for the entropy function H($p_1$)

Now, let's look at the actual equation for the entropy function H($p_1$). 

![alt text](./img/image2.png)

Recall that 

$$p_1 \text{ = is the fraction of examples that are equal to cats}$$ 

so if you have a sample that is 2/3 cats then that sample must have 1/3 not cats. 

So, let me define $p_0$: 

$$p_0 \text{ = is the fraction of examples that are NOT cats}$$ 

$$p_0 = 1 - p_1$$ 

The entropy function $H(p_1)$ is then defined as:

$$ H(p_1) = -p_1log_2(p_1) - p_0log_2(p_0) $$

> [!NOTE] 
> By convention when computing entropy we take logs to base *2* rather than to base *e*

Alternatively, this is also equal to: 

$$ H(p_1) = -p_1log_2(p_1) - (1-p_1)log_2(1-p_1) $$

If you were to plot this function in a computer you will find that it will be exactly this function on the left. 

> [!NOTE] 
>
> We take $log_2$ just to make the peak of this curve equal to one. If we were to take $log_e$ or the base of natural logarithms, then that just vertically scales this function and it will still work but the numbers become a bit hard to interpret because the peak of the function isn't a nice round number like one anymore. 

> [!NOTE] 
> One note on computing this entropy function H: 
> 
> If $p_1 = 0$ or $p_0 = 0$, then an expression like this will look like $0 * log(0)$ and $log(0)$ is technically undefined, it's actually negative infinity. 
> But, by convention, for the purposes of computing entropy, we'll take $0 * log(0) = 0$ and that will correctly compute the entropy $H(0) = 0$ or $H(1) = 0$

If you're thinking that this definition of entropy looks a little bit like the definition of the logistic loss that we learned about in the last course, there is actually a mathematical rationale for why these two formulas look so similar. 

But you don't have to worry about it and we won't get into it in this class. But applying this formula for entropy should work just fine when you're building a decision tree. 

## Summary 
 
To summarize, the entropy function is a measure of the impurity of a set of data. It starts from zero, it goes up to one, and then comes back down to zero as a function of the fraction of positive examples in your sample. 

There are other functions that look like this, they go from zero up to one and then back down. For example, if you look in open source packages you may also hear about something called the **Gini criteria**, which is another function that looks a lot like the entropy function, and that will work well as well for building decision trees but for the sake of simplicity, in these videos I'm going to focus on using the **entropy criteria** which will usually work just fine for most applications. 

Now that we have this definition of entropy, in the next video let's take a look at how you can actually use it to make decisions as to what feature to split on in the nodes of a decision tree.

## Choosing a split: Information Gain

![alt text](./img/image4.png)

When building a decision tree, the way we'll decide what feature to split on at a node will be based on what choice of feature reduces entropy the most (Reduces entropy or reduces impurity, or maximizes purity) 

In decision tree learning, the reduction of entropy is called *Information gain*

Let's take a look, in this video, at how to compute information gain and therefore choose what features to use to split on at each node in a decision tree. 

## How to compute Information Gain (Ig)

Let's use the example of deciding what feature to use at the root node of the decision tree we were building just now for recognizing cats vs not cats

### 1. Splitting on the ear shape feature

If we had split using the ear shape feature at the root node, this is what we would have gotten: Five examples on the left and five on the right and on the left, we would have four out of five cats, so $p_1 = \frac{4}{5} = 0.8$ and on the right, one out of five are cats, so $p_1 = \frac{1}{5} = 0.2$

If you apply the entropy formula 

$$ H(p_1) = -p_1log_2(p_1) - (1-p_1)log_2(1-p_1) $$

from the last video to both the left subset of data and right subset of data, we find that the degree of impurity on the left is entropy of 0.8, which is about 0.72, and on the right, the entropy of 0.2 turns out also to be 0.72. 

* For the left sub branch:

$$ H(p_1) = H(0.8) = 0.72 $$

* For the right sub branch:

$$ H(p_1) = H(0.2) = 0.72 $$

This would be the entropy at the left and right sub branches if we were to split on the ear shape feature. 

### 2. Splitting on the face shape feature

One other option would be to split on the face shape feature. 

If we'd done so then on the left, four of the seven examples would be cats, so $p_1 = \frac{4}{7} = 0.57$ and on the right, 1/3 are cats, so $p_1 = \frac{1}{3} = 0.33$

The entropy of 4/7 and the entropy of 1/3 are 0.99 and 0.92. 

* For the left sub branch:

$$ H(p_1) = H(4/7) = 0.99 $$

* For the right sub branch:

$$ H(p_1) = H(1/3) = 0.92 $$

So, the degree of impurity in the left and right nodes seems much higher, 0.99 and 0.92 compared to previously computed values of 0.72 and 0.72 when splitting on face shape

### 3. Splitting on the whiskers feature

Finally, the third possible choice of feature to use at the root node would be the whiskers feature in which case you split based on whether whiskers are present or absent. 

In this case, $p_1 = \frac{3}{4}$ on the left, $p_1 = \frac{2}{6}$ on the right, and the entropy values are as follows. 

* For the left sub branch:

$$ H(p_1) = H(3/4) = 0.81 $$

* For the right sub branch:

$$ H(p_1) = H(1/3) = 0.92 $$

## Which option works better?

So, the key question we need to answer is, given these three options of a feature to use at the root node, which one do we think works best? 

It turns out that rather than looking at these entropy numbers and comparing them, it would be useful to take a weighted average of them, and here's what I mean. 

If there's a node with a lot of examples in it with high entropy that seems worse than if there was a node with just a few examples in it with high entropy because entropy, as a measure of impurity, is worse if you have a very large and impure dataset compared to just a few examples and a branch of the tree that is very impure. 

So, the key decision is, of these three possible choices of features to use at the root node, which one do we want to use? 

Associated with each of these splits is two numbers:

1. The entropy on the left sub-branch and 
2. the entropy on the right sub-branch. 

In order to pick from these, we like to actually combine these two numbers into a single number so you can just pick of these three choices, which one does best.

## Weighted average entropy

The way we're going to combine these two numbers is by taking a weighted average because how important it is to have low entropy in, say, the left or right sub-branch also depends on how many examples went into the left or right sub-branch because if there are lots of examples in, say, the left sub-branch then it seems more important to make sure that that left sub-branch's entropy value is low. 

### 1. Ear shape calculus

In this example, we have five of the 10 examples went to the left sub-branch, so we can compute the weighted average as 5/10 times the entropy of 0.8, and then add to that 5/10 examples also went to the right sub-branch plus 5/10 times the entropy of 0.2. 

### 2. Face shape calculus

Now, for this example in the middle, the left sub-branch had received seven out of 10 examples and so we're going to compute 7/10 times the entropy of 0.57 plus the right sub-branch had three out of 10 examples, so plus 3/10 times entropy of 0.33 of 1/3. 

### 3. Whiskers calculus

Finally, on the right, we'll compute 4/10 times entropy of 0.75 plus 6/10 times entropy of 0.33. 

The way we will choose a split is by computing these three numbers and picking whichever one is lowest because that gives us the left and right sub-branches with the lowest average weighted entropy. 

## Computing the reduction in entropy

In the way that decision trees are built, we're actually going to make one more change to these formulas to stick to the convention in decision tree building but it won't actually change the outcome which is rather than computing this weighted average entropy, we're going to compute the reduction in entropy compared to if we hadn't split at all. 

So, if we go to the root node, remember that the root node we have started off with all 10 examples in the root node with five cats and five dogs and so at the root node, we had $p_1 = \frac{5}{10} = 0.5$ and the entropy of the root nodes is the entropy of 0.5 was actually equal to 1, $H(p_1) = H(0.5) = 1$. 

This was maximum impurity because it was five cats and five dogs. 

The formula that we're actually going to use for choosing a split is not this weighted entropy at the left and right sub-branches. Instead is going to be the entropy at the root node, which is entropy of 0.5, then minus this formula. 

In this example, if you work out the math, it turns out to be 0.28 for the face shape example, we can compute entropy of the root node, entropy of 0.5 minus this, which turns out to be 0.03, and for whiskers, compute that, which turns out to be 0.12. 

These numbers that we just calculated, 0.28, 0.03, and 0.12, these are called the **Information gain** and what it measures is the reduction in entropy that you get in your tree resulting from making a split because the entropy was originally one at the root node and by making the split, you end up with a lower value of entropy and the difference between those two values is a reduction in entropy and that's 0.28 in the case of splitting on the ear shape. 

## Reduction in entropy vs entropy at each sub branch

Why do we bother to compute reduction in entropy rather than just entropy at the left and right sub-branches? It turns out that one of the stopping criteria for deciding when to not bother to split any further is if the reduction in entropy is too small. In which case you could decide, you're just increasing the size of the tree unnecessarily and risking overfitting by splitting and just decide to not bother if the reduction in entropy is too small or below a threshold. 

In this other example, spitting on ear shape results in the biggest reduction in entropy: 0.28 is bigger than 0.03 or 0.12 and so we would choose to split onto ear shape feature at the root node. 

On the next slide, let's give a more formal definition of Information gain. 

By the way, one additional piece of notation that we'll introduce also in the next slide is these numbers, 5/10 and 5/10 I'm going to call them $w^{left}$ because that's the fraction of examples that went to the left branch, and I'm going to call this $w^{right}$ because that's the fraction of examples that went to the right branch whereas for this another example, $w^{left}$ would be 7/10, and $w^{right}$ will be 3/10

## General formula for how to compute information gain

![alt text](./img/image5.png)

Let's now write down the general formula for how to compute information gain.

## Left sub branch

Using the example of splitting on the ear shape feature, let me define $p_1^{left}$ to be equal to the fraction of examples in the left subtree that have a positive label, that are cats. 

In this example, $p_1^{left}$ will be equal to 4/5. 

Also, let me define $w^{left}$ to be the fraction of examples of all of the examples of the root node that went to the left sub-branch, and so in this example, $w^{left}$ would be 5/10. 

## Right sub branch

Similarly, let's define $p_1^{right}$ to be of all the examples in the right branch, the fraction that are positive examples and so one of the five of these examples being cats, there'll be 1/5, and similarly, $w^{right}$ is 5/10 the fraction of examples that went to the right sub-branch. 

## Root node

Let's also define $p_1^{root}$ to be the fraction of examples that are positive in the root node. In this case, this would be 5/10 or 0.5. 

## Information gain definition

Information gain is then defined as the entropy of $p_1^{root}$, so what's the entropy at the root node, minus that weighted entropy calculation that we had on the previous slide: This is, minus $w^{left}$, those were 5/10 in the example, times the entropy applied to $p_1^{left}$, that's entropy on the left sub-branch, plus $w^{right}$ the fraction of examples that went to the right branch, times entropy of $p_1^{right}$. 

$$ \text{Information gain =} H(p_1^{root}) - [w^{left} * H(p_1^{left}) + w^{right} * H(p_1^{right})] $$

With this definition of entropy, you can calculate the information gain associated with choosing any particular feature to split on in the node. 

> [!IMPORTANT] 
> Then, out of all the possible features you could choose to split on, you **choose the one that gives you the highest information gain**. The goal is for this split to **increase the purity of the resulting subsets on both the left and right branches of the decision tree**

## Summary

Now that you know how to calculate information gain or reduction in entropy, you know how to pick a feature to split on another node. Let's put all the things we've talked about together into the overall algorithm for building a decision tree given a training set. Let's go see that in the next video.

## Putting it together


