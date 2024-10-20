## Making recommendations

Welcome to this second to last week of the machine learning specialization. I'm really happy that together, almost all the way to the finish line. What we'll do this week is discuss recommender systems. 

This is one of the topics that has received quite a bit of attention in academia but the commercial impact and the actual number of practical use cases of recommended systems seems to me to be even vastly greater than the amount of attention it has received in academia. 

Every time you go to an online shopping website like Amazon or a movie streaming sites like Netflix or go to one of the apps or sites that do food delivery, many of these sites will recommend things to you that they think you may want to buy or movies they think you may want to watch or restaurants that they think you may want to try out. And for many companies, a large fraction of sales is driven by their recommended systems. 

So, today for many companies, the economics or the value driven by recommended systems is very large and so what we're doing this week is take a look at how they work. So, with that let's dive in and take a look at what is a recommender system

![alt text](./img/image1.png)

I'm going to use as a running example, the application of predicting movie ratings. So, say you run a large movie streaming website and your users have rated movies using one to five stars and so in a typical recommended system you have a set of users, here we have four users Alice, Bob Carol and Dave which have numbered users 1,2,3,4 as well as a set of movies *Love at last*, *Romance forever*, *Cute puppies of love* and then *Nonstop car chases* and *Sword versus karate* and what the users have done is rated these movies one to five stars or in fact to make some of these examples a little bit easier I'm not going to let them rate the movies from zero to five stars.

![alt text](./img/image2.png)

### Alice rates

So say Alice has rated Love and last five stars, Romance forever five stars. Maybe she has not yet watched cute puppies of love so you don't have a rating for that. And I'm going to denote that via a question mark and she thinks nonstop car chases and sword versus karate deserve zero stars

### Bob rates

Bob race at five stars has not watched that, so you don't have a rating race at four stars, 0,0. 

### Carol rates

Carol on the other hand, thinks that deserve zero stars has not watched that zero stars and she loves nonstop car chases and swords versus karate

### Dave rates

Dave rates the movies as follows. 

## Notation

In the typical recommended system, you have some number of users as well as some number of items. In this case, the items are movies that you want to recommend to the users and even though I'm using movies in this example, the same logic or the same thing works for recommending anything from products or websites to my self, to restaurants, to even which media articles, the social media articles to show to the user that may be more interesting for them. 

The notation I'm going to use is I'm going to use $n_u$ to denote the number of users. So, in this example $n_u = 4$ because you have four users and $n_m$ to denote the number of movies or really the number of items, so in this example $n_m = 5$ because we have five movies. 

I'm going to set $r(i,j)=1$, if user $j$ has rated movie $i$. 

So, for example, user 1, that is Alice, has rated movie one but has not rated movie three and so $r(1,1)=1$, because she has rated movie one, but $r(3,1)=0$ because she has not rated movie number three. 

Then, finally I'm going to use $y^{(i,j)}$ to denote the rating given by user $j$ to movie $i$. So, for example, this rating here from user Bob would be that movie three ($i=3$) was rated by user 2 ($j=2$) to be equal to four, this is $y^{(3,2)} = 4$

> [!WARNING]
> Notice that not every user rates every movie and it's important for the system to know which users have rated which movies. That's why we're going to define $r(i,j)=1$ if user $j$ has rated movie $i$ and $r(i,j)=0$ if user $j$ has NOT rated movie $i$

So, with this framework for recommended systems, one possible way to approach the problem is to look at the movies that users have not rated and to try to predict how users would rate those movies because then we can try to recommend to users things that they are more likely to rate as five stars.

And in the next video we'll start to develop an algorithm for doing exactly that but making one very special assumption which is we're going to assume temporarily that we have access to features or extra information about the movies such as which movies are romance movies, which movies are action movies and using that will start to develop an algorithm but later this week will actually come back and ask what if we don't have these features, how can you still get the algorithm to work then? 

But let's go on to the next video to start building up this algorithm.

## Using per-item features

## Collaborative filtering algorithm

## Binary labels: favs, likes and clicks