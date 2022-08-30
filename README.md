# Multi-Model-Product-Recommendation-System
# <center> BDA Final Project

## Product Recommendation System for e-commerce businesses
A well-developed recommendation system will help businesses improve their shopper's website experience and customer acquisition and retention.

The recommendation system, We have designed below is based on the journey of a new customer from the time he/she lands on the business’s website for the first time to when he/she makes repeat purchases.


* **Recommendation system part:** Product pupularity based system targetted at new customers

* **Recommendation system part:** Model-based collaborative filtering system based on customer's purchase history and ratings provided by other users who bought items similar items

* **Recommendation system part:** When a business is setting up its e-commerce website for the first time withou any product rating

When a new customer without any previous purchase history visits the e-commerce website for the first time, he/she is recommended the most popular products sold on the company's website. Once, he/she makes a purchase, the recommendation system updates and recommends other products based on the purchase history and ratings provided by other users on the website. The latter part is done using collaborative filtering techniques.

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Importing-Libraries" data-toc-modified-id="Importing-Libraries-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Importing Libraries</a></span></li><li><span><a href="#Loading-Datasets" data-toc-modified-id="Loading-Datasets-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Loading Datasets</a></span></li><li><span><a href="#Exploratory-Data-Analysis" data-toc-modified-id="Exploratory-Data-Analysis-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Exploratory Data Analysis</a></span></li><li><span><a href="#Recommender-Systems" data-toc-modified-id="Recommender-Systems-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Recommender Systems</a></span><ul class="toc-item"><li><span><a href="#Popularity-Based-Recommender" data-toc-modified-id="Popularity-Based-Recommender-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Popularity-Based Recommender</a></span></li><li><span><a href="#Collaborative-Recommender" data-toc-modified-id="Collaborative-Recommender-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Collaborative Recommender</a></span><ul class="toc-item"><li><span><a href="#SVD:-Matrix-Factorization-Based-Algorithm" data-toc-modified-id="SVD:-Matrix-Factorization-Based-Algorithm-5.2.1"><span class="toc-item-num">5.2.1&nbsp;&nbsp;</span>SVD: Matrix Factorization Based Algorithm</a></span></li></ul></li><li><span><a href="#Hybrid-Recommender" data-toc-modified-id="Hybrid-Recommender-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Item to item based recommendation system based on product description</a></span></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></div>
## Data Exploration

![33](https://user-images.githubusercontent.com/51040406/187420755-af9e158a-03fc-42a4-95f2-91610f325fd2.png)

![333](https://user-images.githubusercontent.com/51040406/187421087-69ae718a-0f1b-4d5d-90a0-3a282e5079e2.png)

![22](https://user-images.githubusercontent.com/51040406/187420769-054b723a-1f21-41b1-9399-5a03e6aa689c.png)

![333](https://user-images.githubusercontent.com/51040406/187421112-df5e854a-be6b-4f82-b459-601045aa47fd.png)

## Recommendation System - Part I
### Product popularity based recommendation system targeted at new customers
* Popularity based are a great strategy to target the new customers with the most popular products sold on a business's website and is very useful to cold start a recommendation engine.
[[](http://)](http://)
* **Dataset : **[Amazon product review dataset](https://www.kaggle.com/skillsmuggler/amazon-ratings)

![Screenshot 2022-08-30 040325](https://user-images.githubusercontent.com/51040406/187419585-ad606fae-1b04-4183-a6d3-12d6ae11f49d.png)


** Analysis:**

* The above graph gives us the most popular products (arranged in descending order) sold by the business.

* For eaxmple, product, ID # B001MA0QY2 has sales of over 7000, the next most popular product, ID # B0009V1YR8 has sales of  3000, etc.     


### Model 1: Popularity based recommender model

The implementation of Popularity-Based Filtering is straighforward. All we have to do is sort our products based on ratings, and display the top products of our list. Therefore, we should;

- Create a metric to score or rate the products.
- Calculate the score for every product.
- Sort the scores and recommend the best rated product to the users.

We can use the average ratings of the products as the score but using this will not be fair enough since a product with 5average rating and only43 votes cannot be considered better than the product with 4 as average rating but 40 votes. So, we use IMDB's weighted rating formula to score the products, as follows:

Weighted Rating (WR) = $(\frac{v}{v + m} . R) + (\frac{m}{v + m} . C)$ 

- v: the number of votes for the product

- m: the minimum votes required to be listed in the chart

- R: the average rating of the product

- C: the mean vote across the whole report

df_rating['Mean Rating'].mean()

The mean rating for all the products (C) is 4.1 on a scale of 5.

The next step is to determine an appropriate value for m, the minimum number of votes required for a product to be listed in the chart. We use 90th percentile as our cutoff. In other words, for a product to feature in the charts, the number of its votes should be higher than that of 90% of the products in the list.


So the top 10 popular products that this model will recommend to users include:

'B00GJX58PE', 'B00K7ER6LU', 'B00I46E8DC', 'B00F008GFQ', 'B002YFN49I',
 'B00FPROWWU', 'B00IBS9QC6', 'B009OWSHQE', 'B004CNRDBU', 'B008DWSJ1O'.

We should keep in mind that this popularity-based recommender provides a general chart of recommended products to all the users, regardless of the user's personal taste. It is not sensitive to the interests and tastes of a particular user, and it does not give personalized recommendations based on the users.

## Recommendation System - Part II
### Model-based collaborative filtering system

* Recommend items to users based on purchase history and similarity of ratings provided by other users who bought items to that of a particular customer.
* A model based collaborative filtering technique is closen here as it helps in making predicting products for a particular user by identifying patterns based on preferences from multiple user data.

* Surprise library is a Python scikit for building and analyzing recommender systems that deal with explicit rating data. Here we also use the Surprise library that uses extremely powerful algorithms like Singular Value Decomposition (SVD) to minimise Root Mean Square Error (RMSE) that is measured by Kfold Cross Validation and give great recommendations.

#### Utility Matrix based on products sold and user reviews
**Utility Matrix : **An utlity matrix is consists of all possible user-item preferences (ratings) details represented as a matrix. The utility matrix is sparce as none of the users would buy all teh items in the list, hence, most of the values are unknown.

Since data is too huge, due to that colab(free version) wasn't able to process it properly. So inorder to work properly a subset is taken which is top users who had given more than 50 ratings.


## Recommendation System - Part III
* For a business without any user-item purchase history, a search engine based recommendation system can be designed for users. The product recommendations can be based on textual clustering analysis given in product description.
* **Dataset : **[Home Depot's dataset with product dataset.](https://www.kaggle.com/c/home-depot-product-search-relevance/data)

### Item to item based recommendation system based on product description

Applicable when business is setting up its E-commerce website for the first time

Once a cluster is identified based on the user's search words, the recommendation system can display items from the corresponding product clusters based on the product descriptions.

## Evaluation of all models

#### Popularity based Model

The RMSE value for Recommender model is 1.3078273121694615


#### KNN with means
RMSE: 1.0561
1.0561184327784499


#### SVD
RMSE: 0.8718
0.8717742852178509                         

SVD has much better results over others

#### Summary: 

This works best if a business is setting up its e-commerce website for the first time and does not have user-item purchase/rating history to start with initially. This recommendation system will help the users get a good recommendation to start with and once the buyers have a purchased history, the recommendation engine can use the model-based collaborative filtering technique.


