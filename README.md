# Multi-Model-Product-Recommendation-System
# <center> BDA Final Project

## Product Recommendation System for e-commerce businesses
A well-developed recommendation system will help businesses improve their shopper's website experience and customer acquisition and retention.

The recommendation system, We have designed below is based on the journey of a new customer from the time he/she lands on the business’s website for the first time to when he/she makes repeat purchases.

The recommendation system is designed in 3 parts

* **Recommendation system part I:** Product pupularity based system targetted at new customers

* **Recommendation system part II:** Model-based collaborative filtering system based on customer's purchase history and ratings provided by other users who bought items similar items

* **Recommendation system part III:** When a business is setting up its e-commerce website for the first time withou any product rating

When a new customer without any previous purchase history visits the e-commerce website for the first time, he/she is recommended the most popular products sold on the company's website. Once, he/she makes a purchase, the recommendation system updates and recommends other products based on the purchase history and ratings provided by other users on the website. The latter part is done using collaborative filtering techniques.

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Importing-Libraries" data-toc-modified-id="Importing-Libraries-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Importing Libraries</a></span></li><li><span><a href="#Loading-Datasets" data-toc-modified-id="Loading-Datasets-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Loading Datasets</a></span></li><li><span><a href="#Exploratory-Data-Analysis" data-toc-modified-id="Exploratory-Data-Analysis-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Exploratory Data Analysis</a></span></li><li><span><a href="#Recommender-Systems" data-toc-modified-id="Recommender-Systems-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Recommender Systems</a></span><ul class="toc-item"><li><span><a href="#Popularity-Based-Recommender" data-toc-modified-id="Popularity-Based-Recommender-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Popularity-Based Recommender</a></span></li><li><span><a href="#Collaborative-Recommender" data-toc-modified-id="Collaborative-Recommender-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Collaborative Recommender</a></span><ul class="toc-item"><li><span><a href="#SVD:-Matrix-Factorization-Based-Algorithm" data-toc-modified-id="SVD:-Matrix-Factorization-Based-Algorithm-5.2.1"><span class="toc-item-num">5.2.1&nbsp;&nbsp;</span>SVD: Matrix Factorization Based Algorithm</a></span></li></ul></li><li><span><a href="#Hybrid-Recommender" data-toc-modified-id="Hybrid-Recommender-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Item to item based recommendation system based on product description</a></span></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></div>

## Recommendation System - Part I
### Product popularity based recommendation system targeted at new customers
* Popularity based are a great strategy to target the new customers with the most popular products sold on a business's website and is very useful to cold start a recommendation engine.
[[](http://)](http://)
* **Dataset : **[Amazon product review dataset](https://www.kaggle.com/skillsmuggler/amazon-ratings)

#### Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
plt.style.use("ggplot")

import sklearn
from sklearn.decomposition import TruncatedSVD

#### Loading the dataset

amazon_ratings = pd.read_csv('ratings_Beauty.csv')
amazon_ratings = amazon_ratings.dropna()
amazon_ratings.head()


missing_data = pd.DataFrame(amazon_ratings.isnull().mean()*100)
missing_data

All the columns are clean

amazon_ratings.shape

### Exploratory Data Analysis

# IQR
Q1 = np.percentile(amazon_ratings['Rating'], 25,
                   interpolation = 'midpoint')
 
Q3 = np.percentile(amazon_ratings['Rating'], 75,
                   interpolation = 'midpoint')
IQR = Q3 - Q1


# Above Upper bound
upper = amazon_ratings['Rating'] >= (Q3+1.5*IQR)
 
print("Upper bound:",upper)
print(np.where(upper))
 
# Below Lower bound
lower = amazon_ratings['Rating'] <= (Q1-1.5*IQR)
print("Lower bound:", lower)
print(np.where(lower))

amazon_ratings['Rating'][3]


plt.figure(figsize=(10,6))
sns.countplot(x='Rating', data=amazon_ratings, palette='winter')
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Number of Each Rating', fontsize=15)
plt.show()

<center>We can see that most of users rated 5 for the products.

df_rating=pd.DataFrame({'Number of Rating':amazon_ratings.groupby('ProductId').count()['Rating'], 'Mean Rating':amazon_ratings.groupby('ProductId').mean()['Rating']})

df_rating.head()

plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
plt.hist(x='Number of Rating',data=df_rating,bins=30,color='teal')
plt.title('Distribution of Number of Rating', fontsize=15)
plt.xlabel('Number of Rating', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.subplot(1,2,2)
plt.hist(x='Mean Rating',data=df_rating,bins=30, color='slateblue')
plt.title('Distribution of Mean Rating', fontsize=15)
plt.xlabel('Mean Rating', fontsize=12)
plt.yticks([])
plt.show()

From these histograms we can see that most of the number of ratings are between 0 and 1825, and most of the products have a mean rating of 5.

We create a scatter plot to observe the relationship between Number of Rating and Mean Rating.

plt.figure(figsize=(8,6))
sns.jointplot(x='Number of Rating', y='Mean Rating',data=df_rating,color='b', height=7)
plt.suptitle('Mean Rating Versus Number of Rating', fontsize=15, y=0.92)

plt.show()

### Top products based on sales

#Top 10 Products based on sales.
popular_products = pd.DataFrame(amazon_ratings.groupby('ProductId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
most_popular.head(10)

most_popular.head(30).plot(kind = "bar", color='g')

** Analysis:**

* The above graph gives us the most popular products (arranged in descending order) sold by the business.

* For eaxmple, product, ID # B001MA0QY2 has sales of over 7000, the next most popular product, ID # B0009V1YR8 has sales of  3000, etc.     

### Number of Unique users

print('Number of unique users', len(amazon_ratings['UserId'].unique()))

### Number of Products with good ratings

max_ratings1 = amazon_ratings[amazon_ratings['Rating'] >= 4.0]
print('Number of unique products rated high',len(max_ratings1['ProductId'].unique()))

### Model Preparation

#Now we will drop timestamp column as it isn't much of help.

amazon_ratings.drop('Timestamp',axis=1,inplace=True)

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(amazon_ratings, test_size = 0.2, random_state=0)

train_data.head()


test_data.head()

train_data_grouped = train_data.groupby('ProductId').mean().reset_index()


train_data_grouped.head()

train_data.groupby('ProductId')['Rating'].count().sort_values(ascending=False).head(10)

train_data_sort = train_data_grouped.sort_values(['Rating', 'ProductId'], ascending=False)


train_data_sort.head()

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

df_rating['Number of Rating'].quantile(q=0.9)

Now, we filter the products that qualify for the chart and put them in a new dataframe called df_filtered.

Now, we calculate score for each qualified product. To do this, we define a function, weighted_rating(), and apply this function to the DataFrame of qualified products.

def product_score(x):
    v=x['Number of Rating']
    m=df_rating['Number of Rating'].quantile(q=0.9)
    R=x['Mean Rating']
    C=df_rating['Mean Rating'].mean()
    return ((R*v)/(v+m))+((C*m)/(v+m))

df_filtered['score']=df_filtered.apply(product_score, axis=1)

df_filtered.head()


Finally, we sort the dataframe based on the score feature, and we output the the top 10 popular products.

df_highscore=df_filtered.sort_values(by='score', ascending=False).head(10)

df_filtered=df_rating[df_rating['Number of Rating']>df_rating['Number of Rating'].quantile(q=0.9)]

df_filtered.shape

We see that there are 23863 products which qualify to be in this list.

df_highscore

df_highscore.index


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

import surprise
from surprise import KNNWithMeans
from surprise.model_selection import GridSearchCV
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
%matplotlib inline
from surprise import SVD, Reader, Dataset 
from surprise.model_selection import cross_validate

from surprise import KNNBasic, SVD, NormalPredictor, KNNBaseline,KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering, Reader, dataset, accuracy

Since data is too huge, due to that colab(free version) wasn't able to process it properly. So inorder to work properly a subset is taken which is top users who had given more than 50 ratings.


userID = amazon_ratings.groupby('UserId').count()

top_user = userID[userID['Rating'] >= 50].index

topuser_ratings_df = amazon_ratings[amazon_ratings['UserId'].isin(top_user)]

topuser_ratings_df.sort_values(by='Rating', ascending=False).head()

prodID = amazon_ratings.groupby('ProductId').count()

top_prod = prodID[prodID['Rating'] >= 50].index

top_ratings_df = topuser_ratings_df[topuser_ratings_df['ProductId'].isin(top_prod)]

top_ratings_df.sort_values(by='Rating', ascending=False).head()


top_ratings_df.shape

Here we can see the top rated products are 13717

#### Conversion to surprise format


reader = Reader(rating_scale=(0.5, 5.0))

Dataset loading

data = Dataset.load_from_df(top_ratings_df[['UserId', 'ProductId', 'Rating']],reader)

#### Model training

from surprise.model_selection import train_test_split
trainset, testset = train_test_split(data, test_size=.3,random_state=0)


type(trainset)

#### KNN with means

model = KNNWithMeans(k=10, min_k=6, sim_options={'name': 'pearson_baseline', 'user_based': True})
model.fit(trainset)

### SVD

svd = SVD()

reader = Reader()

cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

We get a mean Root Mean Sqaure Error of 1.0 approx which is good enough for our case. Let us now train on our dataset and arrive at predictions.

trainset = data.build_full_trainset()

svd.fit(trainset)

Let us pick the user with userId of 'AKM1MP6P0OYPR' and check the ratings she/he has given so far to different products.

amazon_ratings[amazon_ratings['UserId'] == 'A1APERZNMEU8PW']

As an example, we use the algorithm to predict the score that might be given to the productId of '0970407998' by this specific userId.

svd.predict(uid='A17HMM1M7T9PJ1', iid='0970407998', r_ui=None)

svd.predict(uid='A17HMM1M7T9PJ1', iid='0970407998', r_ui=None).est

Our model predicts that userId of 'A17HMM1M7T9PJ1' will give 4.12 as the rating for productId of '0970407998'.

#### Hyper Parameter Tuning of SVD model

from surprise.model_selection import GridSearchCV
param_grid = {'n_factors' : [5,10,15], "reg_all":[0.01,0.02]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3,refit = True)

gs.fit(data)

gs.best_params

# Use the "best model" for prediction
gs.test(testset)
accuracy.rmse(gs.test(testset))

The RMSE value of collaborative model, at first by KNNwithmeans is 1.05 and with SVD is 1.01. On further tuning of SVD model we got <b>RMSE of SVD model as 0.896 which is much better.

### Recommending 5 Products.

from collections import defaultdict
def get_top_n(predictions, n=5):
  
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

top_n = get_top_n(test_pred, n=5)


for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])

Here are some 5 products recommendations 

## Recommendation System - Part III
* For a business without any user-item purchase history, a search engine based recommendation system can be designed for users. The product recommendations can be based on textual clustering analysis given in product description.
* **Dataset : **[Home Depot's dataset with product dataset.](https://www.kaggle.com/c/home-depot-product-search-relevance/data)

# Importing libraries

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

### Item to item based recommendation system based on product description

Applicable when business is setting up its E-commerce website for the first time

product_descriptions = pd.read_csv('product_descriptions.csv')
product_descriptions.shape

#### Checking for missing values

# Missing values

product_descriptions = product_descriptions.dropna()
product_descriptions.shape
product_descriptions.head()

product_descriptions1 = product_descriptions.head(500)
# product_descriptions1.iloc[:,1]

product_descriptions1["product_description"].head(10)

#### Feature extraction from product descriptions

Converting the text in product description into numerical data for analysis

vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_descriptions1["product_description"])
X1

#### Visualizing product clusters in subset of data

# Fitting K-Means to the dataset

X=X1

kmeans = KMeans(n_clusters = 10, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)
plt.plot(y_kmeans, ".")
plt.show()


def print_cluster(i):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

## Output
* Recommendation of product based on the current product selected by user.
* To recommend related product based on, Frequently bought together. 

#### Top words in each cluster based on product description

# # Optimal clusters is 

true_k = 10

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X1)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print_cluster(i)

#### Predicting clusters based on key search words

def show_recommendations(product):
    #print("Cluster ID:")
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    #print(prediction)
    print_cluster(prediction[0])

* **Keyword : ** cutting tool

show_recommendations("cutting tool")

* **Keyword : **spray paint

show_recommendations("spray paint")

* **Keyword : **steel drill

show_recommendations("steel drill")

In case a word appears in multiple clusters, the algorithm chooses the cluster with the highest frequency of occurance of the word.

* **Keyword : **water

show_recommendations("water")

Once a cluster is identified based on the user's search words, the recommendation system can display items from the corresponding product clusters based on the product descriptions.

## Evaluation of all models

#### Popularity based Model

import sklearn.metrics as metric
from math import sqrt
MSE = metric.mean_squared_error(pred_df['true_ratings'], pred_df['predicted_ratings'])
print('The RMSE value for Recommender model is', sqrt(MSE))

#### Collaborative Based Filtering

print(len(testset))
type(testset)

#### KNN with means

# Evalute on test set
test_pred = model.test(testset)
test_pred[0]

# compute RMSE
accuracy.rmse(test_pred)

#### SVD

# Use the "best model" for prediction
gs.test(testset)
accuracy.rmse(gs.test(testset))

SVD has much better results over others

#### Summary: 

This works best if a business is setting up its e-commerce website for the first time and does not have user-item purchase/rating history to start with initially. This recommendation system will help the users get a good recommendation to start with and once the buyers have a purchased history, the recommendation engine can use the model-based collaborative filtering technique.


