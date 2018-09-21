# Table of Contents
1. [Introduction](#introduction)
2. [Linear Regression](#linearregression)
3. [Random Forrests and Gradient Boosting](#gradientboosting)
4. [Loan Predictions with XGBoost](#loans)

## Introduction <a name="introduction"></a>
Random data science projects. More to come...

## PySpark, MLlib, and Linear Regression <a name="linearregression"></a>

Here I am using Apache Spark’s spark.ml Linear Regression to predicting Boston housing prices. 
The data is from the [Kaggle competition: Housing Values in Suburbs of Boston](https://www.kaggle.com/c/boston-housing/data). 

* For each house observation, we have the following information:
* CRIM — per capita crime rate by town.
* ZN — proportion of residential land zoned for lots over 25,000 sq.ft.
* INDUS — proportion of non-retail business acres per town.
* CHAS — Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
* NOX — nitrogen oxides concentration (parts per 11 million).  
* RM — average number of rooms per dwelling.  
* AGE — proportion of owner-occupied units built prior to 1940.  
* DIS — weighted mean of distances to five Boston employment centres.  RAD — index of accessibility to radial highways.  TAX — full-value property-tax rate per $10,000.  PTRATIO — pupil-teacher ratio by town.  
* BLACK — 1000(Bk — 0.63)² where Bk is the proportion of blacks by town.
* LSTAT — lower status of the population (percent).
* MEDV — median value of owner-occupied homes in $1000s. This is the target variable.

The goal here is to develop a model that predicts the mean value of a given house in the area utilizing details of various other houses in the data set.

## Random Forrests, Gradient Boosting, and the 2016 U.S. elections <a name="gradientboosting"></a>

The 2016 U.S. Presidential elections were nothing short of spectacular failures of prediction models. I was curious if there was an explanation for which segments of the population might have had profound effects. 

This is the start of a longer exploratory project, but in the meantime I have been playing with random forrests and gradient boosting techniques.

The data is all freely available but just takes some tweeking.
The [jupyter notebook](https://github.com/arjology/data_science/blob/master/US%20voting%20and%20census.ipynb) has all the steps for the downloading and processing of the various data sources. 

There is of course some correlation between the population density, as well as gender-specific populations, and the spread (i.e. (votes_dem - votes_rep)/tot_votes)
![Pops vs Spread](https://github.com/arjology/data_science/blob/master/figures/US_voting_spread_vols_vs_pop_density.png)

Let's take a look first at the results, colered in each state at the county level:

![Percent Democrat and Republican](https://github.com/arjology/data_science/blob/master/figures/US_voting_pct_gop_dem.png) 

There is a clear difference between the central states and the coasts (and hence the coastal elites).

...

Ultimately, a simple random forrest had enough predictive power given the various demographic breakdowns of each county. But we can still do better.
![Random forrest predictions](https://github.com/arjology/data_science/blob/master/figures/US_voting_RF_binary_classification.png)

## Loans  <a name="linearregression"></a>

### Problem Statement
#### About Company
Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan.

#### Problem
Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.

### Data
#### Variable: Description
* Loan_ID: Unique Loan ID
* Gender: Male/ Female
* Married: Applicant married (Y/N)
* Dependents: Number of dependents
* Education: Applicant Education (Graduate/ Under Graduate)
* Self_Employed: Self employed (Y/N)
* ApplicantIncome: Applicant income
* CoapplicantIncome: Coapplicant income
* LoanAmount: Loan amount in thousands
* Loan_Amount_Term: Term of loan in months
* Credit_History: credit history meets guidelines
* Property_Area: Urban/ Semi Urban/ Rural
* Loan_Status: Loan approved (Y/N)


### Note: 
1. Evaluation Metric is accuracy i.e. percentage of loan approval you correctly predict.
2. You are expected to upload the solution in the format of "sample_submission.csv"

### Correlations

Let's first have a look at how the data is cross-correlated:
![Application factor correlations](https://github.com/arjology/data_science/blob/master/figures/loan_prediction_correlations.png)

### Gradient boosting regularization

Regularization via shrinkage (`learning_rate < 1.0`) improves performance considerably.
In combination with shrinkage, stochastic gradient boosting (`subsample < 1.0`) can produce more accurate models by reducing the variance via bagging.

![Regularization](https://github.com/arjology/data_science/blob/master/figures/loan_prediction_shrinkage.png)

### Visualizing individual trees from the fully boosted model

After running the model with `num_boost_round=100` and parameters:
`{"objective": "binary:logistic", "learning_rate": 0.1, "max_depth": 2, "n_esimators":200, "subsample":0.1}`

![Trees](https://github.com/arjology/data_science/blob/master/figures/loan_prediction_tree.png)

### Feature importance

We can view the importance of each feature column from the original data by simply counting the number of times each feature is split on across all boostingtrees (rounds) in the model. As you can see, the `Applicant Income` and `Loan Ammount` are scored with the highest importance among all features. This gives us a nice way of doing feature selection going forward.

![Features](https://github.com/arjology/data_science/blob/master/figures/loan_prediction_feature_importances.png)

