# Home Credit Default Risk

## Table of Contents
1. [Introduction](#introduction)
2. [LightGB](#lightgb)
3. [XGBoost](#xgboost)
4. [Apache Spark](#spark)
5. [Results](#results)


## Introduction <a name="introduction"></a>

Using alternatives to credit histories, can we predict the ability of a borrower to repay a loan ? 
[Kaggle competition](https://www.kaggle.com/c/home-credit-default-risk/overview)

Exploratory analysis has been done in the [Jupyter notebook](https://github.com/arjology/data_science/blob/master/home_credit_default_risk/home_credit_default_risk.ipynb)

## LightGB <a name="lightgb"></a>

### Application data

#### Correlations within the first data set

![correlations](https://github.com/arjology/data_science/blob/master/figures/home_credit_application_train_corr_heatmap.png)

#### Exploring feature importance

A small random forrest can gives us insights into which features are most important from this dataset. Normalized scores from external sources are ranked quite high.

![RF_feature_importance](https://github.com/arjology/data_science/blob/master/figures/home_credit_application_train_random_forrest_feat_importance.png)

We can also use gradient boosting to see if there is any large discrepancy in terms of importance.

![GB_feature_importance](https://github.com/arjology/data_science/blob/master/figures/home_credit_application_train_gb_feat_importance.png)

How well do the most important features differentiate between the default and repay groups? 

![feat_importance_diff](https://github.com/arjology/data_science/blob/master/figures/home_credit_application_important_factors_diff.png)

What about including information from the bureau statistics? The most important features are:

```
[('bureau_DAYS_CREDIT_mean', 0.08972896721991676),
 ('DAYS_BIRTH', 0.07823930831003195),
 ('bureau_DAYS_CREDIT_min', 0.07524825103004934),
 ('bureau_DAYS_CREDIT_UPDATE_mean', 0.06892735266963707),
 ('REGION_RATING_CLIENT_W_CITY', 0.060892667564898695),
 ('REGION_RATING_CLIENT', 0.058899014945840766),
 ('bureau_DAYS_ENDDATE_FACT_min', 0.05588737984397668),
 ('DAYS_LAST_PHONE_CHANGE', 0.05521851758157063),
 ('NAME_EDUCATION_TYPE', 0.054698601571243),
 ('CODE_GENDER', 0.05469226185566661)]
 ```

 But only the `DAYS_CREDIT_MEAN` seems to differentiate well:

![bureau_stats_diff](https://github.com/arjology/data_science/blob/master/figures/home_credit_bureau_stats_diff.png)

#### Categorical Data

Of course, we expect a certain amount of the information to be non-numeric in type:

```
SK_ID_CURR                  int64
SK_ID_BUREAU                int64
CREDIT_ACTIVE              object
CREDIT_CURRENCY            object
DAYS_CREDIT                 int64
CREDIT_DAY_OVERDUE          int64
DAYS_CREDIT_ENDDATE       float64
DAYS_ENDDATE_FACT         float64
AMT_CREDIT_MAX_OVERDUE    float64
CNT_CREDIT_PROLONG          int64
AMT_CREDIT_SUM            float64
AMT_CREDIT_SUM_DEBT       float64
AMT_CREDIT_SUM_LIMIT      float64
AMT_CREDIT_SUM_OVERDUE    float64
CREDIT_TYPE                object
DAYS_CREDIT_UPDATE          int64
AMT_ANNUITY               float64
dtype: object
```

Using previous loans from other institutions, we have increased siginificantly the number of features:
```
Original Number of Features:  122
Number of features using previous loans from other institutions data:  394
```

After adding all the data sets:
```
Dataframe has 948 columns.
892 columns have missing values.
```

However, we expect there to be missing data as well! Of these, 6 have over 90% of the rows missing.
```
	Missing Values	% Missing of Total
previous_application_RATE_INTEREST_PRIMARY_max	302902	99.0
previous_application_RATE_INTEREST_PRIVILEGED_min	302902	99.0
previous_application_RATE_INTEREST_PRIVILEGED_max	302902	99.0
previous_application_RATE_INTEREST_PRIVILEGED_mean	302902	99.0
previous_application_RATE_INTEREST_PRIMARY_mean	302902	99.0
previous_application_RATE_INTEREST_PRIMARY_min	302902	99.0
credit_card_balance_AMT_PAYMENT_CURRENT_min	246451	80.0
credit_card_balance_AMT_PAYMENT_CURRENT_mean	246451	80.0
credit_card_balance_AMT_PAYMENT_CURRENT_max	246451	80.0
credit_card_balance_CNT_DRAWINGS_POS_CURRENT_max	246371	80.0
credit_card_balance_CNT_DRAWINGS_POS_CURRENT_mean	246371	80.0
credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_mean	246371	80.0
credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_max	246371	80.0
credit_card_balance_CNT_DRAWINGS_OTHER_CURRENT_min	246371	80.0
credit_card_balance_AMT_DRAWINGS_OTHER_CURRENT_mean	246371	80.0
credit_card_balance_CNT_DRAWINGS_OTHER_CURRENT_max	246371	80.0
credit_card_balance_CNT_DRAWINGS_OTHER_CURRENT_mean	246371	80.0
credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_min	246371	80.0
credit_card_balance_AMT_DRAWINGS_ATM_CURRENT_min	246371	80.0
credit_card_balance_AMT_DRAWINGS_ATM_CURRENT_max	246371	80.0
credit_card_balance_CNT_DRAWINGS_POS_CURRENT_min	246371	80.0
credit_card_balance_AMT_DRAWINGS_OTHER_CURRENT_max	246371	80.0
credit_card_balance_AMT_DRAWINGS_OTHER_CURRENT_min	246371	80.0
credit_card_balance_AMT_DRAWINGS_ATM_CURRENT_mean	246371	80.0
credit_card_balance_AMT_DRAWINGS_POS_CURRENT_mean	246371	80.0
```

And similarly, for the testing data:

```
Dataframe has 947 columns.
889 columns have missing values.
```

```
	Missing Values	% Missing of Total
previous_application_RATE_INTEREST_PRIMARY_max	47632	98.0
previous_application_RATE_INTEREST_PRIVILEGED_min	47632	98.0
previous_application_RATE_INTEREST_PRIMARY_mean	47632	98.0
previous_application_RATE_INTEREST_PRIVILEGED_max	47632	98.0
previous_application_RATE_INTEREST_PRIVILEGED_mean	47632	98.0
previous_application_RATE_INTEREST_PRIMARY_min	47632	98.0
credit_card_balance_CNT_DRAWINGS_POS_CURRENT_mean	37690	77.0
credit_card_balance_CNT_DRAWINGS_OTHER_CURRENT_mean	37690	77.0
credit_card_balance_CNT_DRAWINGS_OTHER_CURRENT_max	37690	77.0
credit_card_balance_CNT_DRAWINGS_OTHER_CURRENT_min	37690	77.0
credit_card_balance_AMT_DRAWINGS_OTHER_CURRENT_mean	37690	77.0
credit_card_balance_AMT_DRAWINGS_OTHER_CURRENT_max	37690	77.0
credit_card_balance_CNT_DRAWINGS_POS_CURRENT_max	37690	77.0
credit_card_balance_CNT_DRAWINGS_POS_CURRENT_min	37690	77.0
credit_card_balance_AMT_DRAWINGS_POS_CURRENT_max	37690	77.0
credit_card_balance_AMT_DRAWINGS_ATM_CURRENT_min	37690	77.0
credit_card_balance_AMT_DRAWINGS_ATM_CURRENT_max	37690	77.0
credit_card_balance_AMT_DRAWINGS_POS_CURRENT_min	37690	77.0
credit_card_balance_AMT_DRAWINGS_ATM_CURRENT_mean	37690	77.0
credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_max	37690	77.0
credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_mean	37690	77.0
credit_card_balance_AMT_DRAWINGS_POS_CURRENT_mean	37690	77.0
credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_min	37690	77.0
credit_card_balance_AMT_DRAWINGS_OTHER_CURRENT_min	37690	77.0
credit_card_balance_AMT_PAYMENT_CURRENT_min	37684	77.0
```

#### Most correlated columns

Columns with the most positive correlations to the target:

```
	TARGET
TARGET	1.000000
credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_mean	0.107692
credit_card_balance_CNT_DRAWINGS_CURRENT_max	0.101389
bureau_DAYS_CREDIT_mean	0.089729
client_bureau_balance_MONTHS_BALANCE_min_mean	0.089038
credit_card_balance_AMT_BALANCE_mean	0.087177
credit_card_balance_AMT_TOTAL_RECEIVABLE_mean	0.086490
credit_card_balance_AMT_RECIVABLE_mean	0.086478
credit_card_balance_AMT_RECEIVABLE_PRINCIPAL_mean	0.086062
credit_card_balance_CNT_DRAWINGS_CURRENT_mean	0.082520

```

And negative:

```
	TARGET
client_bureau_balance_STATUS_C_count_mean	-0.062954
previous_application_NAME_CONTRACT_STATUS_Approved_count_norm	-0.063521
client_bureau_balance_MONTHS_BALANCE_count_max	-0.068792
previous_application_CODE_REJECT_REASON_XAP_count_norm	-0.073930
CREDIT_ACTIVE_Closed_count_norm	-0.079369
bureau_CREDIT_ACTIVE_Closed_count_norm	-0.079369
client_bureau_balance_MONTHS_BALANCE_count_mean	-0.080193
EXT_SOURCE_1	-0.155317
EXT_SOURCE_2	-0.160472
EXT_SOURCE_3	-0.178919
```

![pos_neg_corrs](https://github.com/arjology/data_science/blob/master/figures/home_credit_pos_neg_corrs.png)

#### Colinearity

Columns that are above a threshold of 0.8 we can remove (396 co-linear columns in total), leaving us with 545 columns.

### Using Gradient Boosting with LightGB

Running the model:

```
Training Data Shape:  (307511, 665)
Testing Data Shape:  (48744, 665)
Training until validation scores don't improve for 100 rounds.
[200]	valid's auc: 0.779522	valid's binary_logloss: 0.538929	train's auc: 0.825995	train's binary_logloss: 0.526054
[400]	valid's auc: 0.780505	valid's binary_logloss: 0.514438	train's auc: 0.861452	train's binary_logloss: 0.490052
Early stopping, best iteration is:
[323]	valid's auc: 0.78075	valid's binary_logloss: 0.522973	train's auc: 0.849285	train's binary_logloss: 0.50284
Training until validation scores don't improve for 100 rounds.
[200]	valid's auc: 0.78027	valid's binary_logloss: 0.539718	train's auc: 0.825654	train's binary_logloss: 0.525935
[400]	valid's auc: 0.782096	valid's binary_logloss: 0.513737	train's auc: 0.861764	train's binary_logloss: 0.489628
Early stopping, best iteration is:
[388]	valid's auc: 0.782229	valid's binary_logloss: 0.515039	train's auc: 0.860004	train's binary_logloss: 0.49154
Training until validation scores don't improve for 100 rounds.
[200]	valid's auc: 0.771893	valid's binary_logloss: 0.539808	train's auc: 0.827193	train's binary_logloss: 0.524101
[400]	valid's auc: 0.773948	valid's binary_logloss: 0.514694	train's auc: 0.862809	train's binary_logloss: 0.487946
Early stopping, best iteration is:
[475]	valid's auc: 0.774418	valid's binary_logloss: 0.506341	train's auc: 0.873576	train's binary_logloss: 0.476174
Training until validation scores don't improve for 100 rounds.
[200]	valid's auc: 0.778459	valid's binary_logloss: 0.538044	train's auc: 0.825825	train's binary_logloss: 0.525914
[400]	valid's auc: 0.779532	valid's binary_logloss: 0.513873	train's auc: 0.861201	train's binary_logloss: 0.489868
Early stopping, best iteration is:
[356]	valid's auc: 0.779787	valid's binary_logloss: 0.51875	train's auc: 0.8544	train's binary_logloss: 0.497065
Training until validation scores don't improve for 100 rounds.
[200]	valid's auc: 0.781147	valid's binary_logloss: 0.538431	train's auc: 0.825405	train's binary_logloss: 0.526119
[400]	valid's auc: 0.781902	valid's binary_logloss: 0.515182	train's auc: 0.861293	train's binary_logloss: 0.490137
Early stopping, best iteration is:
[481]	valid's auc: 0.78225	valid's binary_logloss: 0.506862	train's auc: 0.87286	train's binary_logloss: 0.477572
```

#### Feature importance
We see the following features in order of importance:

```
feature	importance
22	EXT_SOURCE_1	418.4
24	EXT_SOURCE_3	351.8
23	EXT_SOURCE_2	349.0
5	DAYS_BIRTH	278.4
2	AMT_CREDIT	253.8
3	AMT_ANNUITY	209.0
514	previous_application_CNT_PAYMENT_mean	200.8
6	DAYS_EMPLOYED	186.0
111	bureau_DAYS_CREDIT_ENDDATE_max	146.0
105	bureau_DAYS_CREDIT_max	145.4
8	DAYS_ID_PUBLISH	143.8
249	POS_cash_balance_CNT_INSTALMENT_mean	141.0
263	installments_payments_AMT_INSTALMENT_max	126.8
9	OWN_CAR_AGE	124.4
7	DAYS_REGISTRATION	124.2

```

![lightgb_feats](https://github.com/arjology/data_science/blob/master/figures/home_credit_lightgb_feats.png)

### Save results and done!
submission.to_csv("submission.csv", index=False)

## XGBoost <a href="xgboost"></a>

Using Scikit-Learn's XGBoost with the following parameters:

```
'scale_pos_weight':1,
'learning_rate':0.5,  
'colsample_bytree':0.5,
'subsample':.8,
'objective':'binary:logistic', 
'n_estimators':1000, 
'reg_lambda':1,
'max_depth':2, 
'gamma':1,
'alpha':1
```

![xgb_coors](https://github.com/arjology/data_science/blob/master/figures/home_credit_xgb_corrs.png)

Correlation plot from XG Boost
![xgb_coors](https://github.com/arjology/data_science/blob/master/figures/home_credit_xgb_corrs.png)

## Results <a href="results"></a>

### LightGB
Private score: `0.77583`
Public score: `0.77499`

### XGBoost
Private score: `0.74768`
Public score: `0.75189`

### Spark

