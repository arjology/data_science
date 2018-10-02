## Predicting Loan Applications with XGBoost  <a name="loans"></a>

![Jupyter notebook](https://github.com/arjology/data_science/loan_prediction/loan_prediction.ipynb)

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

