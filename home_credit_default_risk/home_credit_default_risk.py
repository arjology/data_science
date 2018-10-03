import numpy as np
import scipy as sp
import pandas as pd
from typing import Tuple, List, Union, Iterable

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, roc_auc_score
import xgboost as xgb

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix  


########################
### Helper functions ###
########################
# Categorize and count a column
def categorize_and_count(df: pd.DataFrame, 
                         name: str,
                         group_by_variable: str='SK_ID_CURR'
                        ) -> pd.DataFrame:
    categorical_df = pd.get_dummies(df.select_dtypes('object'))
    categorical_df[group_by_variable] = df[group_by_variable]
    categorical_df = categorical_df.groupby(group_by_variable).agg(['sum', 'mean'])
    
    columns = []
    for var in categorical_df.columns.levels[0]:
            for stat in ['count', 'count_norm']:
                columns.append('%s_%s_%s' % (name, var, stat))
    categorical_df.columns = columns
    return categorical_df

# Return the statistics on a selected column
def aggregate_numerically(df: pd.DataFrame,
                          group_by_variable: str,
                          name: str
                         ) -> pd.DataFrame:
    for col in df:
        if col != group_by_variable and 'SK_ID' in col:
            df = df.drop(columns=col)
    group_IDs = df[group_by_variable]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_by_variable] = group_IDs
    aggregate_df = numeric_df.groupby(group_by_variable).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
    columns = [group_by_variable]
    for var in aggregate_df.columns.levels[0]:
        if var != group_by_variable:
            for stat in ag gregate_df.columns.levels[1][:-1]:
                columns.append('%s_%s_%s' % (name, var, stat))

    aggregate_df.columns = columns
    return aggregate_df    

# Find missing values by column
def find_missing_values_by_column(df: pd.DataFrame) -> pd.DataFrame:
    
    missing_values = df.isnull().sum()
    percent_missing = 100.0*df.isnull().sum()/len(df)
    missing_value_table = pd.concat([missing_values, percent_missing], axis=1)
    
    missing_value_renamed_columns = missing_value_table.rename(
        columns = {0: 'Missing Values', 1: '% Missing of Total'}
    )
    
    missing_value_renamed_columns = missing_value_renamed_columns[
        missing_value_renamed_columns.iloc[:,1] != 0
    ].sort_values('% Missing of Total', ascending=False).round()
    
    print ("Dataframe has " + str(df.shape[1]) + " columns.\n"      
           + str(missing_value_renamed_columns.shape[0]) +
              " columns have missing values.")
    
    return missing_value_renamed_columns

#################
### Load data ###
#################
application_test = pd.read_csv("../data/home_credit_default_risk/application_test.csv")
application_train = pd.read_csv("../data/home_credit_default_risk/application_train.csv")
POS_CASH_balance = pd.read_csv("../data/home_credit_default_risk/POS_CASH_balance.csv")
# HomeCredit_columns_description = pd.read_csv("../data/home_credit_default_risk/HomeCredit_columns_description.csv")
bureau = pd.read_csv("../data/home_credit_default_risk/bureau.csv")
bureau_balance = pd.read_csv("../data/home_credit_default_risk/bureau_balance.csv")
credit_card_balance = pd.read_csv("../data/home_credit_default_risk/credit_card_balance.csv")
installments_payments = pd.read_csv("../data/home_credit_default_risk/installments_payments.csv")
previous_application = pd.read_csv("../data/home_credit_default_risk/previous_application.csv")
sample_submission = pd.read_csv("../data/home_credit_default_risk/sample_submission.csv")

###########################
### Feature engineering ###
###########################
# Including information on previous loans
training_df = application_train.join(bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'}), on='SK_ID_CURR', rsuffix='_r')
training_df = training_df.drop(['SK_ID_CURR_r'], axis=1)
training_df['previous_loan_counts'] = training_df['previous_loan_counts'].fillna(0)

# Statistics about bureau data
bureau_stats = bureau.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
columns = ['SK_ID_CURR']
for var in bureau_stats.columns.levels[0]:
    if var != 'SK_ID_CURR':
        for stat in bureau_stats.columns.levels[1][:-1]:
            columns.append('bureau_%s_%s' % (var, stat))
bureau_stats.columns = columns
training_df = training_df.merge(bureau_stats, on='SK_ID_CURR', how='left')

bureau_agg_new = aggregate_numerically(bureau.drop(columns = ['SK_ID_BUREAU']), group_by_variable='SK_ID_CURR', name='bureau')

# Taking care of categorical data

categorical = pd.get_dummies(bureau.select_dtypes('object'))
categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
categorical_grouped = categorical.groupby('SK_ID_CURR').agg(['sum', 'mean'])
columns = []
for var in categorical_grouped.columns.levels[0]:
    if var != 'SK_ID_CURR':
        for stat in ['count', 'count_norm']:
            columns.append('%s_%s' % (var, stat))
categorical_grouped.columns = columns
training_df = training_df.merge(categorical_grouped, left_on = 'SK_ID_CURR', right_index = True, how = 'left')

# Adding more categorical data

bureau_counts = categorize_and_count(df=bureau, group_by_variable='SK_ID_CURR', name='bureau')
bureau_balance_counts = categorize_and_count(df=bureau_balance, group_by_variable='SK_ID_BUREAU', name='bureau_balance')
bureau_balance_agg = aggregate_numerically(df=bureau_balance, group_by_variable='SK_ID_BUREAU', name='bureau_balance')
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index=True, left_on='SK_ID_BUREAU', how='outer')
bureau_by_loan = bureau_by_loan.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], on='SK_ID_BUREAU', how='left')
bureau_balance_by_client = aggregate_numerically(df=bureau_by_loan.drop(columns=['SK_ID_BUREAU']), group_by_variable='SK_ID_CURR', name='client')
training_df = training_df.merge(bureau_balance_by_client, on='SK_ID_CURR', how='left')

### Other sources

# Credit card balances
credit_card_balance_counts = categorize_and_count(df=credit_card_balance,
                                                  group_by_variable='SK_ID_CURR',
                                                  name='credit_card_balance'
                                                 )
credit_card_balance_agg = aggregate_numerically(df=credit_card_balance.drop(columns = ['SK_ID_PREV']),
                                                group_by_variable='SK_ID_CURR',
                                                name='credit_card_balance'
                                               )
training_df = training_df.merge(credit_card_balance_counts, on = 'SK_ID_CURR', how = 'left')
training_df = training_df.merge(credit_card_balance_agg, on = 'SK_ID_CURR', how = 'left')

# Point of sale cash balances
POS_CASH_balance_counts = categorize_and_count(df=POS_CASH_balance,
                                                  group_by_variable='SK_ID_CURR',
                                                  name='POS_cash_balance'
                                                 )
POS_CASH_balance_agg = aggregate_numerically(df=POS_CASH_balance.drop(columns = ['SK_ID_PREV']),
                                                group_by_variable='SK_ID_CURR',
                                                name='POS_cash_balance'
                                               )
training_df = training_df.merge(POS_CASH_balance_counts, on = 'SK_ID_CURR', how = 'left')
training_df = training_df.merge(POS_CASH_balance_agg, on = 'SK_ID_CURR', how = 'left')

# Installment payments
installments_payments_agg = aggregate_numerically(df=installments_payments.drop(columns = ['SK_ID_PREV']),
                                                group_by_variable='SK_ID_CURR',
                                                name='installments_payments'
                                               )
training_df = training_df.merge(installments_payments_agg, on = 'SK_ID_CURR', how = 'left')

# Previous applications
previous_application_counts = categorize_and_count(df=previous_application,
                                                  group_by_variable='SK_ID_CURR',
                                                  name='previous_application'
                                                 )
previous_application_agg = aggregate_numerically(df=previous_application.drop(columns = ['SK_ID_PREV']),
                                                group_by_variable='SK_ID_CURR',
                                                name='previous_application'
                                               )
training_df = training_df.merge(previous_application_counts, on = 'SK_ID_CURR', how = 'left')
training_df = training_df.merge(previous_application_agg, on = 'SK_ID_CURR', how = 'left')

### Adding other sources to the test data
test = application_test.join(bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'}), on='SK_ID_CURR', rsuffix='_r')
test = test.drop(['SK_ID_CURR_r'], axis=1)
test['previous_loan_counts'] = test['previous_loan_counts'].fillna(0)
test = test.merge(categorical_grouped, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_stats, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
test = test.merge(credit_card_balance_counts, on = 'SK_ID_CURR', how = 'left')
test = test.merge(credit_card_balance_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(POS_CASH_balance_counts, on = 'SK_ID_CURR', how = 'left')
test = test.merge(POS_CASH_balance_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(installments_payments_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(previous_application_counts, on = 'SK_ID_CURR', how = 'left')
test = test.merge(previous_application_agg, on = 'SK_ID_CURR', how = 'left')

train = train.drop(columns=training_missing_vars)
test = test.drop(columns=testing_missing_vars)

### Removing colinear columns
thresh = 0.8
above_thresh_vars = {}
for col in corrs:
    above_thresh_vars[col] = list(corrs.index[corrs[col] > thresh])
    
cols_to_remove = []
cols_seen = []
cols_to_remove_pair = []
for key, value in above_thresh_vars.items():
    cols_seen.append(key)
    for x in value:
        if x == key:
            next
        else:
            if x not in cols_seen:
                cols_to_remove.append(x)
                cols_to_remove_pair.append(key)
cols_to_remove = list(set(cols_to_remove))
train_corrs_removed = train.drop(columns = cols_to_remove)
test_corrs_removed = test.drop(columns = cols_to_remove)


#############################
### Modeling with LightGB ###
#############################
