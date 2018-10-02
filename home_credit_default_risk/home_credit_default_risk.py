import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


# Can use a function here to streamline the process
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

