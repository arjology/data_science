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
def model(features, test_features, encoding = 'ohe', n_folds = 5):
    
    """Train and test a light gradient boosting model using
    cross validation. 
    
    Parameters
    --------
        features (pd.DataFrame): 
            dataframe of training features to use 
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame): 
            dataframe of testing features to use
            for making predictions with the model. 
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): 
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame): 
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame): 
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
        
    """
    
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        # Create a label encoder
        label_encoder = LabelEncoder()
        
        # List for storing categorical indices
        cat_indices = []
        
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # Extract feature names
    feature_names = list(features.columns)
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
    # Create the kfold object
    k_fold = model_selection.KFold(n_splits = n_folds, shuffle = False, random_state = 50)
    
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics

submission, fi, metrics = model(train_corrs_removed, test_corrs_removed)

#############################
### Modeling with XGBoost ###
#############################
features = train_corrs_removed
test_features = test_corrs_removed

# Extract the ids
train_ids = features['SK_ID_CURR']
test_ids = test_features['SK_ID_CURR']

# Extract the labels for training
labels = features['TARGET']

# Remove the ids and target
features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
test_features = test_features.drop(columns = ['SK_ID_CURR'])

# Create a label encoder
label_encoder = LabelEncoder()

# List for storing categorical indices
cat_indices = []

# Iterate through each column
for i, col in enumerate(features):
    if features[col].dtype == 'object':
        # Map the categorical features to integers
        features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
        test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

        # Record the categorical indices
        cat_indices.append(i)

imp = Imputer(missing_values=np.nan, strategy='mean')

imp = imp.fit(features)
features_imputed = imp.transform(features)

imp = imp.fit(test_features)
test_features_imputed = imp.transform(test_features)        

# Extract feature names
feature_names = list(features.columns)

# Convert to np arrays
features = np.array(features_imputed)
test_features = np.array(test_features_imputed)

# Empty array for feature importances
feature_importance_values = np.zeros(len(feature_names))

# Empty array for test predictions
test_predictions = np.zeros(test_features_imputed.shape[0])

params = {'scale_pos_weight':1,
          'learning_rate':0.5,  
          'colsample_bytree':0.5,
          'subsample':.8,
          'objective':'binary:logistic', 
          'n_estimators':1000, 
          'reg_lambda':1,
          'max_depth':2, 
          'gamma':1,
          'alpha':1
         }
data_dmatrix = xgb.DMatrix(data=features_imputed,label=labels)
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, 
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

def rmsle_eval(y, y0):
    
    y0=y0.get_label()    
    assert len(y) == len(y0)
    return 'error',np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

params = {'scale_pos_weight':1,
          'learning_rate':0.5,  
          'colsample_bytree':0.5,
          'subsample':.8,
          'objective':'binary:logistic', 
          'n_estimators':1000, 
          'reg_lambda':1,
          'max_depth':2, 
          'gamma':1,
          'alpha':1,
          'silent': 1,
          'n_jobs': -1
          
         }
watchlist  = [(data_dmatrix,'train')]
xg_reg = xgb.train(params=params,
                   dtrain=data_dmatrix,
                   num_boost_round=1000, 
                   evals=watchlist,
                   feval=rmsle_eval,
                   early_stopping_rounds=50)


pred_dmatrix = xgb.DMatrix(data=test_features_imputed)
test_predictions = xg_reg.predict(pred_dmatrix)

submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
submission.to_csv('submission_xgboost.csv', index = False)