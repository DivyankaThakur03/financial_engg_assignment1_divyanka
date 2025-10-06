#!/usr/bin/env python
# coding: utf-8

# <h2><font color='#306998'><center>451 Financial Engineering: Programming Assignment 1</center></font></h2>
# <h3><center>Thomas W. Miller, June 18, 2025</center></h3>
# ---

# ## Overview 
# We use machine learing classifiers, including tree-based ensemble boosting methods, to predict the direction of oil futures (up or down) using a number of lagged price features. In particular, we look at daily closing spot prices for [West Texas Intermediate (ticker WTI)](https://en.wikipedia.org/wiki/West_Texas_Intermediate) with lags of one to seven days, as well as features based on opening, closing, high, and low price points, and daily trading volume.
# 
# A model for predicting the direction of daily returns sets the stage for testing the predictive utility of additional features. The domain of potential features or leading indicators is wide, including those associated with other price series, economic indicators, international events, securities filings, analyst and news reports, and media measures.

# ### Import Libraries
# We draw on Python packages for data manipulation and modeling. Most important are Polars, a high-performance alternative to Pandas for data manipulation, and Scikit-Learn for machine learning study design and modeling algorithms.

# In[1]:


import os
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import Python Packages for data manipulation, data pipelines, and databases
import numpy as np
import pyarrow # foundation for polars
import polars as pl # DataFrame work superior to Pandas

# Plotting
import matplotlib.pyplot as plt
# Display static plots directly in the notebook output 
# get_ipython().run_line_magic('matplotlib', 'inline')
# create stylized visualizations, including heat maps
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (RandomizedSearchCV, 
                                    TimeSeriesSplit)
from sklearn.model_selection import cross_validate

# utilized in all possible subsets classification work
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# needed for randomized search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# metrics in xgboost tuning and final model evaluation
from sklearn.metrics import (accuracy_score,
                             classification_report, 
                             roc_curve, 
                             roc_auc_score,
                             RocCurveDisplay,
                             ConfusionMatrixDisplay,
                             confusion_matrix,
                             precision_score,
                             recall_score,
                             f1_score
                            )

# XGBoost Package... more complete than SciKit-Learn boosting methods
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier, plot_importance

# import yfinance as yf  # used earlier to obtain the price series
# import yfinance as yf

import warnings
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore') 


# ### Retrieve Data
# In previous work, we retrieved price data for WTI from Yahoo Finance. The code is shown in the next commented-out cell.

# In[2]:


'''
Previous work to retrieve data from Yahoo Finance

symbol = 'WTI'
start_date = '2000-01-01'
end_date = '2025-05-27'

symbol = 'WTI'
ticker = yf.Ticker(symbol)
historical_data = ticker.history(start = start_date, end = end_date, period = '1mo')
print(historical_data)

print("type of historical_data", type(historical_data))

historical_data.to_csv("wti_historical_data.csv")
'''


# ### Polars DataFrame Development
# The following code cell demonstrates Polars use with the time series DataFrame for our selected market/ticker, WTI.

# In[3]:


wti = pl.read_csv("msds_getdata_yfinance_nvdl.csv", try_parse_dates=True)

# check the original schema
print(wti.schema)

# drop useless columns Dividends and StockSplits
wti = wti.drop(['Dividends', 'Stock Splits'])

# create lag price features
wti = wti.with_columns((pl.col('Close')).shift().alias('CloseLag1'))
wti = wti.with_columns((pl.col('CloseLag1')).shift().alias('CloseLag2'))
wti = wti.with_columns((pl.col('CloseLag2')).shift().alias('CloseLag3'))

# create high-minus-low (HML) for day and its lags
wti = wti.with_columns((pl.col('High') - pl.col('Low')).alias('HML'))
wti = wti.with_columns((pl.col('HML')).shift().alias('HMLLag1'))
wti = wti.with_columns((pl.col('HMLLag1')).shift().alias('HMLLag2'))
wti = wti.with_columns((pl.col('HMLLag2')).shift().alias('HMLLag3'))

# create a net change for the day as the open minus closing price OMC
# also create the corresponding lag metrics
wti = wti.with_columns((pl.col('Open') - pl.col('Close')).alias('OMC'))
wti = wti.with_columns((pl.col('OMC')).shift().alias('OMCLag1'))
wti = wti.with_columns((pl.col('OMCLag1')).shift().alias('OMCLag2'))
wti = wti.with_columns((pl.col('OMCLag2')).shift().alias('OMCLag3'))

# create volume lag metrics
wti = wti.with_columns((pl.col('Volume')).shift().alias('VolumeLag1'))
wti = wti.with_columns((pl.col('VolumeLag1')).shift().alias('VolumeLag2'))
wti = wti.with_columns((pl.col('VolumeLag2')).shift().alias('VolumeLag3'))

# compute 10-day exponential moving averages of closing prices
# compute acround CloseLag1 to avoid any "leakage" in explanatory variable set
# note also the 10-day buffer between train and test in time-series cross-validation
wti = wti.with_columns((pl.col('CloseLag1').ewm_mean(half_life=1,ignore_nulls=True)).alias('CloseEMA2'))
wti = wti.with_columns((pl.col('CloseLag1').ewm_mean(half_life=2,ignore_nulls=True)).alias('CloseEMA4'))
wti = wti.with_columns((pl.col('CloseLag1').ewm_mean(half_life=4,ignore_nulls=True)).alias('CloseEMA8'))

# log daily returns
wti = wti.with_columns(np.log(pl.col('Close')/pl.col('CloseLag1')).alias('LogReturn'))

# set volume features to Float64 for subsequent use in Numpy arrays
wti = wti.with_columns(
    pl.col('Volume').cast(pl.Float64).round(0),
    pl.col('VolumeLag1').cast(pl.Float64).round(0),
    pl.col('VolumeLag2').cast(pl.Float64).round(0),
    pl.col('VolumeLag3').cast(pl.Float64).round(0),
    )

# round other features to three decimal places for reporting and subsequent analytics
wti = wti.with_columns(
    pl.col('Open').round(3),
    pl.col('High').round(3),    
    pl.col('Low').round(3),
    pl.col('Close').round(3),      
    pl.col('CloseLag1').round(3),
    pl.col('CloseLag2').round(3),  
    pl.col('CloseLag3').round(3),
    pl.col('HML').round(3),  
    pl.col('HMLLag1').round(3),
    pl.col('HMLLag2').round(3),  
    pl.col('HMLLag3').round(3),
    pl.col('OMC').round(3),  
    pl.col('OMCLag1').round(3),
    pl.col('OMCLag2').round(3),  
    pl.col('OMCLag3').round(3), 
    pl.col('CloseEMA2').round(3),
    pl.col('CloseEMA4').round(3), 
    pl.col('CloseEMA8').round(3))

# define binary target/response 1 = market price up since previous day, 0 = even or down 
wti = wti.with_columns(pl.when(pl.col('LogReturn')>0.0).then(pl.lit(1)).otherwise(pl.lit(0)).alias('Target'))

print(wti.schema)

# save to external comma-delimited text file for checking calculations in Excel
wti.write_csv("wti-with-computed-features.csv")


# ### Descriptive Statistics for Price Features

# In[4]:


# Drop the rows with null values such as the initial lag rows
wti = wti.drop_nulls()

# Descriptive statistics
wtiStatistics = wti.drop('Date').describe()

print(wtiStatistics.columns)

wtiStatisticsToPrint = wtiStatistics.transpose(include_header=True).drop(['column_1', 'column_5', 'column_7'])

print(wtiStatisticsToPrint.schema)

with pl.Config(
    tbl_rows = 60,
    tbl_width_chars = 200,
    tbl_cols = -1,
    float_precision = 3,
    tbl_hide_dataframe_shape = True,
    tbl_hide_column_data_types = True):
    print(wtiStatisticsToPrint)



# ### Feature List
# Features or explanatory variables, also known as an independent variables, are used to predict the values of target variables. The initial list of featrues includes the price-based features defined above, everything except the continuous response **LogReturn** if we wanted to employ regression and the binary response **Target** for classification, which is the focus of this project. This complete feature list is used in evaluating all methods.

# In[5]:


# Select Features for the Model, exclude current day price variables ... no "leakage"
# note for moving averages, we have excluded the current day, and provide a 10-day gap
# so these may be included in the set 
X = wti.drop(['Date', 'LogReturn', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'HML', 'OMC'])
X.head()


# ### Examine the Distribution of LogReturn Values

# In[6]:


# Define and examine the target for regression model development
print(wti['LogReturn'].describe())

y = np.array(wti['LogReturn'])


# ### Standardize All Features
# Standardization is carried out for the complete set of features.

# In[7]:


# Standardize features
featureNames = X.columns
print("Feature names correspond to Numpy array columns:",featureNames)
scaler = StandardScaler()
X = scaler.fit_transform(np.array(X))


# ### Target Definition for Classification (Target)
# 
# Identify the binary target variable for model development.

# In[8]:


# Define and examine the target for classification model development
print(wti['Target'].value_counts())

y = np.array(wti['Target'])


# ### Feature Selection Using All Possible Classifications 
# We draw on traditional applied statistis, selecting the best combination of features using the Akaike Information Criterion (AIC). 

# In[9]:


# Polars DataFrame for storing results from all possible subsets 
resultsSchema = {"trialNumber": pl.Int64, "features": pl.String, 'aic': pl.Float64} 
resultsDataFrame = pl.DataFrame(schema = resultsSchema) 

def getAIC(X, y): 
    model = LogisticRegression()
    model.fit(X, y)
    # Calculate log-likelihood
    loglik = -log_loss(y, model.predict_proba(X)) * len(y)    
    # Calculate the number of parameters
    k = X.shape[1] + 1    
    # Calculate AIC... smaller is better
    aic = 2 * k -2 * loglik     
    # print(f"AIC: {aic}") # print for initial testing
    return aic

from itertools import chain, combinations

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

trialNumber = 0 # initialize count
for c in powerset(range(X.shape[1])):
    trialNumber = trialNumber + 1
    # print(c)
    thisAIC = getAIC(X[:,c], y)
    thisTrialDataFrame = pl.DataFrame({"trialNumber": trialNumber,
                                       "features": ' '.join(map(str, c)),
                                       "aic": thisAIC},
                                        schema=resultsSchema)          
    resultsDataFrame = pl.concat([resultsDataFrame,thisTrialDataFrame])

# one more set of features... all features
trialNumber = trialNumber + 1
thisAIC = getAIC(X, y)
thisTrialDataFrame = pl.DataFrame({"trialNumber": trialNumber,
                                   "features": "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14",
                                   "aic": thisAIC},
                                    schema=resultsSchema, strict=False)          
resultsDataFrame = pl.concat([resultsDataFrame,thisTrialDataFrame])


# In[10]:


print(resultsDataFrame.sort('aic').head(10))


# ### Selected Feature Subset
# Reviewing the ten lowest *AIC* models, we selected five features for subsequent model development:
# - **CloseLag3** Lag-three daily closing price
# - **HMLLag1** Lag-one high minus low daily prices
# - **OMCLag2** Lag-two open minus closing daily prices
# - **OMCLag3** Lag-three open minus closing daily prices
# - **CloseEMA8** Exponential moving average across eight days

# In[11]:


# examine relationships among five selected features, along with LogReturn and CloseLag1
XStudy = wti.select('LogReturn','CloseLag1','CloseLag3','HMLLag1','OMCLag2','OMCLag3','CloseEMA8')

# prepare correlation heat map using seaborn
corrMatrix = XStudy.corr()
print(corrMatrix)
sns.heatmap(corrMatrix, cmap='coolwarm', annot=True)
plt.show()


# In[12]:


# select subset of five columns as features
X = wti.select('CloseLag3','HMLLag1','OMCLag2','OMCLag3','CloseEMA8')


# ### Define Cross-Validation Training and Test Sets
# Recognizing that time series observations are not independent observations, we use Scikit-Learn [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) for cross-validation. 
# 
# This cross-validation object is a variation of multi-fold cross-validation for independent observations. In the kth split, it returns first k folds as train set and the (k+1)th fold as test set.
# 
# Unlike cross-validation for independent observations, however, successive training sets are supersets of those that come before them. A listing of observation index numbers shows what this means for the time series market data in this demonstration.
# 

# In[13]:


# Splitting the datasets into train and test sets
# gap is the number of samples to exclude from 
# the end of each train set and before the next test set.
tscv = TimeSeriesSplit(gap=10, n_splits=5)

all_splits = list(tscv.split(X, y))
train_0, test_0 = all_splits[0]
train_1, test_1 = all_splits[1]
train_2, test_2 = all_splits[2]
train_3, test_3 = all_splits[3]
train_4, test_4 = all_splits[4]

# examine the objects created for cross-validation splits
print("type(all_splits):", type(all_splits), " outer list length", len(all_splits))
print()
print("train_0 has",len(train_0),"with indices from ",min(train_0),"to",max(train_0))
print("test_0 has",len(test_0),"with indices from ",min(test_0),"to",max(test_0))
print()
print("train_1 has",len(train_1),"with indices from ",min(train_1),"to",max(train_1))
print("test_1 has",len(test_1),"with indices from ",min(test_1),"to",max(test_1))
print()
print("train_2 has",len(train_2),"with indices from ",min(train_2),"to",max(train_2))
print("test_2 has",len(test_2),"with indices from ",min(test_2),"to",max(test_2))
print()
print("train_3 has",len(train_3),"with indices from ",min(train_3),"to",max(train_3))
print("test_3 has",len(test_3),"with indices from ",min(test_3),"to",max(test_3))
print()
print("train_4 has",len(train_4),"with indices from ",min(train_4),"to",max(train_4))
print("test_4 has",len(test_4),"with indices from ",min(test_4),"to",max(test_4))

# to see all indices we can uncomment these statements
# print("elements of all_splits list of lists,\n shows index numbers for each the five lists")
# print(all_splits)


# ### Define an Initial Classification Model to Be Evaluated¶
# We again select gradient boosting from the XGBoost package, this time defining a classification model to be evaluated. Hyperparameters are retained at defauld settings, except for n_estimators.

# In[14]:


model = XGBClassifier(objective='binary:logistic', n_estimators=1000, random_state=2025)


# ### Evaluate a Classification Model Within the Time Series Cross-Validation Design
# Prior to executing a full-blown search for the "best" classification model, we test the cross-validation design on a binary classification model, revising code provided in online documentation for Scikit-Learn: [Time-related feature engineerng](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html#sphx-glr-auto-examples-applications-plot-cyclical-feature-engineering-py). In particular, we define appropriate metrics for assessing classification performance.

# In[15]:


def evaluate(model, X, y, cv, model_prop=None, model_step=None):
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy"],
        return_estimator=model_prop is not None,
    )
    if model_prop is not None:
        if model_step is not None:
            values = [
                getattr(m[model_step], model_prop) for m in cv_results["estimator"]
            ]
        else:
            values = [getattr(m, model_prop) for m in cv_results["estimator"]]
        print(f"Mean model.{model_prop} = {np.mean(values)}")
    accuracy = -cv_results["test_accuracy"]

    # print used in earlier testing
    # print(
    #    f"Mean Accuracy:     {-accuracy.mean():.3f} +/- {accuracy.std():.3f}\n"
    # )
    return (-accuracy.mean(), accuracy.std())

evaluate(model, X, y, cv=tscv, model_prop="n_estimators")



# In[16]:


# print results from evaluate
accuracyMean, accuracyStd = evaluate(model, X, y, cv=tscv, model_prop="n_estimators")
print(
        f"Mean Accuracy:     {accuracyMean:.3f} +/- {accuracyStd:.3f}\n"
     )


# ### Randomized Search for Hyperparameter Settings
# We search for effective values on five XGBoost hyperparameters: **max_depth**, **min_child_weight**, **subsample**, **learning_rate**, and **n_estimators**.

# In[17]:


# Randomized search to find the best set of hyperparameters

param_dist = {
    'max_depth': randint(3, 10),
    'min_child_weight': randint(1, 10),
    'subsample': uniform(0.5, 1),
    'learning_rate': uniform(0.01, 0.1),
    'n_estimators': randint(100, 1000),
}
xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=2025)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=100, # Number of parameter settings that are sampled.
    scoring='accuracy',
    cv = TimeSeriesSplit(gap=10, n_splits=5),
    random_state=2025,
    n_jobs=-1 # Use all available cores
)

random_search.fit(X, y)

print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)


# ### Evaluate the Model
# We define define an XGBoost subset model with these hyperparameter values and evaluate on the full data set.

# In[18]:


# final model evaluation
finalModel = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=2025,
                          max_depth = 9, min_child_weight = 9, subsample = 0.50, learning_rate = 0.09, n_estimators = 273)

finalModel.fit(X, y)
ypred = finalModel.predict(X)
RocCurveDisplay.from_predictions(y, ypred)



# In[19]:


print("Confusion Matrix")
print(confusion_matrix(y, ypred))
disp = ConfusionMatrixDisplay.from_predictions(y, ypred,
                              display_labels =["Negative Return","Positive Return"],
                                              cmap = plt.cm.Blues)
plt.title("Confusion Matrix for Returns")
plt.xlabel("Predicted Return")
plt.ylabel("Actual Return")
plt.tight_layout()
plt.show()                              


# In[21]:


print(classification_report(y, ypred, labels = ["0","1"]))


# ## References
# 
# * [yfinance GitHub](https://github.com/ranaroussi/yfinance)
# * [yfinance Documentation](https://ranaroussi.github.io/yfinance/)
# * [Polars Online User Guide](https://docs.pola.rs/)
# * [Build Polars Database](https://www.pyquantnews.com/free-python-resources/build-stock-database-locally-with-polars)
# * [YouTube. Polars and Time Series: What It Can Do, and How to Overcome Any Limitation](https://www.youtube.com/watch?v=qz-zAHBz6Ks)
# * [Awesome Quant: Python for Quantiative Finance](https://wilsonfreitas.github.io/awesome-quant/)
# * [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
# * [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
# * [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
# * [Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html#grid-search)
# * [Metrics and Scoring](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
# * [Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)
# * [XGBoost documentation](https://xgboost.readthedocs.io/en/latest/index.html)
# * [XGBoost in Python documentation](https://xgboost-clone.readthedocs.io/en/latest/python/python_intro.html)
# * [Auto-Sklearn for AutoML in an Scikit-Learn Environment](https://www.automl.org/automl-for-x/tabular-data/auto-sklearn/).

# In[ ]:




