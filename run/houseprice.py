#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import os

train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

# ======================= EDA =======================
print('='*20, 'Number of unique values in "object" columns', '='*20)
for col in train_df.columns:
    if train_df.dtypes[col] == 'object':
        print(col, train_df[col].unique().shape[0])
print('='*20, 'Min and max in numerical columns', '='*20)
for col in train_df.columns:
    if train_df.dtypes[col] != 'object':
        print(col, 'min:', train_df[col].min(), 'max:', train_df[col].max())
        
        
# search NA
nan_count = train_df.isna().sum()
nan_count = nan_count[nan_count > 0]
dict(nan_count.sort_values(ascending=False))

# plot numerical data
for col in train_df.columns:
    if train_df.dtypes[col] != 'object':
        plt.figure()
        plt.title(col)
        train_df[col].plot()
        plt.show()
        
# ======================= Feature =======================
train_df.drop(['Id'], axis=1, inplace=True)
test_df.drop(['Id'], axis=1, inplace=True)

# define GrLivArea >= 4500 are outliers and drop them
train_df = train_df[train_df.GrLivArea < 4500]
train_df.reset_index(drop=True, inplace=True)

# transform target
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
target = train_df['SalePrice']

# grouping, but now aborted due to poor performance
#def groupencoder(train_df, test_df, col, index, agg_type, new_col_name):
#    temp_df = train_df.groupby([col])[index].agg([agg_type]).reset_index().rename(
#        columns={agg_type: new_col_name})
#    print(temp_df)
#    temp_df.index = list(temp_df[col])
#    temp_df = temp_df[new_col_name].to_dict()
#    train_df[new_col_name] = train_df[col].map(temp_df)
#    test_df[new_col_name] = test_df[col].map(temp_df)

#groupencoder(train_df, test_df, 'MSSubClass', 'SalePrice', 'mean', "Class_Price_mean")
#groupencoder(train_df, test_df, 'YrSold', 'SalePrice', 'std', "YrSold_Price_std")
#groupencoder(train_df, test_df, 'MoSold', 'SalePrice', 'count', "MoSold_Price_mean")

train_features = train_df.drop(['SalePrice'], axis=1)
test_features = test_df
features = pd.concat([train_features, test_features]).reset_index(drop=True)

# plot some of the features for analysis
plt.figure()
plt.title('MSSubClass')
features['MSSubClass'].plot()
plt.show()

plt.figure()
plt.title('YrSold')
features['YrSold'].plot()
plt.show()

plt.figure()
plt.title('MoSold')
features['MoSold'].plot()
plt.show()

# cast to 'object' for encoding later
features['MSSubClass'] = features['MSSubClass'].astype('object')
features['YrSold'] = features['YrSold'].astype('object')
features['MoSold'] = features['MoSold'].astype('object')

print('='*20, 'Statistics of columns with NaN', '='*20)
nan_count = features.isna().sum()
nan_count = nan_count[nan_count > 0]
nan_count.sort_values(ascending=False, inplace=True)
print(dict(nan_count))

# fill NA with "None" or 0
for col in nan_count.keys():
    if features[col].dtype == 'object':
        features[col] = features[col].fillna("None")
        
for col in nan_count.keys():
    if features[col].dtype != 'object':
        print(col, features[col].min(), features[col].max())
        features[col] = features[col].fillna(features[col].mode()[0])

# encode category
def encode(train_df, col):
    set_value = train_df[col].unique()
    encode = {key:value for key, value in zip(set_value, range(set_value.shape[0]))}
    train_df[col] = train_df[col].map(encode)

for col in features.columns:
    if features.dtypes[col] == 'object':
        encode(features, col)

# add new features: "Total Square Feet" and "Total Porch Square Feet" to depict total building and porch area 
features['TotalSquareFeet'] = features['1stFlrSF'] + features['2ndFlrSF'] + features['TotalBsmtSF']
features['TotalPorchSquareFeet'] = (features['OpenPorchSF'] + features['EnclosedPorch'] + features['3SsnPorch'] + features['WoodDeckSF'] + features['ScreenPorch'] )

features = features.drop(['Utilities', 'Street', 'PoolQC','TotalBsmtSF','1stFlrSF','2ndFlrSF','OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF'], axis=1)

# print statistics of input features
print('='*20, 'Number of input features', '='*20)
print(features.shape[1])

#===================== Modelling ==========================
print('='*20, 'Modelling', '='*20)
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

models = []
models.append(make_pipeline(RobustScaler(), RidgeCV(alphas=np.linspace(14.5, 15.5, 11).tolist(), cv=kfolds)))
models.append(make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=np.linspace(5e-5, 8e-4, 9).tolist(), random_state=42, cv=kfolds)))
models.append(make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=np.linspace(1e-4, 7e-4, 7).tolist(), cv=kfolds, l1_ratio=np.linspace(0.8, 1.0, 21).tolist()))  )
models.append(GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state = 42))
models.append(LGBMRegressor(objective='regression',
                                       num_leaves=31,
                                       learning_rate=0.01,
                                       n_estimators=5000,
                                       max_bin=255,
                                       bagging_fraction=0.75,
                                       bagging_freq=5,
                                       bagging_seed=7,
                                       feature_fraction=0.9,
                                       feature_fraction_seed=42,
                                       verbose=-1,
                                       ))
models.append(XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006))

import sys
sys.setrecursionlimit(1000000)
for model in models:
    print(model)
    model.fit(features.iloc[:len(target), :].to_numpy(), target)
    
# blend models
def predict(data):
    blending_weights = [0.1, 0.1, 0.15, 0.15, 0.25, 0.25]
    output = np.zeros(data.shape[0])
    for index, model in enumerate(models):
        output += blending_weights[index] * model.predict(data)
    return output

test_data = features.iloc[len(target):, :].to_numpy()

submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission.iloc[:,1] = np.floor(np.expm1(predict(test_data)))

print(submission.head())

submission.to_csv("submission.csv", index=False)

