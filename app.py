# coding=utf-8
import json
import os

import numpy as np

from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split

from catboost.datasets import titanic

# get training data
train, test = titanic()

# remove nans
train, test = train.fillna(-999), test.fillna(-999)

# split into train and test
X, y = train.drop(['PassengerId', 'Survived'], axis=1), train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# define categorical features
cat_indices = np.where(X_train.dtypes != float)[0]

# define model params
params = {'iterations': 1000,
          'depth': 2,
          'loss_function': 'Logloss',
          'eval_metric': 'F1',
          'use_best_model': True,
          'verbose': True}

# define and train model
model = CatBoostClassifier(**params)
model.fit(X_train, y_train, cat_features=cat_indices, eval_set=(X_test, y_test))

# evaluate the quality of the model with 10-fold cross validation
cv_data = Pool(data=X, label=y, cat_features=cat_indices)
scores = cv(cv_data, model.get_params(), fold_count=10)

# print f1 metric
f1_metric = np.max(scores['test-F1-mean'])
print(f"f1 score: {round(f1_metric, 3)}")

# persist model
model.save_model('model')

# write metrics
if not os.path.exists('metrics'):
    os.mkdir('metrics')
with open('metrics/f1.metric', 'w+') as f:
    json.dump(f1_metric, f)
