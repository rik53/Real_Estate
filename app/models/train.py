import numpy as np
import pandas as pd
import dill
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from datetime import datetime

DATASET_PATH_TRAIN = 'train.csv'
df_train = pd.read_csv(DATASET_PATH_TRAIN, sep = ',')
X = df_train.drop('Price',axis=1)
y = pd.DataFrame(df_train['Price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#save test
X_test.to_csv("X_test.csv", index=None)
y_test.to_csv("y_test.csv", index=None)
#save train
X_train.to_csv("X_train.csv", index=None)
y_train.to_csv("y_train.csv", index=None)


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]
features = ['Square', 'Rooms', 'Social_3', 'Social_1']
target = 'Price'
final_transformers = list()

for col in features:
    transformer = Pipeline([
                ('selector', NumberSelector(key=col)),
                ('scaler', StandardScaler())
            ])
    final_transformers.append((col, transformer))
feats = FeatureUnion(final_transformers)
pipeline = Pipeline([
    ('feats',feats),
    ('model', GradientBoostingRegressor(criterion='mse',
                                     max_depth=5,
                                     min_samples_leaf=40,
                                     random_state=42,
                                     n_estimators=300)),
])

pipeline.fit(X_train[features], y_train)
with open("pipeline.dill", "wb") as f:
    dill.dump(pipeline, f)