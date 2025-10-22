import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

class ForestRegressions:
    def __init__(self, df_train:pd.DataFrame, df_test:pd.DataFrame):
        self.df_train = df_train
        self.df_test = df_test
        self.features = df_train.columns
        self.models = {}

    def fit(self,feature_list:list, param_list=None):

        tscv = TimeSeriesSplit(n_splits=5)

        X_train = self.df_train[feature_list]
        y_train = self.df_train['Energy']
        
        if param_list is None:
            param_list = {
            'n_estimators': [50, 100, 200],
            'max_depth': [2, 5, 10],
            'min_samples_split': [3, 4, 10],
            'min_samples_leaf': [3, 4, 10]}

        search = GridSearchCV(RandomForestRegressor(),param_grid=param_list,cv=tscv,scoring='neg_root_mean_squared_error',n_jobs=-1)
        search.fit(X_train,y_train)
        
        self.models[tuple(feature_list)] = search.best_estimator_

    def get_model(self, feature_list:list):
        return self.models.get(tuple(feature_list))
    
    def predictions(self, feature_list:list):
        model = self.get_model(tuple(feature_list))
        df = self.df_test[["Energy"]].copy()
        df['Prediction'] = model.predict(self.df_test[feature_list])
        return df

class ExtraTreesRegressions:
    def __init__(self, df_train:pd.DataFrame, df_test:pd.DataFrame):
        self.df_train = df_train
        self.df_test = df_test
        self.features = df_train.columns
        self.models = {}

    def fit(self,feature_list:list, param_list=None):

        tscv = TimeSeriesSplit(n_splits=5)

        X_train = self.df_train[feature_list]
        y_train = self.df_train['Energy']
        
        if param_list is None:
            param_list = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [None, 2, 5],
            'min_samples_split': [3, 4, 10],
            'min_samples_leaf': [3, 4, 10]}

        search = GridSearchCV(ExtraTreesRegressor(),param_grid=param_list,cv=tscv,scoring='neg_root_mean_squared_error',n_jobs=-1)
        search.fit(X_train,y_train)
        
        self.models[tuple(feature_list)] = search.best_estimator_

    def get_model(self, feature_list:list):
        return self.models.get(tuple(feature_list))
    
    def predictions(self, feature_list:list):
        model = self.get_model(tuple(feature_list))
        df = self.df_test[["Energy"]].copy()
        df['Prediction'] = model.predict(self.df_test[feature_list])
        return df

class XGBoostRegressions:
    def __init__(self, df_train:pd.DataFrame, df_test:pd.DataFrame):
        self.df_train = df_train
        self.df_test = df_test
        self.features = df_train.columns
        self.models = {}

    def fit(self,feature_list:list, param_list=None):

        tscv = TimeSeriesSplit(n_splits=5)

        X_train = self.df_train[feature_list]
        y_train = self.df_train['Energy']
        
        if param_list is None:
            param_list = {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]}

        search = GridSearchCV(XGBRegressor(),param_grid=param_list,cv=tscv,scoring='neg_root_mean_squared_error',n_jobs=-1)
        search.fit(X_train,y_train)
        
        self.models[tuple(feature_list)] = search.best_estimator_


    def get_model(self, feature_list:list):
        return self.models.get(tuple(feature_list))
    
    def predictions(self, feature_list:list):
        model = self.get_model(tuple(feature_list))
        df = self.df_test[["Energy"]].copy()
        df['Prediction'] = model.predict(self.df_test[feature_list])
        return df

