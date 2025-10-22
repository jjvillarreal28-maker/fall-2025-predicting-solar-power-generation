import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from itertools import product
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

class ForestRegressions:
    def __init__(self, df_train:pd.DataFrame, df_test:pd.DataFrame):
        self.df_train = df_train
        self.df_test = df_test
        self.features = df_train.columns
        self.models = {}

    def fit(self,feature_list:list, param_list=None):

        X_train = self.df_train[feature_list]
        y_train = self.df_train["Energy"]
        X_test = self.df_test[feature_list]
        y_test = self.df_test["Energy"]
        
        if param_list is None:
            param_list = {
            'n_estimators': [50, 100, 200],
            'max_depth': [2, 5, 10],
            'min_samples_split': [3, 4, 10],
            'min_samples_leaf': [3, 4, 10]}

        keys, values = zip(*param_list.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        best_rmse = np.inf
        best_model = None

        for params in param_combinations:
            model = RandomForestRegressor(**params,n_jobs=-1)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, preds)
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
        
        self.models[tuple(feature_list)] = best_model

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

        X_train = self.df_train[feature_list]
        y_train = self.df_train["Energy"]
        X_test = self.df_test[feature_list]
        y_test = self.df_test["Energy"]
        
        if param_list is None:
            param_list = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [None, 2, 5],
            'min_samples_split': [3, 4, 10],
            'min_samples_leaf': [3, 4, 10]}

        keys, values = zip(*param_list.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        best_rmse = np.inf
        best_model = None

        for params in param_combinations:
            model = ExtraTreesRegressor(**params,n_jobs=-1)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, preds)
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
        
        self.models[tuple(feature_list)] = best_model

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

        X_train = self.df_train[feature_list]
        y_train = self.df_train["Energy"]
        X_test = self.df_test[feature_list]
        y_test = self.df_test["Energy"]
        
        if param_list is None:
            param_list = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 2, 5]}

        keys, values = zip(*param_list.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        best_rmse = np.inf
        best_model = None
        best_parameters = None

        for params in param_combinations:
            model = XGBRegressor(**params,n_jobs=-1)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, preds)
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_parameters = params
        
        self.models[tuple(feature_list)] = best_model

    def get_model(self, feature_list:list):
        return self.models.get(tuple(feature_list))
    
    def get_parameters(self, feature_list:list):
        return self.models.get(tuple(feature_list))
    
    def predictions(self, feature_list:list):
        model = self.get_model(tuple(feature_list))
        df = self.df_test[["Energy"]].copy()
        df['Prediction'] = model.predict(self.df_test[feature_list])
        return df

