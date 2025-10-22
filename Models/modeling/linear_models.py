import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

class LinearRegressions:
    def __init__(self, df_train:pd.DataFrame, df_test:pd.DataFrame):
        self.df_train = df_train
        self.df_test = df_test
        self.features = df_train.columns
        self.models = {}

    def fit(self,features,degree=2,interaction=True,trig=False):
        #Initialize entries
        self.models.setdefault(degree,{})
        self.models[degree].setdefault(interaction,{})
        self.models[degree][interaction].setdefault(trig,{})

        X_train = self.df_train[features]
        y_train = self.df_train["Energy"]

        if trig: #Apply cosine to the zenith angle if true
            model = Pipeline(steps=[
                ('trig_apply', ColumnTransformer(transformers=[('cos',FunctionTransformer(np.cos),['Solar Zenith Angle'])],remainder='passthrough')),
                ('poly',PolynomialFeatures(degree=degree,interaction_only=interaction)),
                ('scale',StandardScaler()),
                ('regression', LinearRegression(n_jobs=-1))])
        else: #Otherwise skip
            model = Pipeline(steps=[
                    ('poly',PolynomialFeatures(degree=degree,interaction_only=interaction)),
                    ('scale',StandardScaler()),
                    ('regression', LinearRegression(n_jobs=-1))])
            
        model.fit(X_train,y_train)

        self.models[degree][interaction][trig][tuple(features)] = model

    def get_model(self, features, degree=2,interaction=True,trig=False):
        return self.models[degree][interaction][trig][tuple(features)]
    
    def predictions(self, features, degree=2,interaction=True,trig=False):
        model = self.get_model(features,degree,interaction,trig)
        df = self.df_test[["Energy"]].copy()
        df['Prediction'] = np.clip(model.predict(self.df_test[features]),0,None) #Clip negative values to 0 for minor improvement
        return df