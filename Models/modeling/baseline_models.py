import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

def MeanBaseline(df:pd.DataFrame,df_t:pd.DataFrame=None):
    base_m = DummyRegressor(strategy='mean')
    base_x = df[['Solar Zenith Angle']]
    base_y = df[['Energy']].copy()
    base_m.fit(base_x,base_y)
    if df_t is not None:
        base_x_t = df_t[['Solar Zenith Angle']]
        base_y_t = df_t[['Energy']].copy()
        base_y_t['Prediction'] = base_m.predict(base_x_t)
        return base_y_t
    else:
        base_y['Prediction'] = base_m.predict(base_x)
        return base_y

def PreviousYearsBaseline(df:pd.DataFrame,df_t:pd.DataFrame):
    df_p = df_t[['Energy']].copy()
    df_pred = df.groupby([df.index.month,df.index.day,df.index.hour]).agg('mean')['Energy']
    df_pred.index = pd.to_datetime(['2024-{0:02d}-{1:02d} {2:02d}:00:00+00:00'.format(m,d,h) for m,d,h in df_pred.index])
    df_p['Prediction'] = df_pred
    return df_p.dropna()