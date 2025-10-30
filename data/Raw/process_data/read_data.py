import numpy as np
import pandas as pd
import cfgrib
import xarray as xr

def MergeData(frames:list):
    #Merge given datasets by matching (timezone aware) timestamps
    df = frames[0]
    for frame in frames[1:]:
        df = pd.merge(df,frame,left_index=True,right_index=True,how="inner")
    return df

def ImportClean(file:str,year:int):
    if 'NREL' in file:
        #Read in weather file: converting time columns to DateTime and selecting used columns (Solar Zenith Angle + Relative Humidity)
        df = pd.read_csv(file,header=2,usecols=[0,1,2,3,4,20,21])
        df['Date'] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute']])
        df = df.set_index('Date').tz_localize(tz='UTC',ambiguous='NaT',nonexistent='NaT') #Data given in UTC
        df['Solar Zenith Angle'] *= np.pi/180
        df = df.drop(columns=['Year','Month','Day','Hour','Minute'])
        df.index.names = ['Date']
        df = df.resample('h').agg('mean')
        return df
    
    elif 'power' in file:
        #Read in power file: converting time columns to DateTime, selecting power columns, and returning energy column
        df = pd.read_csv(file,index_col=0,usecols=[0,9,10])
        df = df.set_index(pd.to_datetime(df.index))
        df['Power'] = df.iloc[:,[0,1]].sum(axis=1)
        #Energy ~ Power*Time
        df['Energy'] = df['Power']/12
        df = df[['Energy']].dropna(axis=0)
        df.index.names = ['Date']
        df = df.iloc[df.index.year == year].tz_localize(tz='America/Denver',ambiguous='NaT',nonexistent='NaT') #Data given in local Mountain Time
        df = df.resample('h').agg('sum')
        return df

    elif 'environment' in file:
        #Read in environment file: converting time columns to DateTime and selecting used columns (Ambient Temperature+Wind Sensor)
        df = pd.read_csv(file,index_col=[0],usecols=[0,1])
        df = df.set_index(pd.to_datetime(df.index))
        df = df.rename(columns={df.columns[0]: "Ambient Temperature"})
        df['Ambient Temperature'] += 273
        df.dropna(inplace=True)
        df.index.names = ['Date']
        df = df[df.index.year == year].tz_localize(tz='America/Denver',ambiguous='NaT',nonexistent='NaT') #Data given in local Mountain Time
        df = df.resample('h').agg('mean')
        return df
    
    elif 'grib' in file:
        #Read in grib dataset: extract cloud cover and average over the nearby lat/long
        df = xr.open_dataset(file,engine='cfgrib',filter_by_keys={'shortName':'tcc'}).to_dataframe()
        df = df.groupby('time').mean()
        df = df[df.index.year == year][['tcc']]
        df = df.tz_localize(tz='UTC',ambiguous='NaT',nonexistent='NaT') #Data given in UTC
        return df