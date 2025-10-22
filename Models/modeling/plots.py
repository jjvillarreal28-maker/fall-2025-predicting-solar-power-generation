import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score

def plot_data(title:str,df:pd.DataFrame,day=None):
    if day is not None:#Optional day argument starts are chosen day
        start = pd.Timestamp(year=df.index.year[1],month=1,day=1).tz_localize('UTC')+pd.Timedelta(days=day-1)
        end = start+pd.Timedelta(days=14)
    else:#Otherwise randomly choose a day to begin plot
        days = [day for day in df.index.unique() if day.day_of_year < 350]
        start = np.random.choice(days)
        end = start+pd.Timedelta(days=14)

    plot_df = df.asfreq('h')#Add NaN rows so pyplot skips interpolation across jumps
    plot_df = plot_df.loc[start:end]

    #Metrics added to plot labeling
    rmse = root_mean_squared_error(df['Energy'],df['Prediction'])
    r2 = r2_score(df['Energy'],df['Prediction'])

    plt.figure(figsize=(20, 5))
    plt.plot(plot_df.index, plot_df['Energy'], label='Actual Energy', alpha=0.5)
    plt.plot(plot_df.index, plot_df['Prediction'], label='Predicted Energy', linestyle='--', alpha=0.9)
    plt.title(f"Actual vs Predicted Energy Over Time ({title})")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.text(0.94,1.05,f'RMSE: {rmse:.2f}\n r2: {r2:.4f}',transform=plt.gca().transAxes,bbox=dict(boxstyle="square", facecolor="white"))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_daily_data(title:str,df:pd.DataFrame):
    #Convert dataframe of hourly data to daily data of total energy
    df_daily = df.resample('d').agg('sum')

    #Metrics added to plot labeling
    rmse = root_mean_squared_error(df_daily['Energy'],df_daily['Prediction'])
    r2 = r2_score(df_daily['Energy'],df_daily['Prediction'])

    plt.figure(figsize=(20, 5))
    sns.scatterplot(df_daily, x=df_daily.index,y='Energy',label='Actual Energy')
    sns.scatterplot(df_daily, x=df_daily.index,y='Prediction',label='Predicted Energy')
    plt.title(f"Actual vs Predicted Energy Over Time Daily Total ({title})")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.text(0.94,1.05,f'RMSE: {rmse:.2f}\n r2: {r2:.4f}',transform=plt.gca().transAxes,bbox=dict(boxstyle="square", facecolor="white"))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()