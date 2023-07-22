"""
This module contains the functions to explore the data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from numpy import sqrt, abs, round
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller

def plot_seasonal_analysis(_df, column, imgdir, filename):
    """
    This function plots the seasonal analysis of the data.
    :param df: the dataframe that contains the data to be processed.
    :param column: the column of the dataframe that contains the data to be processed.
    :param dir: the directory where the data is stored.
    :param filename: the name of the file.
    """
    plt.clf()
    result = sm.tsa.seasonal_decompose(_df[column], model="addictive", period=30)
    result.plot()
    plt.savefig(imgdir / f"{filename}.png")


def plot_correlation(_df,imgdir, filename):
    """
    This function plots the correlation matrix of the data.
    :param df: the dataframe that contains the data to be processed.
    :param dir: the directory where the data is stored.
    :param filename: the name of the file.
    """

    plt.clf()
    plt.figure(figsize = (15,15))
    corr = _df.corr()
    sns.heatmap(
        corr,
        cmap=sns.light_palette("seagreen"),
        annot=True,
        fmt=".2f",
    )
    plt.savefig(imgdir / f"{filename}.png")

def two_samples_t_test(df1, column1, df2, column2, significance_level=0.05):
    """
    Executes a two sample T-test with the statistic properties passed.
    :param df1: the first dataframe that contains the data to be processed.
    :param column1: the column of the first dataframe that contains the data to be processed.
    :param df2: the second dataframe that contains the data to be processed.
    :param column2: the column of the second dataframe that contains the data to be processed.
    :param significance_level: the significance level of the test.
    """
    mean1 = df1[column1].mean()
    mean2 = df2[column2].mean()
    sigma1= df1[column1].std()
    sigma2= df2[column2].std()
    size1= len(df1[column1])
    size2= len(df2[column2])
    # Compute Z-statistic
    overall_sigma = sqrt(sigma1 ** 2 / size1 + sigma2 ** 2 / size2)
    z_statistic = round((mean1 - mean2) / overall_sigma, 5)
    # Compute p-value from Z-statistic
    # Two tails -> H0:x1=x2 H1:x1!=x2
    print('\nTwo Tails test. H0 is x1=x2 and H1 is x1!=x2')
    p_value = round(2 * (1 - norm.cdf(abs(z_statistic))), 5)
    # Reject or not the Null hypothesis
    if p_value < significance_level:
        print(f'Statistic:{z_statistic} - P-value:{p_value} - Reject Null Hypothesis')
    else:
        print(f'Statistic:{z_statistic} - P-value:{p_value} - Do Not Reject Null Hypothesis')



def test_stationarity(timeseries, imgdir, filename):
    """
    This function tests the stationarity of the data.
    :param timeseries: the dataframe that contains the data to be processed.
    :param dir: the directory where the data is stored.
    :param filename: the name of the file.
    """
    #Determining rolling statistics
    rolmean = timeseries.rolling(4).mean() # around 4 weeks on each month
    rolstd = timeseries.rolling(4).std()
    #Plot rolling statistics:
    plt.clf()
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.savefig(imgdir / f"{filename}.png", bbox_inches='tight')
    #Perform Dickey-Fuller test:
    print ('\nResults of Dickey-Fuller Test of Stock Price:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
    index=['Test Statistic','p-value',
    '#Lags Used','Number of Observations Used']
    )
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    if dfoutput['p-value'] < 0.05:
        print('result : time series is stationary')
    else : print('result : time series is not stationary')
