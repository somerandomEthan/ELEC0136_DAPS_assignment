"""
This module contains the functions to make predictions with Facebook Prophet. The functions are used in the notebook
"""
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def prepare_data(data, target_feature):
    """
    Facebook Prophet requires the inputs to be in a particular format. The time-series to forecast must be called 'y'
    and the date column 'ds'

    :param data: training dataset.
    :param target_feature: name of the column associated to the time series.
    :return: the input data transformed.
    """
    new_data = data.copy()
    new_data.reset_index(inplace=True)
    new_data = new_data.rename({'index':'ds', '{}'.format(target_feature):'y'}, axis=1)
    
    return new_data

def create_basic_model_prediction(train, test):
    """
    Create a basic model with default parameters.

    :param train: training dataset.
    :param test: test dataset.
    :return: the forcast result.
    """
    model = Prophet(seasonality_mode='additive',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False)
    model.fit(train)
    future = model.make_future_dataframe(periods=len(test), freq='1D')
    forecast = model.predict(future)
    return forecast

def make_predictions_df(forecast, data_train, data_test): 
    """
    Function to convert the output Prophet dataframe to a datetime index and append the actual target values at the end

    :param forecast: the output of Prophet.
    :param data_train: the training dataset.
    :param data_test: the test dataset.
    :return: the forecast dataframe with the actual target values.
    """
    forecast.index = pd.to_datetime(forecast.ds)
    data_train.index = pd.to_datetime(data_train.ds)
    data_test.index = pd.to_datetime(data_test.ds)
    data = pd.concat([data_train, data_test], axis=0)
    forecast.loc[:,'y'] = data.loc[:,'y']
    
    return forecast

def plot_predictions(forecast, start_date, dir, filename):
    """
    Function to plot the predictions 

    :param forecast: the output of Prophet.
    :param start_date: the date from which to start the plot.
    :param dir: the directory where to save the plot.
    :param filename: the name of the file to save.
    :return: the prediction value y and yhat.
    """
    plt.clf()
    fig, ax = plt.subplots(figsize=(14, 8))
    
    train = forecast.loc[start_date:'2022-04-30',:]
    ax.plot(train.index, train.y, 'ko', markersize=3)
    ax.plot(train.index, train.yhat, color='steelblue', lw=0.5)
    ax.fill_between(train.index, train.yhat_lower, train.yhat_upper, color='steelblue', alpha=0.3)
    
    test = forecast.loc['2022-05-01':,:]
    ax.plot(test.index, test.y, 'ro', markersize=3)
    ax.plot(test.index, test.yhat, color='coral', lw=0.5)
    ax.fill_between(test.index, test.yhat_lower, test.yhat_upper, color='coral', alpha=0.3)
    ax.axvline(forecast.loc['2022-05-01', 'ds'], color='k', ls='--', alpha=0.7)

    ax.grid(ls=':', lw=0.5)
    
    fig.savefig(dir / f"{filename}.png", dpi=200)

    return test.y, test.yhat

def metrics(y, y_hat):
    """
    Computes metrics (MAPE, RMSE, CORR, R2, MAE, MPE, MSE) to evaluate models performance.

    :param y: the true values of the test dataset.
    :param y_hat: the predicted values.
    """
    # Compute errors
    errors = y - y_hat
    # Compute and print metrics
    mse = np.mean(errors ** 2)
    mae = np.mean(abs(errors))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(y_hat - y) / np.abs(y))
    mpe = np.mean((y_hat - y) / y)
    corr = np.corrcoef(y_hat, y)[0, 1]
    print('\nMatrices to evaluate the prediction result:\n',
          f'- MAPE: {mape:.3} \n',
          f'- RMSE: {rmse:.3f} \n',
          f'- CORR: {corr:.3f} \n',
          f'- MAE: {mae:.3f} \n',
          f'- MPE: {mpe:.3f} \n',
          f'- MSE: {mse:.3f}\n')

def create_multiregre_model(train, test, ws_all):
    """
    Create a basic model with default parameters.

    :param train: training dataset.
    :param test: test dataset.
    :param ws_all: the dataset with the wind speed.
    :return: the forcast result.
    """
    model = Prophet(seasonality_mode='additive',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False)
    model.add_regressor('Wind Speed', mode='multiplicative')
    model.fit(train)
    future = model.make_future_dataframe(periods=len(test), freq='1D')

    futures = pd.concat([future, ws_all.loc[:, ['Wind Speed']].reset_index()], axis=1)
    forecast = model.predict(futures)
    return forecast
