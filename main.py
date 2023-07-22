"""
This is the main module responsible for solving the tasks.
To solve each task just run `python main.py`.
"""


from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

import src.acquiring
import src.storing
import src.processing
import src.exploration
import src.inference


KEY = "34AX7I88JX9JJYM9"

basedir = Path.cwd()
filedir = basedir / "Data File"
imgdir = basedir / "Images"
imgdir.mkdir(parents=True, exist_ok=True)
filedir.mkdir(parents=True, exist_ok=True)


start_date = datetime(2017, 4, 1)
end_date = datetime(2022, 4, 30)
start_date_test = datetime(2022, 5, 1)
end_date_test = datetime(2022, 5, 31)


def acquire_data():
    """
    Acquire the data from the API and save it in the Data File folder.

    """
    src.acquiring.get_stock_prices(KEY, start_date, end_date, filedir, "AAL.csv")
    src.acquiring.get_oil_data(KEY, start_date, end_date, filedir)
    src.acquiring.get_all_city_weather(start_date, end_date, filedir)
    src.acquiring.get_covid_data(start_date, end_date, filedir)


def storing():
    """
    Store the data in the database.

    """
    # get the stock price data frame
    stock_price = src.processing.read_file_formatting(filedir, "AAL.csv")
    src.storing.create(stock_price, "Adjusted Close", "AAL Adjusted Close")
    # get the new york weather data frame
    new_york_weather = src.processing.read_weather_formatting(filedir, "New York.csv")
    src.storing.create(
        new_york_weather, "Average Temperature", "New York Average Temperature"
    )
    src.storing.create(new_york_weather, "Wind Speed", "New York Wind Speed")
    src.storing.create(new_york_weather, "Pressure", "New York Pressure")
    # get the los angeles weather data frame
    los_angeles_weather = src.processing.read_weather_formatting(
        filedir, "Los Angeles.csv"
    )
    src.storing.create(
        los_angeles_weather, "Average Temperature", "Los Angeles Average Temperature"
    )
    src.storing.create(los_angeles_weather, "Wind Speed", "Los Angeles Wind Speed")
    src.storing.create(los_angeles_weather, "Pressure", "Los Angeles Pressure")
    # get the chicago weather data frame
    chicago_weather = src.processing.read_weather_formatting(filedir, "Chicago.csv")
    src.storing.create(
        chicago_weather, "Average Temperature", "Chicago Average Temperature"
    )
    src.storing.create(chicago_weather, "Wind Speed", "Chicago Wind Speed")
    src.storing.create(chicago_weather, "Pressure", "Chicago Pressure")
    # get the crude oil price
    oil_price = src.processing.read_file_formatting(filedir, "Crude Oil Price.csv")
    src.storing.create(oil_price, "value", "Crude Oil Price")
    # get the covid data
    covid_data = src.processing.read_file_formatting(filedir, "covid.csv")
    src.storing.create(covid_data, "confirmed", "Covid Confirmed")


def processing():
    """
    Process the data
    """
    # get the stock price data frame and boxplot
    stock_price = src.processing.read_file_formatting(filedir, "AAL.csv")
    src.processing.boxplot_save(
        _df=stock_price,
        column="Adjusted Close",
        label="Adjusted Close ($ per share)",
        filename="Boxplot of the Stock Price",
        imgdir=imgdir,
    )
    # get the new york weather data frame and boxplot
    new_york_weather = src.processing.read_weather_formatting(filedir, "New York.csv")
    src.processing.boxplot_save(
        _df=new_york_weather,
        column="Average Temperature",
        label=r"Average Temperature($^\circ$C)",
        filename="Boxplot of the Average Temperature of New York",
        imgdir=imgdir,
    )
    src.processing.boxplot_save(
        _df=new_york_weather,
        column="Wind Speed",
        label=r"Wind Speed(km/h)",
        filename="Boxplot of the Wind Speed of New York",
        imgdir=imgdir,
    )
    src.processing.boxplot_save(
        _df=new_york_weather,
        column="Pressure",
        label=r"Average Temperature($^\circ$C)",
        filename="Boxplot of the Pressure of New York",
        imgdir=imgdir,
    )
    # get the los angeles weather data frame and boxplot
    los_angeles_weather = src.processing.read_weather_formatting(
        filedir, "Los Angeles.csv"
    )
    src.processing.boxplot_save(
        _df=los_angeles_weather,
        column="Average Temperature",
        label=r"Average Temperature($^\circ$C)",
        filename="Boxplot of the Average Temperature of Los Angeles",
        imgdir=imgdir,
    )
    src.processing.boxplot_save(
        _df=los_angeles_weather,
        column="Wind Speed",
        label=r"Wind Speed(km/h)",
        filename="Boxplot of the Wind Speed of Los Angeles",
        imgdir=imgdir,
    )
    src.processing.boxplot_save(
        _df=los_angeles_weather,
        column="Pressure",
        label=r"Average Temperature($^\circ$C)",
        filename="Boxplot of the Pressure of Los Angeles",
        imgdir=imgdir,
    )
    # get the chicago weather data frame and boxplot
    chicago_weather = src.processing.read_weather_formatting(filedir, "Chicago.csv")
    src.processing.boxplot_save(
        _df=chicago_weather,
        column="Average Temperature",
        label=r"Average Temperature($^\circ$C)",
        filename="Boxplot of the Average Temperature of Chicago",
        imgdir=imgdir,
    )
    src.processing.boxplot_save(
        _df=chicago_weather,
        column="Wind Speed",
        label=r"Wind Speed(km/h)",
        filename="Boxplot of the Wind Speed of Chicago",
        imgdir=imgdir,
    )
    src.processing.boxplot_save(
        _df=chicago_weather,
        column="Pressure",
        label=r"Average Temperature($^\circ$C)",
        filename="Boxplot of the Pressure of Chicago",
        imgdir=imgdir,
    )
    # get the crude oil price and boxplot
    oil_price = src.processing.read_file_formatting(filedir, "Crude Oil Price.csv")
    oil_price = oil_price.replace(".", 0)
    src.processing.boxplot_save(
        _df=oil_price,
        column="value",
        label="Crude Oil Price ($ )",
        filename="Boxplot of the Crude Oil Price",
        imgdir=imgdir,
    )
    # get the active covid confirmed and boxplot
    covid = src.processing.read_file_formatting(filedir, "Covid.csv")
    covid["Active"] = np.insert(np.diff(covid.values.flatten()), 0, 0)
    covid.drop(covid.columns[0], axis=1, inplace=True)
    src.processing.boxplot_save(
        _df=covid,
        column="Active",
        label="Active Covid confirmed cases",
        filename="Boxplot of the Active Covid confirmed cases",
        imgdir=imgdir,
    )
    # give the missing index with nan and fill in the value with KNN imputer for crude oil price
    oil_price = src.processing.time_fill(oil_price, start_date, end_date)
    oil_price = src.processing.imputation(oil_price, "value")
    src.processing.plot_genral(
        oil_price, "value", imgdir, "Overview of crude oil price"
    )
    print("\nCrude oil price summary")
    print(oil_price.describe())

    # give the missing index with nan and fill in the value with KNN imputer for stock price
    stock_price = src.processing.time_fill(stock_price, start_date, end_date)
    stock_price = src.processing.imputation(stock_price, "Adjusted Close")
    src.processing.plot_genral(
        stock_price, "Adjusted Close", imgdir, "Overview of stock price"
    )
    print("\nStock price summary")
    print(stock_price.describe())
    # print(stock_price.isnull().sum())

    # fill in the missing index and pick out the outlier and fill in nan for new york weather
    _, new_york_weather = src.processing.iqr_outlier_replace(
        new_york_weather, "Wind Speed"
    )
    _, new_york_weather = src.processing.iqr_outlier_replace(
        new_york_weather, "Pressure"
    )
    new_york_weather = src.processing.time_fill(new_york_weather, start_date, end_date)
    # fill in the nan with KNN imputer
    new_york_weather = src.processing.imputation(
        new_york_weather, "Average Temperature"
    )
    new_york_weather = src.processing.imputation(new_york_weather, "Wind Speed")
    new_york_weather = src.processing.imputation(new_york_weather, "Pressure")
    src.processing.plot_genral(
        new_york_weather,
        "Average Temperature",
        imgdir,
        "Overview of Average Temperature of New York",
    )
    src.processing.plot_genral(
        new_york_weather, "Wind Speed", imgdir, "Overview of Wind Speed of New York"
    )
    src.processing.plot_genral(
        new_york_weather, "Pressure", imgdir, "Overview of Pressure of New York"
    )
    print("\nNew York weather summary")
    print(new_york_weather.describe())
    # print(new_york_weather.isnull().sum())

    # fill in the missing index and pick out the outlier and fill in nan for los angeles weather
    _, los_angeles_weather = src.processing.iqr_outlier_replace(
        los_angeles_weather, "Average Temperature"
    )
    _, los_angeles_weather = src.processing.iqr_outlier_replace(
        los_angeles_weather, "Wind Speed"
    )
    _, los_angeles_weather = src.processing.iqr_outlier_replace(
        los_angeles_weather, "Pressure"
    )
    los_angeles_weather = src.processing.time_fill(
        los_angeles_weather, start_date, end_date
    )
    # fill in the nan with KNN imputer
    los_angeles_weather = src.processing.imputation(
        los_angeles_weather, "Average Temperature"
    )
    los_angeles_weather = src.processing.imputation(los_angeles_weather, "Wind Speed")
    los_angeles_weather = src.processing.imputation(los_angeles_weather, "Pressure")
    src.processing.plot_genral(
        los_angeles_weather,
        "Average Temperature",
        imgdir,
        "Overview of Average Temperature of Los Angeles",
    )
    src.processing.plot_genral(
        los_angeles_weather,
        "Wind Speed",
        imgdir,
        "Overview of Wind Speed of Los Angeles",
    )
    src.processing.plot_genral(
        los_angeles_weather, "Pressure", imgdir, "Overview of Pressure of Los Angeles"
    )
    print("\nLos Angeles weather summary")
    print(los_angeles_weather.describe())
    # print(los_angeles_weather.isnull().sum())

    # fill in the missing index and pick out the outlier and fill in nan for chicago weather
    _, chicago_weather = src.processing.iqr_outlier_replace(
        chicago_weather, "Average Temperature"
    )
    _, chicago_weather = src.processing.iqr_outlier_replace(
        chicago_weather, "Wind Speed"
    )
    _, chicago_weather = src.processing.iqr_outlier_replace(chicago_weather, "Pressure")
    chicago_weather = src.processing.time_fill(chicago_weather, start_date, end_date)
    # fill in the nan with KNN imputer
    chicago_weather = src.processing.imputation(chicago_weather, "Average Temperature")
    chicago_weather = src.processing.imputation(chicago_weather, "Wind Speed")
    chicago_weather = src.processing.imputation(chicago_weather, "Pressure")
    src.processing.plot_genral(
        chicago_weather,
        "Average Temperature",
        imgdir,
        "Overview of Average Temperature of Chicago",
    )
    src.processing.plot_genral(
        chicago_weather, "Wind Speed", imgdir, "Overview of Wind Speed of Chicago"
    )
    src.processing.plot_genral(
        chicago_weather, "Pressure", imgdir, "Overview of Pressure of Chicago"
    )
    print("\nChicago weather summary")
    print(chicago_weather.describe())
    # print(chicago_weather.isnull().sum())

    # give the missing index and fill in the nan for covid data
    covid = src.processing.time_fill(covid, start_date, end_date)
    _, covid = src.processing.iqr_outlier_replace(covid, "Active")
    covid = src.processing.imputation(covid, "Active")
    src.processing.plot_genral(
        covid, "Active", imgdir, "Overview of Covid active individuals"
    )
    # print(covid.isnull().sum())
    print("\nCovid data summary")
    print(covid.describe())

    # merge the weather data
    weather = pd.merge(
        new_york_weather,
        los_angeles_weather,
        "outer",
        left_index=True,
        right_index=True,
        suffixes=("_NY", "_LA"),
    ).merge(chicago_weather, "outer", left_index=True, right_index=True)
    weather = weather.sort_index(ascending=True)
    # PCA and the explained variance
    print("\n Weather data PCA and the explained variance")
    src.processing.pca(weather)
    print("\n Weather data Kernek PCA with cosine and the explained variance")
    src.processing.kernel_pca(weather, "cosine")
    return (
        stock_price,
        covid,
        new_york_weather,
        weather,
        oil_price,
    )


def exploration(
    stock_price,
    covid,
    new_york_weather,
    weather,
    oil_price,
):
    """
    Exploratory data analysis
    """
    # trend, seasonality and random noise
    src.exploration.plot_seasonal_analysis(
        stock_price, "Adjusted Close", imgdir, "Stock Price Analysis"
    )
    src.exploration.plot_seasonal_analysis(
        new_york_weather,
        "Average Temperature",
        imgdir,
        "New York Average Temperature Analysis",
    )
    src.exploration.plot_seasonal_analysis(
        new_york_weather, "Wind Speed", imgdir, "New York Wind Speed Analysis"
    )
    src.exploration.plot_seasonal_analysis(
        new_york_weather, "Pressure", imgdir, "New York Wind Speed Pressure Analysis"
    )

    all_data = (
        pd.merge(stock_price, covid, "outer", left_index=True, right_index=True)
        .merge(weather, "outer", left_index=True, right_index=True)
        .merge(oil_price, "outer", left_index=True, right_index=True)
    )
    src.exploration.plot_correlation(all_data, imgdir, "Correlation of all data")

    src.exploration.test_stationarity(
        stock_price, imgdir, " Stcok Price Rolling Mean & Standard Deviation"
    )

    src.exploration.two_samples_t_test(stock_price, "Adjusted Close", covid, "Active")
    return stock_price, covid, weather, oil_price


def inference(stock_price, new_york_weather):
    """
    Inference
    """
    # create the model and make the prediction
    stock_test = src.acquiring.get_stock_prices(
        KEY, start_date_test, end_date_test, filedir, "AAL_test.csv"
    )
    # fill in the missing dates to meet the requirement of the model
    stock_test = src.processing.fill_missing_dates(
        stock_test, start_date_test, end_date_test
    )
    stock_test = src.processing.imputation(stock_test, "Adjusted Close")
    stock_tested = src.inference.prepare_data(stock_test, "Adjusted Close")
    _stock = src.inference.prepare_data(stock_price, "Adjusted Close")
    forecast = src.inference.create_basic_model_prediction(_stock, stock_tested)
    result = src.inference.make_predictions_df(forecast, _stock, stock_tested)
    # plot the result form 2022 Jan 1
    test_y, test_yhat = src.inference.plot_predictions(
        result, "2022-01-01", imgdir, "Stock Price Prediction"
    )
    # get the metrics for the prediction
    src.inference.metrics(test_y, test_yhat)
    # get the prediction with wind speed of LA
    ws_train = pd.DataFrame(new_york_weather["Wind Speed"])
    ws_train = ws_train.rename({"index": "ds"}, axis=1)
    ws_test = src.acquiring.get_new_york_ws(start_date_test, end_date_test)
    ws_test = src.processing.fill_missing_dates(ws_test, start_date_test, end_date_test)
    ws_test = ws_test.rename({"index": "ds"}, axis=1)
    stock_ws_train = pd.merge(
        stock_price, ws_train, "outer", left_index=True, right_index=True
    )
    stock_ws_test = pd.merge(
        stock_test, ws_test, "outer", left_index=True, right_index=True
    )
    stock_ws_trained = src.inference.prepare_data(stock_ws_train, "Adjusted Close")
    stock_ws_tested = src.inference.prepare_data(stock_ws_test, "Adjusted Close")
    ws_all = pd.concat([ws_train, ws_test])
    ws_all = ws_all.rename({"index": "ds"}, axis=1)
    aux_forecast = src.inference.create_multiregre_model(
        stock_ws_trained, stock_ws_tested, ws_all
    )
    aux_result = src.inference.make_predictions_df(
        aux_forecast, stock_ws_trained, stock_ws_tested
    )
    aux_test_y, aux_test_yhat = src.inference.plot_predictions(
        aux_result, "2022-01-01", imgdir, "Stock Price Prediction with Wind Speed"
    )
    src.inference.metrics(aux_test_y, aux_test_yhat)


def main():
    """
    Main function
    """
    acquire_data()
    storing()
    (
        stock_price,
        covid,
        new_york_weather,
        weather,
        oil_price,
    ) = processing()
    stock_price, covid, weather, oil_price = exploration(
        stock_price,
        covid,
        new_york_weather,
        weather,
        oil_price,
    )
    inference(stock_price, new_york_weather)


if __name__ == "__main__":
    main()
