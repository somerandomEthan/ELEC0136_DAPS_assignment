"""
A module acquires stock prices and auxiliary data
It contains methods to construct, perform, and post-process requests.
"""


import requests
import pandas as pd

from alpha_vantage.timeseries import TimeSeries
from meteostat import Point, Daily
from covid19dh import covid19


KEY = "34AX7I88JX9JJYM9"
COMPANY = "AAL"
CITIES = {
    "New York": (40.730610, -73.935242),
    "Los Angeles": (34.052235, -118.243683),
    "Chicago": (41.875562, -87.624210),
}


def get_stock_prices(key, start_date, end_date, filedir, filename):
    """
    Gets the stock prices data.

    :param key: the Alpha Vantage API key.
    :param start_date: the start date of the time range of required data.
    :param end_date: the end date of the time range of required data.
    :param filedir: the directory where the data will be stored.
    :param filename: the name of the file
    :return: the dataframe containing the data.
    """
    _ts = TimeSeries(key=key, output_format="pandas")
    _df, _ = _ts.get_daily_adjusted(symbol=COMPANY, outputsize="full")
    _df = _df.sort_index(ascending=True)
    _df = _df.rename(
        columns={
            "5. adjusted close": "Adjusted Close",
        }
    )
    start_date = start_date.date()
    end_date = end_date.date()
    _df = _df.loc[start_date:end_date]
    _df = pd.DataFrame(_df["Adjusted Close"])
    _df.to_csv(filedir / filename)
    return _df


def get_oil_data(key, start_date, end_date, filedir):
    """
    Gets the crude oil prices data.

    :param key: the Alpha Vantage API key.
    :param start_date: the start date of the time range of required data.
    :param end_date: the end date of the time range of required data.
    :param filedir: the directory where the data will be stored.
    :return: the dataframe containing the data.
    """
    _url = "https://www.alphavantage.co/query?function=WTI&interval=daily&apikey=" + key
    _r = requests.get(_url)
    _data = _r.json()
    data_oil = _data["data"]
    df_oil = pd.DataFrame(data_oil)
    df_oil = df_oil.set_index("date")
    df_oil = df_oil.sort_index(ascending=True)
    df_oil.index = pd.to_datetime(df_oil.index)
    df_oil = df_oil.loc[start_date:end_date]
    df_oil.to_csv(filedir / "Crude Oil Price.csv")

    return df_oil


def get_city_weather_data(cord, start_date, end_date):
    """
    Gets the weather data of a location by speicing its coordinates.

    :param cord: the coordinates of the location.
    :param start_date: the start date of the time range of required data.
    :param end_date: the end date of the time range of required data.
    :return: the data frame stored the weather data.
    """
    city = Point(*cord)
    data = Daily(city, start_date, end_date)
    data = data.fetch()
    return data


def get_all_city_weather(start_date, end_date, filedir):
    """
    Gets the weather data of all cities.

    :param start_date: the start date of the time range of required data.
    :param end_date: the end date of the time range of required data.
    :param filedir: the directory where the data will be stored.
    :return: the data frame stored the weather data.
    """
    # create an empty frame which has the length of number of cities
    for city, coord in CITIES.items():
        _data = get_city_weather_data(coord, start_date, end_date)
        _data = _data.rename(
            columns={
                "tavg": "Average Temperature",
                "wspd": "Wind Speed",
                "pres": "Pressure",
            }
        )
        _data = _data[["Average Temperature", "Wind Speed", "Pressure"]]
        _data.to_csv(filedir / f"{city}.csv")


def get_new_york_ws(start_date_test, end_date_test):
    """
    Gets the weather data of New York City.

    :param start_date_test: the start date of the time range of required data.
    :param end_date_test: the end date of the time range of required data.
    :return: the data frame stored the new york weather data.
    """
    _ny = (40.730610, -73.935242)
    _city = Point(*_ny)
    _data = Daily(_city, start_date_test, end_date_test)
    _data = _data.fetch()
    _data = _data.rename(columns={"wspd": "Wind Speed"})
    _data = pd.DataFrame(_data["Wind Speed"])
    return _data


def get_covid_data(start_date, end_date, filedir):
    """
    Gets the daily records about the ongoing pandemic of Covid-19.

    :param start_date: the start date of the time range of required data.
    :param end_date: the end date of the time range of required data.
    :param filedir: the directory where the data will be stored.
    :return the data frame stored the Covid data.
    """
    _df, _ = covid19("USA", start=start_date, end=end_date, verbose=False)
    _df = _df.set_index("date")
    _df.drop(_df.columns[0], axis=1, inplace=True)
    _df = pd.DataFrame(_df["confirmed"])
    _df.to_csv(filedir / "Covid.csv")
    return _df
