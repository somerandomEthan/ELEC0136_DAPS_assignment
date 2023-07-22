""""
This module contains the functions that are used to process the data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def read_file_formatting(filedir, filename):
    """
    Returns the dataframe with the date as the index

    :param filedir: the directory where the data is stored.
    :param filename: the name of the file.
    :return: the dataframe with the date as the index.
    """
    filepath = filedir / filename
    _df = pd.read_csv(filepath)
    _df = _df.set_index("date")
    _df.index = pd.to_datetime(_df.index)
    return _df


def read_weather_formatting(filedir, filename):
    """
    Returns the dataframe with the time as the index

    :param filedir: the directory where the data is stored.
    :param filename: the name of the file.
    :return: the dataframe with the date as the index.
    """
    filepath = filedir / filename
    _df = pd.read_csv(filepath)
    _df = _df.set_index("time")
    _df.index = pd.to_datetime(_df.index)
    return _df


def fill_missing_dates(_df, start_date, end_date):
    """
    Returns the dataframe with continuous time index. Fill in all the missing dates if there is any.

    :param _df: the dataframe that contains the data to be processed.
    :param start_date: the starting date of the time range of  data.
    :param end_date: the ending date of the time range of data.
    :return: the dataframe with continuous time index.
    """
    new_index = pd.date_range(start_date, end_date)
    _df = _df.reindex(new_index)
    return _df


def plot_genral(_df, column, imgdir, filename):
    """
    Returns the plot of the data from specific column of a dataframe

    :param df: the dataframe that contains the data to be processed.
    :param column: the label of the column which contains the value that needs to be processed.
    :param dir: the parent directory of the plot.
    :param filename: the name of the plot file.
    """
    plt.clf()
    fig, _ax = plt.subplots(figsize=(16, 7))
    _ax.plot(pd.DataFrame(_df[column]))
    _ax.grid(ls="--", c="k", alpha=0.2)
    _ax.set_xlabel("Time")
    _ax.set_ylabel(f"{filename}")
    fig.savefig(imgdir / f"{filename}.png")


def boxplot_save(_df, column, label, filename, imgdir):
    """
    Returns the boxplot of the data from specific column of a dataframe

    :param df: the dataframe that contains the data to be processed.
    :param column: the label of the column which contains the value that needs to be processed.
    :param label: the label of x-axis.
    :param filename: the name of the boxplot file.
    :param dir: the parent directory of the boxplot.
    """
    plt.cla()
    sns.boxplot(_df[column], orient="v")
    plt.xlabel(label)
    plt.savefig(imgdir / f"{filename}.png")


def iqr_outlier_replace(_df, column):
    """
    Returns the location of the outlier, the outlier value and the time that outlier appears

    :param df: the dataframe that contains the data to be processed.
    :param column: the label of the column which contains the value that needs to be processed.
    :return: a dictionary containing the following keys and their respective values
    """
    _value = _df[column].values
    # _value = np.array(_value[:-1])  # change the list into numpy array
    # _time = np.array(_time[:-1])  # change the list into numpy array
    _q1 = np.quantile(_value, 0.25)
    _q3 = np.quantile(_value, 0.75)
    _iqr = _q3 - _q1
    _minimum = _q1 - 1.5 * _iqr
    _maximum = _q3 + 1.5 * _iqr
    _outlier = np.where((_value < _minimum) | (_value > _maximum))
    _value[_outlier] = np.nan
    _df[column] = _value
    return _outlier, _df


def time_fill(_df, start_date, end_date):
    """
    Returns the dataframe with continuous time index. Fill in all the missing dates if there is any.

    :param df: the dataframe that contains the data to be processed.
    :param start_date: the starting date of the time range of data.
    :param end_date: the ending date of the time range of data.
    :return: the dataframe with continuous time index.
    """
    new_index = pd.date_range(start_date, end_date)
    _df = _df.reindex(new_index)
    return _df


def imputation(_df, column):
    """
    Returns the dataframe with the missing value filled in.

    :param df: the dataframe that contains the data to be processed.
    :param column: the label of the column which contains the value that needs to be processed.
    :return: the dataframe with the missing value filled in.
    """
    _df[column] = _df[column].interpolate(method="linear")
    imputer = KNNImputer(n_neighbors=2)
    _df[column] = imputer.fit_transform(pd.DataFrame(_df[column]))

    return _df


def pca(_df):
    """
    Prints the explained variance ratio of the data from specific column of a dataframe using PCA
    :param df: the dataframe that contains the data to be processed.
    """
    _x = _df.values
    x_std = StandardScaler().fit_transform(_x)
    _pca = PCA()
    _y = _pca.fit_transform(x_std)
    explained_variance = np.var(_y, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    print(explained_variance_ratio)


def kernel_pca(_df, _type):
    """
    Prints the explained variance ratio of using kernel PCA
    :param df: the dataframe that contains the data to be processed.
    :param type: the type of kernel to be used.
    """
    _x = _df.values
    x_std = StandardScaler().fit_transform(_x)
    kpca = KernelPCA(kernel=_type)
    y_k = kpca.fit_transform(x_std)
    explained_variance = np.var(y_k, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    print(explained_variance_ratio)
