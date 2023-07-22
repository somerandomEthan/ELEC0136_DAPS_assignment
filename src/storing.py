"""
This module contains functions to store data in a MongoDB database.
"""

import pymongo

MONGODB_SERVER_ADDRESS = (
    "mongodb+srv://user:user@final.1np1kea.mongodb.net/?retryWrites=true&w=majority"
)

def format_data(_df, column):
    """
    Formats the data to be stored in the database.
    :param df: the dataframe containing the data.
    :param column: the column name of the data.
    :return: the formatted data.
    """
    _time = _df.index.tolist()
    _list = _df[column].tolist()
    _list_all=[]
    _num = len(_list)
    for i in range(_num):
        _list_all.append({
            'date' : _time[i],
            column : _list[i]
        })
    return _list_all


def create(_df, column, collection_name):
    """
    Creates a collection in the database and stores the data in it.

    :param df: the dataframe containing the data.
    :param column: the column name of the data.
    :param collection_name: the name of the collection.
    :return: the result of the operation.
    """
    # connect to your local mongo instance
    data = format_data(_df, column)
    server = pymongo.MongoClient(MONGODB_SERVER_ADDRESS)
    # grab the collection you want to push the data into
    database = server['Final']
    collection = database[collection_name]
    # push the data
    return collection.insert_many(data)
