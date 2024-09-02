##########################################
# This file contains utility functions   #
##########################################

#%% Import libraries

import pickle
import os
import numpy as np
import pandas as pd
import ast   # Abstract Syntax Trees - used in the "converter" for reading .csv


#%% Store dictionaries and load dictionaries
def store_dict_output_pickle(data, output_dir, filename):
    """Save dictionary of DataFrames to a Pickle file."""
    with open(os.path.join(output_dir, f'{filename}_dict'), 'wb') as file:
        pickle.dump(data, file)


def load_dfs_from_pickle(output_dir, filename):
    """Load dictionary of DataFrames from a Pickle file."""

    with open(os.path.join(output_dir, f'{filename}_dict'), 'rb') as file:
        data = pickle.load(file)
    return data


#%% Read local curves from csv. The csv stores the numpy arrays as strings,
# so must use a "converter" to convert string to np array
# https://stackoverflow.com/questions/42755214/how-to-keep-numpy-array-when-saving-pandas-dataframe-to-csv

def from_np_array(array_string):
    """Read local curves from csv. The csv stores the numpy arrays as strings, so must use a "converter" to convert string to np array."""
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))



#%% Export fns
def export_to_csv(dataframes, names, output_dir, infile):
    for df, name in zip(dataframes, names):
        df.to_csv(f'{output_dir}/{infile}_{name}.csv', index=True)


def export_to_pickle(dataframes, names, output_dir, infile):
    for df, name in zip(dataframes, names):
        output_path = os.path.join(output_dir, f'{infile}_{name}.pkl')
        df.to_pickle(output_path)


def multiply_numeric_entries_by_100(df):
    """Multiply numeric values in a df by 100."""

    def multiply_if_numeric(x):
        return x * 100 if pd.api.types.is_numeric_dtype(type(x)) else x
    
    return df.applymap(multiply_if_numeric)


def filter_by_quotedate(df, start_date, end_date):

    filtered_df = df.loc[(df.index.get_level_values('quotedate') >= start_date) & 
                         (df.index.get_level_values('quotedate') <= end_date)]
    return filtered_df
