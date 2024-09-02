import pandas as pd
import numpy as np
import sys
import xlsxwriter
from datetime import datetime

sys.path.append('../curve_utils/src/package')
sys.path.append('../curve_utils/src/development')
import DateFunctions_1 as dates
from crsp_data_processing import clean_crsp



# Load the data
# file_path = '1930to1935.csv'
# bonddata = pd.read_csv(file_path)
# bonddata.head()

# Check type
# bonddata['MCALDT'].dtype


# %% Clean up dataset

def clean_raw_crsp(filepath):
    """Clean raw bond data, without throwing out bonds."""

    bonddata = pd.read_csv(filepath)

    # Add callflag
    non_call = pd.isnull(bonddata['TFCALDT'])     # TRUE when call is nan
    bonddata.loc[non_call , 'TFCALDT'] = 0.0      # Replace nan s with 0s
    bonddata['callflag'] = np.where(bonddata['TFCALDT'] == 0, 0, 1)
    
    # Convert to datetime
    bonddata['MCALDT'] = bonddata['MCALDT'].astype('Int64')
    bonddata['TMATDT'] = bonddata['TMATDT'].astype('Int64')

    # Discard consol bonds
    bonddata = bonddata[bonddata['TMATDT'] != 20990401]

    bonddata['quote_date'] = pd.to_datetime(bonddata['MCALDT'].astype(str), format='%Y%m%d')
    bonddata['maturity_date'] = pd.to_datetime(bonddata['TMATDT'].astype(str), format='%Y%m%d')

    # Calculate days to maturity
    bonddata['days_to_maturity'] = (bonddata['maturity_date'] -
                                    bonddata['quote_date']).dt.days
    
    return bonddata


# Define maturity buckets
# Proposed base breaks: 1wk, 1mth, 3mth, 6mth, 1yr, 2, 3, 5, 7, 10, 20, 30
maturity_buckets = {
    '7days': (0, 7),
    '7days-1mon': (7, 365.25/12),
    '1mon-3mons': (365.25/12, 365.25/4),
    '3mons-6mons': (365.25/4, 365.25/2),
    '6mons-1YR': (365.25/2, 365.25),
    '1YR-2YR': (365.25, 2*365.25),
    '2YR-3YR': (2*365.25, 3*365.25),
    '3YR-5YR': (3*365.25, 5*365.25),
    '5YR-7YR': (5*365.25, 7*365.25),
    '7YR-10YR': (7*365.25, 10*365.25),
    '10YR-20YR': (10*365.25, 20*365.25),
    '20YR-30YR': (20*365.25, 30*365.25)
    #'20YR and beyond': (20*365.25, float('inf'))
}

maturity_bucket_name = [
    '7days',
    '7days-1mon',
    '1mon-3mons',
    '3mons-6mons',
    '6mons-1YR',
    '1YR-2YR', 
    '2YR-3YR', 
    '3YR-5YR', 
    '5YR-7YR', 
    '7YR-10YR',
    '10YR-20YR',
    '20YR-30YR'
]

# '0-35 days',
# '36-50 days',
# '50days-6months',
# '6mons-1year',
# '1YR-3yr',
# '3YR-5YR',
# '5YR-7YR',
# '7YR-10YR',
# '10YR-15YR',
# '15YR-20YR',
# '20YR and beyond'


# Classify into maturity buckets
def classify_into_buckets(days):
    for name, (lower_bound, upper_bound) in maturity_buckets.items():
        if lower_bound < days <= upper_bound:
            return name
    return 'Out of range'



# %% bond statistics per quote date by taxability

def generate_n_table_breaks(data):
    """Pivot table for number of bonds per basic break."""
    num_bonds_pivot = data.pivot_table(index='quote_date', columns='maturity_bucket',
                                           aggfunc='size', fill_value=0)
    num_bonds_pivot = num_bonds_pivot.sort_index()
    num_bonds_pivot = num_bonds_pivot.reindex(columns=maturity_bucket_name, fill_value=0)
    # num_bonds_pivot.loc[:,'Row_Total']= num_bonds_pivot.sum(numeric_only=True, axis=1)
    return num_bonds_pivot


def generate_breaks_df(df):

    df['maturity_bucket'] = df['days_to_maturity'].apply(
        classify_into_buckets)
    num_bonddata = generate_n_table_breaks(df)

    # Define breaks for each quote date
    base_breaks = [round(7/365.25,4), 0.0833, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    maturity_bucket_name = [
        '7days', '7days-1mon', '1mon-3mons', '6mons-1YR', '1YR-2YR', 
        '2YR-3YR', '3YR-5YR', '5YR-7YR', '7YR-10YR', '10YR-20YR', '20YR-30YR'
    ]

    breaks_df = []

    for date, row in num_bonddata.iterrows():
        # Collect non-zero maturity buckets
        non_zero_breaks = [base_breaks[i] for i, bucket in enumerate(maturity_bucket_name) if row[bucket] > 0]
        breaks_df.append([date, non_zero_breaks])

    breaks_df = pd.DataFrame(breaks_df, columns=['quote_date', 'breaks'])
    breaks_df['quote_date'] = pd.to_datetime(breaks_df['quote_date'])
    breaks_df['quote_date'] = breaks_df['quote_date'].dt.strftime('%Y%m%d').astype(int)

    return breaks_df



def generate_n_table(data, tax, billflag):
    """Pivot table for number of bonds."""
    data_tax = data[data['ITAX'] == tax]
    if billflag == True:
        data_tax = data[data['ITYPE'] == 4]
    num_bonds_pivot = data_tax.pivot_table(index='quote_date', columns='maturity_bucket',
                                           aggfunc='size', fill_value=0)
    num_bonds_pivot = num_bonds_pivot.sort_index()
    num_bonds_pivot = num_bonds_pivot.reindex(columns=maturity_bucket_name, fill_value=0)
    num_bonds_pivot.loc[:,'Row_Total']= num_bonds_pivot.sum(numeric_only=True, axis=1)
    return num_bonds_pivot


def generate_notional_table(data, tax, billflag):
    """Pivot table for notional amounts."""
    data_tax = data[data['ITAX'] == tax]
    if billflag == True:
        data_tax = data[data['ITYPE'] == 4]
    notional_bonds_pivot = data_tax.pivot_table(index='quote_date', columns='maturity_bucket',
                                                values='TMTOTOUT', aggfunc='sum', fill_value=0)
    notional_bonds_pivot = notional_bonds_pivot.sort_index()
    notional_bonds_pivot = notional_bonds_pivot.reindex(columns=maturity_bucket_name, fill_value=0)
    notional_bonds_pivot.loc[:,'Row_Total']= notional_bonds_pivot.sum(numeric_only=True, axis=1)
    return notional_bonds_pivot


def generate_proportional_notional_table(data, tax, billflag):
    """Pivot table for proportional notional amounts among total notional."""
    data_tax = data[data['ITAX'] == tax]
    if billflag == True:
        data_tax = data[data['ITYPE'] == 4]
    
    # Pivot table for notional amounts by maturity bucket
    notional_bonds_pivot = data_tax.pivot_table(index='quote_date', columns='maturity_bucket',
                                                values='TMTOTOUT', aggfunc='sum', fill_value=0)
    
    # Calculate the proportional value for each bucket by dividing by the total notional for each quote_date
    total_notional_per_date = notional_bonds_pivot.sum(axis=1)
    proportional_notional_pivot = notional_bonds_pivot.div(total_notional_per_date, axis=0)
    proportional_notional_pivot = proportional_notional_pivot.sort_index()
    proportional_notional_pivot = proportional_notional_pivot.reindex(columns=maturity_bucket_name, fill_value=0)
    
    return proportional_notional_pivot


def generate_callable_table(data, tax, billflag):
    """Pivot table for number of callable bonds."""
    data_tax = data[(data['ITAX'] == tax) & (data['callflag'] == 1)]
    if billflag == True:
        data_tax = data[data['ITYPE'] == 4]
    callable_bonds_pivot = data_tax.pivot_table(index='quote_date', columns='maturity_bucket',
                                                aggfunc='size', fill_value=0)
    callable_bonds_pivot = callable_bonds_pivot.sort_index()
    callable_bonds_pivot = callable_bonds_pivot.reindex(columns=maturity_bucket_name, fill_value=0)
    callable_bonds_pivot.loc[:,'Row_Total']= callable_bonds_pivot.sum(numeric_only=True, axis=1)

    return callable_bonds_pivot


def generate_flower_table(data, tax, billflag):
    """Pivot table for number of flower bonds."""
    data_tax = data[(data['ITAX'] == tax) & (data['IFLWR'] != 1)]
    if billflag == True:
        data_tax = data[data['ITYPE'] == 4]
    flower_bonds_pivot = data_tax.pivot_table(index='quote_date', columns='maturity_bucket',
                                              aggfunc='size', fill_value=0)
    flower_bonds_pivot = flower_bonds_pivot.sort_index()
    flower_bonds_pivot = flower_bonds_pivot.reindex(columns=maturity_bucket_name, fill_value=0)
    flower_bonds_pivot.loc[:,'Row_Total']= flower_bonds_pivot.sum(numeric_only=True, axis=1)

    return flower_bonds_pivot


def output_tables(data, name, billflag):
    """Output seperate bond stats tables in csv and an all-in-one excel, of either bond or bill."""
    
    suffix = "bill" if billflag else "bond"

    writer = pd.ExcelWriter(f'Bond_Stats_{name}_{suffix}.xlsx', engine='xlsxwriter')
    
    for tax in [1, 2, 3]:
        n_table = generate_n_table(data, tax, billflag)
        # notional_table = generate_notional_table(data, tax, billflag)
        prop_notional_table = generate_proportional_notional_table(data, tax, billflag)
        callable_table = generate_callable_table(data, tax, billflag)
        flower_table = generate_flower_table(data, tax, billflag)
        n_table.loc[callable_table.index,'%_Callable'] = callable_table.loc[:,'Row_Total'] /n_table.loc[callable_table.index,'Row_Total']
        n_table.loc[callable_table.index,'%_Call15+'] = (callable_table.loc[:,'15YR-20YR']+callable_table.loc[:,'20YR and beyond']) / (n_table.loc[callable_table.index,'15YR-20YR']+n_table.loc[callable_table.index,'20YR and beyond'])
        n_table.loc[flower_table.index,'%_Flower'] = flower_table.loc[:,'Row_Total'] / n_table.loc[flower_table.index,'Row_Total']
        n_table.loc[flower_table.index,'%_Flow15+'] = (flower_table.loc[:,'15YR-20YR']+flower_table.loc[:,'20YR and beyond']) / (n_table.loc[flower_table.index,'15YR-20YR']+n_table.loc[flower_table.index,'20YR and beyond'])
            
        n_table.to_csv(f'n_table_tax{tax}_{name}_{suffix}.csv')
        # notional_table.to_csv(f'notional_table_tax{tax}_{suffix}.csv')
        prop_notional_table.to_csv(f'prop_notional_tax{tax}_{name}_{suffix}.csv')
        callable_table.to_csv(f'callable_tax{tax}_{name}_{suffix}.csv')
        flower_table.to_csv(f'flower_tax{tax}_{name}_{suffix}.csv')

        # Add each table to a different sheet in one Excel workbook
        n_table.to_excel(writer, sheet_name=f'N_Tax{tax}_{name}_{suffix}')
        prop_notional_table.to_excel(writer, sheet_name=f'Prop_Notional_Tax{tax}_{name}_{suffix}')
        callable_table.to_excel(writer, sheet_name=f'Callable_Tax{tax}_{name}_{suffix}')
        flower_table.to_excel(writer, sheet_name=f'Flower_Tax{tax}_{name}_{suffix}')
        
        writer.close()



def main():

    filepath = '../curve_utils/data/1916to2024_YYYYMMDD.csv'
    filepath = '../curve_utils/data/USTMonthly.csv'
    bonddata_raw = clean_raw_crsp(filepath)
    bonddata_cleaned = clean_crsp(filepath)

    # Produce breaks for each quote date
    OUTPUT_DIR = '../curve_utils/output'
    breaks_df = generate_breaks_df(bonddata_cleaned)
    breaks_df.to_csv(OUTPUT_DIR+'/'+'breaks_df.csv')
    breaks_df.to_pickle(OUTPUT_DIR+'/'+'breaks_df.pkl')


    bonddata_raw['maturity_bucket'] = bonddata_raw['days_to_maturity'].apply(
        classify_into_buckets)
    bonddata_cleaned['maturity_bucket'] = bonddata_cleaned['days_to_maturity'].apply(
        classify_into_buckets)


    num_bonddata = generate_n_table_breaks(bonddata_cleaned)

    # # Define breaks for each quote date

    # base_breaks = [7/365.25, 0.0833, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    # maturity_bucket_name = [
    #     '7days', '7days-1mon', '1mon-3mons', '6mons-1YR', '1YR-2YR', 
    #     '2YR-3YR', '3YR-5YR', '5YR-7YR', '7YR-10YR', '10YR-20YR', '20YR-30YR'
    # ]

    # breaks_df = []

    # for date, row in num_bonddata.iterrows():
    #     # Collect non-zero maturity buckets
    #     non_zero_breaks = [base_breaks[i] for i, bucket in enumerate(maturity_bucket_name) if row[bucket] > 0]
    #     breaks_df.append([date, non_zero_breaks])

    # breaks_df = pd.DataFrame(breaks_df, columns=['quote_date', 'breaks'])


    # All Bond
    output_tables(bonddata_raw, 'Raw', False)
    output_tables(bonddata_cleaned, 'Cleaned', False)
    
    # Bill
    output_tables(bonddata_raw, 'Raw', True)
    output_tables(bonddata_cleaned, 'Cleaned', True)



if __name__ == "__main__":
    main()


# %%
