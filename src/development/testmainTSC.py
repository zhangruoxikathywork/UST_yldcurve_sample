# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% main commands

OUTPUT_DIR = '../../output'

# Define user inputs
# quotedate = 20150130 # 19321231
calltype = 0  # 0 to keep all bonds, 1 for callable bonds, 2 for non-callable bonds
curvetypes = ['pwcf', 'pwlz', 'pwtf']
start_date = 19890101
end_date = 19891231
breaks = np.array([7/365.25, 14/365.25, 21/365.25, 28/365.25, 35/365.25, 52/365.25, 92/365.25, 
                   184/365.25, 1, 2, 4, 8, 16, 24, 32])  # np.array([0.0833, 0.5, 1.,2.,5.,10.,20.,30.])
curve_points_yr = np.arange(.01,10,.01)

# New wrds data
filepath = '../../data/USTMonthly.csv'  # '../../data/1916to2024_YYYYMMDD.csv'
bonddata = data_processing.clean_crsp(filepath)

all_dfs = produce_curve_wrapper(bonddata, curvetypes, start_date, end_date, breaks, '1989', calltype=0, wgttype=1, lam1=1, lam2=2, padj=False, padjparm=0, yield_to_worst=True)

# Plotly plot for December
plot_rates.plot_rates_by_one_month(all_dfs, 'pwcf', 1)

# filepath = "../../data/CRSP12-1932to3-1933.txt"
# fortran preprocessed
# parms, prices = read_and_process_fortrandata(filepath, calltype=0)

# Use uncleaned data
# bonddata_raw = analysis.clean_raw_crsp(filepath)
# # Add callflag
# non_call = pd.isnull(bonddata['TFCALDT'])     # TRUE when call is nan
# bonddata.loc[non_call , 'TFCALDT'] = 0.0      # Replace nan s with 0s
# bonddata['callflag'] = np.where(bonddata['TFCALDT'] == 0, 0, 1)
# parms, prices, weights = read_and_process_csvdata(bonddata, quotedate, calltype=0)
# curve, bondpv = calculate_rate_notax(parms, prices, weights, quotedate, breaks, curvetype, weight_flag=0)
# curvetax = calculate_rate_with_tax(parms, prices, quotedate, breaks, curvetype)
# price_yield_df = get_predicted_actual_yieldprice_notax(parms, bondpv, prices, quotedate, curvetype)
# plot_no_tax_curve(parms, prices, quotedate, breaks, curve_points_yr)
# plot_one_type_fwdcurve_tax(parms, prices, quotedate, breaks, curve_points_yr, curvetype)


#%% Wrapper commands



"""Loop through months from a given start and end date for cleaned crsp UST bond data to produce forward rates
and predicted versus actual price and yield. Output to individual csv files and consolidated excel files with different tabs.

Args:
    bonddata (pd.Dataframe): Cleaned CRSP bond dataset, after applying data_processing.clean_crsp and
                             data_processing.create_weight to the WRDS CRSP monthly UST data.
    curvetypes (list): A list of curve type strings.
    start_date (int): Start date for end result.
    end_date (int): End date for end result.
    breaks (np.array): An array of date breaks, in years, e.g. np.array([0.0833, 0.5, 1.,2.,5.,10.,20.,30.])
    calltype (int, optional): Whether to filter callable bond. 0=all, 1=callable only, 2=non-callable only.
                              Defaults to 0.
    weight_flag (int, optional): Whether to use weights for SSQ. Defaults to 0. >> could use weight_flag
                                 to determine weight method. 

Returns:
    all_dfs: Dictionary containing final_curve_df and final_price_yield_df for each curvetype. Dataframes can be individually selected.
"""

# curvetype = 'pwcf'

# Convert start_date and end_date to datetime format if they are not already
if not isinstance(start_date, datetime.datetime):
    start_date = pd.to_datetime(str(start_date), format='%Y%m%d')
if not isinstance(end_date, datetime.datetime):
    end_date = pd.to_datetime(str(end_date), format='%Y%m%d')

# Dictionary to store final DataFrames for each curve type
all_dfs = {}

wb = Workbook()

#for curvetype in curvetypes:
# Instead of looping over all curvetypes, let's just try the first
curvetype = curvetypes[0]
    
curve_data_list = []
price_yield_data_list = []

filtered_data = bonddata[(bonddata['quote_date'] > start_date) & (bonddata['quote_date'] < end_date)]
quotedates = list(set(filtered_data['MCALDT']))

#for quotedate in quotedates:
# Instead of all quotedates, let's try the first
quotedate = quotedates[0]
# !!! Error here "['BWT'] not in index"
parms = inputs.read_and_process_csvdata(filtered_data, quotedate, calltype)
quotedate = int(quotedate)
parms = inputs.filter_yield_to_worst_parms(quotedate, parms, yield_to_worst)
parms = inputs.create_weight(parms, wgttype, lam1, lam2)
curve, prices, bondpv = outputs.calculate_rate_notax(parms, quotedate, breaks, curvetype, wgttype, lam1, lam2,
                                                     padj, padjparm)
# curve_tax = outputs.calculate_rate_tax(parms, quotedate, breaks, curvetype, wgttype, lam1, lam2,
#                                                      padj, padjparm)
price_yield_df = outputs.get_predicted_actual_yieldprice_notax(parms, bondpv, prices, quotedate, curvetype, padj)
price_yield_df.insert(0, 'QuoteDate', quotedate)

# Aggregate curve and price_yield_df data
curve_df = pd.DataFrame(curve[3].reshape(1, -1), index=[quotedate], columns=breaks)
curve_data_list.append(curve_df)
price_yield_data_list.append(price_yield_df)

final_curve_df = pd.concat(curve_data_list)
final_curve_df = final_curve_df.sort_index()
final_price_yield_df = pd.concat(price_yield_data_list)
final_price_yield_df = final_price_yield_df.sort_values(by=['QuoteDate', 'MatYr', 'MatMth', 'MatDay'])

all_dfs[curvetype] = {'curve_df': final_curve_df, 'price_yield_df': final_price_yield_df}

# Export to CSV
final_curve_df.to_csv(os.path.join(OUTPUT_DIR, f'curve_{curvetype}_{filename}.csv'))
final_price_yield_df.to_csv(os.path.join(OUTPUT_DIR, f'price_yield_{curvetype}_{filename}.csv'))

# Export to Excel on separate sheets
excel_file_path = os.path.join(OUTPUT_DIR, f'{curvetype}_{filename}_data.xlsx')
with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
    final_curve_df.to_excel(writer, sheet_name='Curve Data')
    final_price_yield_df.to_excel(writer, sheet_name='Price Yield Data')
