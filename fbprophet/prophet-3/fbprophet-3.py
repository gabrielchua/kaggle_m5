part = 3

start_pt = 3049 * (part)
end_pt = 3049 * (part + 1)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fbprophet import Prophet

df = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")
#df2 = pd.read_csv("/kaggle/input/calculate-revenue/revenue.csv")

launch_date = pd.read_csv("/kaggle/input/identify-first-sell-date-andprices/m5_launch_date.csv")
launch_date = np.array(launch_date['d'].values)
prices = pd.read_csv("/kaggle/input/identify-first-sell-date-andprices/m5_prices_wide.csv")
weekends = pd.read_csv("/kaggle/input/identify-first-sell-date-andprices/m5_weekends.csv").values.flatten()
snap = pd.read_csv("/kaggle/input/identify-first-sell-date-andprices/snap.csv").values
holidays = pd.read_csv("/kaggle/input/identify-first-sell-date-andprices/m5_holidays.csv")


cum7 = pd.read_csv("/kaggle/input/identify-first-sell-date-andprices/cum_7_freq_zero.csv").values
# cum14 = pd.read_csv("/kaggle/input/identify-first-sell-date-andprices/cum_14_freq_zero.csv").values
# cum28 = pd.read_csv("/kaggle/input/identify-first-sell-date-andprices/cum_28_freq_zero.csv").values
# cum56 = pd.read_csv("/kaggle/input/identify-first-sell-date-andprices/cum_56_freq_zero.csv").values
cum_max = pd.read_csv("/kaggle/input/identify-first-sell-date-andprices/cumulative_max.csv").values
cum_zero = pd.read_csv("/kaggle/input/identify-first-sell-date-andprices/cumulative_zero.csv").values


day_columns = df.columns[df.columns.str.contains("d_")]
day_list_int = day_columns.str.replace("d_", "").astype(int)

results_store = np.zeros(1941).reshape(-1,1)
residuals_store = np.zeros(1913).reshape(-1,1)

for i in range(start_pt, end_pt):
    #start_t = time.time()
    start_date = launch_date[i]
    time_series = df.iloc[i,6+(start_date-1):].values
    time_series = pd.DataFrame([day_list_int[start_date-1:], time_series]).transpose()
    prices_vec = prices.iloc[i,start_date:-28].values
    
    time_series['weekends'] = weekends[(start_date-1):-28]
    time_series['snap'] = snap[i,(start_date-1):-28]
    #time_series['floor'] = 0
    
    if np.max(prices_vec) - np.min(prices_vec) > 2: # was the opposite previously
        price_regressor = True
        time_series['price'] = prices_vec
        #time_series.columns = ['ds', 'y', 'weekends', 'snap', 'floor', 'price']
        time_series.columns = ['ds', 'y', 'weekends', 'snap', 'price']
    else: 
        price_regressor = False
        #time_series.columns = ['ds', 'y', 'weekends', 'snap', 'floor']
        time_series.columns = ['ds', 'y', 'weekends', 'snap']
        
    #time_max = np.max(time_series['y']) * 1.1
    #time_series['cap'] = time_max

    time_series['cum7'] = cum7[i, (start_date-1):-28]
    # time_series['cum14'] = cum14[i, (start_date-1):-28]
    # time_series['cum28'] = cum28[i, (start_date-1):-28]
    # time_series['cum56'] = cum56[i, (start_date-1):-28]
    time_series['cum_max'] = cum_max[i, (start_date-1):-28]
    time_series['cum_zero'] = cum_zero[i, (start_date-1):-28]
    
    end_train = len(time_series) - 28
    time_series.loc[:, 'ds'] = pd.datetime(2011,1,29) + pd.to_timedelta(time_series['ds'] - 1, unit = 'd')
    
    m = Prophet(uncertainty_samples = 0, holidays = holidays, changepoint_prior_scale=0.9, holidays_prior_scale=0.05, yearly_seasonality = 5) #growth='logistic')

    # m.add_country_holidays(country_name='US')
    
    if price_regressor == True:
        m.add_regressor('price')
    
    m.add_regressor('weekends')
    m.add_regressor('snap')

    m.add_regressor('cum7')
    # m.add_regressor('cum14')
    # m.add_regressor('cum28')
    # m.add_regressor('cum56')
    m.add_regressor('cum_max')
    m.add_regressor('cum_zero')
        
    m.add_seasonality(name='monthly', period=30.5, fourier_order=4)
    m.add_seasonality(name='quarterly', period=91, fourier_order=4) # new
    m.fit(time_series)
    future = m.make_future_dataframe(periods=28)
    
    if price_regressor == True:
        future['price'] = prices.iloc[i,start_date:].values
    
    future['snap'] = snap[i,(start_date)-1:]
    future['weekends'] = weekends[start_date-1:]

    time_series['cum7'] = cum7[i, (start_date-1):]
    # time_series['cum14'] = cum14[i, (start_date-1):]
    # time_series['cum28'] = cum28[i, (start_date-1):]
    # time_series['cum56'] = cum56[i, (start_date-1):]
    time_series['cum_max'] = cum_max[i, (start_date-1):]
    time_series['cum_zero'] = cum_zero[i, (start_date-1):]

    #future['floor'] = 0
    #future['cap'] = time_max
    
    forecast = m.predict(future)
    
    forecast2 = forecast['yhat'].values
    forecast2 = np.pad(forecast2, (1941-len(forecast),0), 'constant')
    results_store = np.concatenate([results_store, forecast2.reshape(-1,1)], axis = 1)

    residuals = time_series['y'] - forecast['yhat'].values[:-28]
    residuals = np.pad(residuals, (1913-len(residuals),0), 'constant')
    residuals_store = np.concatenate([residuals_store, residuals.reshape(-1,1)], axis = 1)
    #sample.iloc[i, 1:] = forecast

results_store = results_store[:, 1:]
results_store = pd.DataFrame(results_store).transpose()
results_store.to_csv("part_{}.csv".format(part), index = False)

residuals_store = residuals_store[:, 1:]
residuals_store = pd.DataFrame(residuals_store).transpose()
residuals_store.to_csv("residuals_part_{}.csv".format(part), index = False)
