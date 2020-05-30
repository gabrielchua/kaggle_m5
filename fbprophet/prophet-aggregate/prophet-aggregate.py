import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fbprophet import Prophet

df = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

day_columns = df.columns[df.columns.str.contains("d_")]
day_list_int = day_columns.str.replace("d_", "").astype(int)




lvl_1 = pd.DataFrame(np.sum(df.iloc[:, 6:], axis = 0)).transpose()
lvl_1['level'] = 1
print(lvl_1.shape)
lvl_1.head()



lvl_2 = df.groupby('state_id')[day_columns].sum()
lvl_2['level'] = 2
print(lvl_2.shape)
lvl_2.head()



lvl_3 = df.groupby('store_id')[day_columns].sum()
lvl_3['level'] = 3
print(lvl_3.shape)
lvl_3.head()


lvl_4 = df.groupby(['store_id', 'cat_id'])[day_columns].sum()
lvl_4['level'] = 4
print(lvl_4.shape)
lvl_4.head()


lvl_4 = lvl_4.reset_index()
lvl_4['index'] = lvl_4['store_id'] + '_'+ lvl_4['cat_id']
lvl_4 = lvl_4.drop(['store_id', 'cat_id'], axis = 1)
lvl_4 = lvl_4.set_index(['index'])



lvl_5 = df.groupby(['store_id', 'cat_id', 'dept_id'])[day_columns].sum()
lvl_5['level'] = 5


lvl_5 = lvl_5.reset_index()
lvl_5['index'] = lvl_5['store_id'] + '_'+ lvl_5['cat_id']  + '_'+ lvl_5['dept_id']
lvl_5 = lvl_5.drop(['store_id', 'cat_id', 'dept_id'], axis = 1)
lvl_5 = lvl_5.set_index(['index'])


df2 = lvl_1.append(lvl_2).append(lvl_3).append(lvl_4).append(lvl_5)


output = pd.DataFrame([df2.index, df2['level']]).transpose()
output.columns = ['id', 'level']
for i in range(1, 29):
    output["d_" + str(i)] = 0

for i in range(0, 114):
    time_series = df2.iloc[i,:-1].values
    time_series = pd.DataFrame([day_list_int, time_series]).transpose()
    time_series.columns = ['ds', 'y']
    end_train = len(time_series) - 28
    time_series_train = time_series.iloc[:end_train,:]
    time_series_test = time_series.iloc[end_train:,:]
    time_series_train.loc[:, 'ds'] = pd.datetime(2011,1,29) + pd.to_timedelta(time_series_train['ds'] - 1, unit = 'd')
    m = Prophet(uncertainty_samples = 0, changepoint_prior_scale=0.9, holidays_prior_scale=0.05, yearly_seasonality = 5)
    m.add_country_holidays(country_name='US')
    m.fit(time_series_train)
    forecast = m.predict(m.make_future_dataframe(periods=28))
    
    forecast = forecast.tail(28)['yhat'].values
    output.iloc[i, 2:] = forecast