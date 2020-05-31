import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("/kaggle/input/fbprophet-0/part_0.csv")

for i in range(1,10):
    df = df.append(pd.read_csv("/kaggle/input/fbprophet-{}/part_{}.csv".format(i, i)))

df2 = df.iloc[:, -28:]
df = df.iloc[:, :-28]

submit = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")

submit.iloc[:30490, 1:] = df2.values

submit2 = submit
submit2 = submit2.drop(['id'], axis = 1) 
submit2[submit2 < 0] = 0
submit = pd.concat([submit['id'], submit2], axis = 1)
submit.to_csv("submission.csv", index = False)

import gc
del submit, df2
gc.collect()

df3 = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")



df[df < 0] = 0
df4 = pd.DataFrame(df3.iloc[:, 6:].values - df.values)
df4.columns = df3.columns[6:]
df4 = pd.concat([df3.iloc[:, :6], df4], axis = 1)
del df3
gc.collect()
df4.to_csv("sales_train_validation_mod.csv", index = False)



# 
# residuals = pd.read_csv("/kaggle/input/fbprophet-0/residuals_part_0.csv")
# for i in range(1,10):
#     residuals = residuals.append(pd.read_csv("/kaggle/input/fbprophet-{}/residuals_part_{}.csv".format(i, i)))

