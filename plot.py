import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("result-py.csv", sep="\s+")
df1.columns = ['y', 'vr', 'vfr']

df2 = pd.read_csv("results-f.csv", sep="\s+")
df2 = df2.drop(df2[df2.kuta == "kuta"].index)


df1["y"] = (df1["y"].astype('float32')).round(decimals=1)
df1["vr"] = df1["vr"].astype('float64')
df1["vfr"] = df1["vfr"].astype('float64')

df2["kuta"] = df2["kuta"].astype('int')
df2["y"] = (df2["y"].astype('float32')).round(decimals=1)
df2["vr"] = df2["vr"].astype('float64')
df2["vfr"] = df2["vfr"].astype('float64')

# r = np.unique(df2.vr.to_numpy(dtype='float64'))

y_ = 1.0

sub1_ = df1.loc[(df1['y'] == y_)]
r1 = np.concatenate(sub1_[['vr']].to_numpy(), axis=0)
n1 = np.concatenate(sub1_[['vfr']].to_numpy(), axis=0)

sub2_ = df2.loc[(df2['kuta'] == 4) & (df2['y'] == y_)]
r2 = np.concatenate(sub2_[['vr']].to_numpy(),axis=0)
n2 = np.concatenate(sub2_[['vfr']].to_numpy(),axis=0)

plt.xscale('log')
plt.plot(r1, n1)
plt.plot(r2, n2)
plt.show()

