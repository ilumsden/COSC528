import pandas as pd
import numpy as np

def drop_unnessecary_data(df):
    num_rows = df.shape[0]
    nanrow = 0
    for i in range(num_rows):
        try:
            val = float(df.iloc[i, 1])
            if np.isnan(val):
                nanrow = i
                break
        except ValueError:
            continue
    df.drop(index=range(i, num_rows), inplace=True)
    df.drop(df.columns[[0, 1, 2]], axis=1, inplace=True)

def convert_discrete(df):
    for i, v in df["HBC"].iteritems():
        if str(v) == "No":
            df.loc[i, "HBC"] = 0.0
        else:
            df["HBC"][i] = 1.0
    for i, v in df["2014 Med School"].iteritems():
        try:
            if np.isnan(float(v)):
                df.loc[i, "2014 Med School"] = 0.0
                continue
        except ValueError:
            pass
        if str(v) == "x":
            df.loc[i, "2014 Med School"] = 1.0
        else:
            df.loc[i, "2014 Med School"] = 2.0
    for i, v in df["Vet School"].iteritems():
        try:
            if np.isnan(float(v)):
                df.loc[i, "Vet School"] = 0.0
                continue
        except ValueError:
            pass
        df.loc[i, "Vet School"] = 1.0

# Assumes unnesecary data is removed
def remove_dollar_signs(df):
    tmp = df[df.columns[:]].replace("[\$,]", "", regex=True)
    tmp.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    num_rows = tmp.shape[0]
    num_cols = tmp.shape[1]
    for r in range(num_rows):
        for c in range(num_cols):
            try:
                val = str(tmp.iloc[r, c]).strip()
                if val == "-":
                    tmp.iloc[r, c] = np.nan
            except ValueError:
                continue
    tmp = tmp.values
    tmp = tmp.astype(dtype=np.float64)
    return tmp

def remove_nan(data):
    means = np.nanmean(data, axis=0)
    for i, m in enumerate(means):
        for r, row in enumerate(data):
            if np.isnan(row[i]):
                data[r, i] = m

def normalize(data):
    #from scipy import stats
    #return stats.zscore(data, axis=1)
    means = np.mean(data, axis=1)
    means = np.reshape(means, (len(means), 1))
    return np.subtract(data, means)
