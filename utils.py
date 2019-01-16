import numpy as np
import pandas as pd
from datetime import datetime


exch_rate = {'丹麦克朗': 1.0571, '加元': 5.2668, '美元': 6.931, '欧元': 7.9076, '日元': 0.06094, '瑞士法郎': 6.9158,
             '香港元': 0.8857, '新加坡元': 5.0521, '英镑': 9.1016}


def get_date(x):
    today = datetime.strptime('2019-01-01', '%Y-%m-%d')
    if '/' in x:
        today = datetime.strptime('2019/01/01', '%Y/%m/%d')
        return (today - datetime.strptime(x, '%Y/%m/%d')).days
    return (today - datetime.strptime(x, '%Y-%m-%d')).days


def get_diff_mean(x):
    if x.shape[0] == 1:
        return 0
    x.sort_values(inplace=True)
    values = x.values
    diffs = []
    for i in range(len(values) - 1):
        diffs.append(np.abs(values[i + 1] - values[i]))
    return np.mean(diffs)


def get_df(filename, features):
    train_df1 = pd.read_csv(open('data/round1/train/' + filename + '.csv', encoding='utf-8'))
    train_df2 = pd.read_csv(open('data/round2/train/' + filename + '.csv', encoding='utf-8'))
    train_df = pd.concat([train_df1, train_df2], axis=0, ignore_index=True)
    train_num = train_df.shape[0]
    test_df = pd.read_csv(open('data/round2/test/' + filename + '.csv', encoding='utf-8'))
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    for col in df.columns.values:
        if col not in features:
            del df[col]
    new_columns = ['企业名称']
    for col in df.columns.values[1:]:
        new_columns.append(filename + '_' + col)
    df.rename(columns=dict(zip(df.columns.values, new_columns)), inplace=True)
    return df, train_num


def get_agg(group, feature, funcs):
    d = {}
    for func in funcs:
        if 'diff_mean' == func:
            d[feature + '_' + func] = get_diff_mean
            continue
        d[feature + '_' + func] = func
    return group[feature].agg(d)


def binning(x, intervals):
    for i in range(len(intervals)):
        if x <= intervals[i]:
            return i
    return len(intervals)


def get_ratio(df, feature_pairs):
    for pair in feature_pairs:
        df = pd.merge(df, df.groupby(pair[0], as_index=False)['企业名称'].agg({pair[0] + '_count': 'count'}),
                      on=pair[0], how='left')
        df = pd.merge(df, df.groupby(pair[1], as_index=False)['企业名称'].agg({pair[1] + '_count': 'count'}),
                      on=pair[1], how='left')
        df = pd.merge(df, df.groupby(pair, as_index=False)['企业名称'].agg(
            {pair[0] + '_' + pair[1] + '_count': 'count'}), on=pair, how='left')
        df[pair[0] + '_ratio_' + pair[1]] = df[pair[0] + '_' + pair[1] + '_count'] / df[pair[1] + '_count']
        df[pair[1] + '_ratio_' + pair[0]] = df[pair[0] + '_' + pair[1] + '_count'] / df[pair[0] + '_count']
        del df[pair[0] + '_count'], df[pair[1] + '_count'], df[pair[0] + '_' + pair[1] + '_count']
    return df
