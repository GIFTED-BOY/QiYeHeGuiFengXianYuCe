import numpy as np
import pandas as pd
import utils
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def fe1():
    prefix = '企业基本信息&高管信息&投资信息'
    df, train_num = utils.get_df(prefix, ['企业名称', '注册资金', '注册资本(金)币种名称', '企业(机构)类型名称',
                                          '行业门类代码', '成立日期', '核准日期', '住所所在地省份', '姓名',
                                          '法定代表人标志', '首席代表标志', '职务', '投资人', '出资比例'])
    df[prefix + '_出资比例'] = df[prefix + '_出资比例'].apply(lambda x: x if x <= 1 else x / 100)
    df[prefix + '_注册资金'] = df[[prefix + '_注册资金', prefix + '_注册资本(金)币种名称']].apply(
        lambda x: x[prefix + '_注册资金'] if x[prefix + '_注册资本(金)币种名称'] not in utils.exch_rate.keys()
        else x[prefix + '_注册资金'] * utils.exch_rate[x[prefix + '_注册资本(金)币种名称']], axis=1).fillna(0)
    df[prefix + '_注册资金_binning'] = df[prefix + '_注册资金'].apply(
        lambda x: utils.binning(x, [300, 500, 1000, 3000, 6000]))
    df[prefix + '_成立日期'] = df[prefix + '_成立日期'].astype('str').apply(
        lambda x: utils.get_date(x[:10]) if x != 'nan' else np.nan)
    df[prefix + '_核准日期'] = df[prefix + '_核准日期'].apply(lambda x: utils.get_date(x[:10]))
    df[prefix + '_成立日期_核准日期_diff'] = df[prefix + '_成立日期'] - df[prefix + '_核准日期']
    df[prefix + '_法定代表人职务'] = df[[prefix + '_法定代表人标志', prefix + '_职务']].apply(
        lambda x: x[prefix + '_职务'] if x[prefix + '_法定代表人标志'] == '是' else np.nan, axis=1)
    df[prefix + '_首席代表职务'] = df[[prefix + '_首席代表标志', prefix + '_职务']].apply(
        lambda x: x[prefix + '_职务'] if x[prefix + '_首席代表标志'] == '是' else np.nan, axis=1)
    df = pd.merge(df, df.dropna(subset=[prefix + '_姓名']).groupby(
        prefix + '_姓名', as_index=False)['企业名称'].agg({prefix + '_姓名_企业名称_nunique': 'nunique'}),
                  on=prefix + '_姓名', how='left')
    df = pd.merge(df, df.dropna(subset=[prefix + '_投资人']).groupby(
        prefix + '_投资人', as_index=False)['企业名称'].agg({prefix + '_投资人_企业名称_nunique': 'nunique'}),
                  on=prefix + '_投资人', how='left')
    group = df.groupby('企业名称', as_index=False)
    df = pd.merge(df, utils.get_agg(group, prefix + '_姓名', ['nunique']), on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_投资人', ['nunique']), on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_姓名_企业名称_nunique', ['max', 'mean', 'sum']),
                  on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_投资人_企业名称_nunique', ['max', 'mean', 'sum']),
                  on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_出资比例', ['max', 'min', 'mean']), on='企业名称', how='left')
    f_pairs = [
        [prefix + '_住所所在地省份', prefix + '_企业(机构)类型名称'],
        [prefix + '_住所所在地省份', prefix + '_行业门类代码'],
        [prefix + '_注册资金_binning', prefix + '_企业(机构)类型名称'],
        [prefix + '_注册资金_binning', prefix + '_行业门类代码'],
        [prefix + '_企业(机构)类型名称', prefix + '_行业门类代码']
    ]
    df = utils.get_ratio(df, f_pairs)
    for f in ['注册资本(金)币种名称', '姓名', '法定代表人标志', '首席代表标志', '职务', '投资人', '出资比例',
              '姓名_企业名称_nunique', '投资人_企业名称_nunique']:
        del df[prefix + '_' + f]
    train_df, test_df = df[:train_num], df[train_num:]
    train_df.drop_duplicates('企业名称', inplace=True)
    train_df.dropna(subset=[prefix + '_成立日期'], inplace=True)
    test_df.drop_duplicates('企业名称', inplace=True)
    for f in ['法定代表人职务', '首席代表职务', '企业(机构)类型名称', '行业门类代码', '住所所在地省份']:
        label_dict = dict(zip(train_df[prefix + '_' + f].unique(), range(train_df[prefix + '_' + f].nunique())))
        train_df[prefix + '_' + f] = train_df[prefix + '_' + f].map(label_dict).fillna(-1).astype('int16')
        test_df[prefix + '_' + f] = test_df[prefix + '_' + f].map(label_dict).fillna(-1).astype('int16')
    return prefix, train_df, test_df


def fe2():
    def recruiting_numbers(x):
        if '若干' == x:
            return np.nan
        if '人' in x:
            return x[:-1]
        return x
    prefix = '招聘数据'
    df, train_num = utils.get_df(prefix, ['企业名称', '招聘人数', '招聘日期'])
    df[prefix + '_招聘人数'] = df[prefix + '_招聘人数'].fillna('若干').astype('str').apply(
        recruiting_numbers).astype('float32')
    df[prefix + '_招聘日期'] = df[prefix + '_招聘日期'].apply(utils.get_date)
    group = df.groupby('企业名称', as_index=False)
    df = pd.merge(df, group['企业名称'].agg({prefix + '_count': 'count'}), on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_招聘人数', ['mean', 'sum']),
                  on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_招聘日期', ['max', 'min', 'mean', 'diff_mean']),
                  on='企业名称', how='left')
    df[prefix + '_freq'] = df[prefix + '_count'] / (
            df[prefix + '_招聘日期_max'] - df[prefix + '_招聘日期_min'] + 1)
    for f in ['招聘人数', '招聘日期']:
        del df[prefix + '_' + f]
    train_df, test_df = df[:train_num], df[train_num:]
    train_df.drop_duplicates('企业名称', inplace=True)
    test_df.drop_duplicates('企业名称', inplace=True)
    return prefix, train_df, test_df


def fe3():
    def business_scope(x):
        x = re.sub('（.*?）', '', x)
        if '。' == x[-1]:
            x = x[:-1]
        if '。' in x:
            if '；' in x:
                x.replace('。', '；')
            elif '、' in x:
                x.replace('。', '、')
            else:
                x.replace('。', ' ')
        return x
    def business_scope_to_key_words(x, key_words):
        for w in key_words:
            if w in x:
                return w
        return '其他'
    prefix = '机构设立（变更）登记信息'
    df, train_num = utils.get_df(prefix, ['企业名称', '注册（开办）资金', '实收资金', '经营范围', '企业类型代码',
                                          '所属行业代码', '机构地址（住所）', '发证日期', '行政区划'])

    df[prefix + '_经营范围'] = df[prefix + '_经营范围'].astype('str').apply(business_scope)
    sentences = df[prefix + '_经营范围'].values
    docs = []
    for s in sentences:
        if '；' in s:
            docs.append(' '.join(s.split('；')))
        elif '、' in s:
            docs.append(' '.join(s.split('、')))
        else:
            docs.append(' '.join(s.split(' ')))
    v = TfidfVectorizer(max_df=0.7, min_df=323)
    v.fit_transform(docs)
    key_words = v.get_feature_names()
    df[prefix + '_经营范围'] = df[prefix + '_经营范围'].apply(
        lambda x: business_scope_to_key_words(x, key_words))

    df[prefix + '_注册（开办）资金'] = df[[prefix + '_注册（开办）资金', prefix + '_实收资金']].apply(
        lambda x: x[prefix + '_注册（开办）资金'] if x[prefix + '_实收资金'] not in utils.exch_rate.keys()
        else x[prefix + '_注册（开办）资金'] * utils.exch_rate[x[prefix + '_实收资金']], axis=1).fillna(0)
    df[prefix + '_注册（开办）资金_binning'] = df[prefix + '_注册（开办）资金'].apply(
        lambda x: utils.binning(x, [300, 500, 1000, 3000, 6000]))
    df[prefix + '_发证日期'] = df[prefix + '_发证日期'].apply(
        lambda x: utils.get_date(x) if '.' not in x else utils.get_date(x[:10]))
    df[prefix + '_机构地址（住所）'] = df[prefix + '_机构地址（住所）'].apply(lambda x: x[:2])
    df[prefix + '_行政区划'] = df[prefix + '_行政区划'].astype('str').apply(
        lambda x: x if x != '999999' else 'nan').astype('float32')
    f_pairs = [
        [prefix + '_行政区划', prefix + '_企业类型代码'],
        [prefix + '_行政区划', prefix + '_所属行业代码'],
        [prefix + '_经营范围', prefix + '_企业类型代码'],
        [prefix + '_经营范围', prefix + '_所属行业代码'],
        [prefix + '_经营范围', prefix + '_注册（开办）资金_binning'],
        [prefix + '_机构地址（住所）', prefix + '_企业类型代码'],
        [prefix + '_机构地址（住所）', prefix + '_所属行业代码'],
        [prefix + '_机构地址（住所）', prefix + '_行政区划'],
        [prefix + '_机构地址（住所）', prefix + '_注册（开办）资金_binning']
    ]
    df = utils.get_ratio(df, f_pairs)
    del df[prefix + '_注册（开办）资金'], df[prefix + '_注册（开办）资金_binning'], df[prefix + '_实收资金']
    train_df, test_df = df[:train_num], df[train_num:]
    train_df.drop_duplicates('企业名称', keep='last', inplace=True)
    test_df.drop_duplicates('企业名称', keep='last', inplace=True)
    for f in ['企业类型代码', '所属行业代码', '行政区划', '经营范围', '机构地址（住所）']:
        label_dict = dict(zip(train_df[prefix + '_' + f].unique(), range(train_df[prefix + '_' + f].nunique())))
        train_df[prefix + '_' + f] = train_df[prefix + '_' + f].map(label_dict).fillna(-1).astype('int16')
        test_df[prefix + '_' + f] = test_df[prefix + '_' + f].map(label_dict).fillna(-1).astype('int16')
    return prefix, train_df, test_df


def fe4():
    prefix = '双公示-法人行政许可信息'
    df, train_num = utils.get_df(prefix, ['企业名称', '许可决定日期', '许可截止期'])
    df[prefix + '_许可决定日期'] = df[prefix + '_许可决定日期'].astype('str').apply(
        lambda x: utils.get_date(x) if x != 'nan' else np.nan)
    df[prefix + '_许可截止期'] = df[prefix + '_许可截止期'].astype('str').apply(
        lambda x: utils.get_date(x) if x != 'nan' else np.nan)
    df[prefix + '_许可决定日期_许可截止期_diff'] = df[prefix + '_许可决定日期'] - df[prefix + '_许可截止期']
    group = df.groupby('企业名称', as_index=False)
    df = pd.merge(df, utils.get_agg(group, prefix + '_许可决定日期', ['max', 'min', 'mean', 'diff_mean']),
                  on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_许可截止期', ['max', 'min', 'mean']),
                  on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_许可决定日期_许可截止期_diff', ['max', 'min', 'mean']),
                  on='企业名称', how='left')
    for f in ['许可决定日期', '许可截止期', '许可决定日期_许可截止期_diff']:
        del df[prefix + '_' + f]
    train_df, test_df = df[:train_num], df[train_num:]
    train_df.drop_duplicates('企业名称', inplace=True)
    test_df.drop_duplicates('企业名称', inplace=True)
    return prefix, train_df, test_df


def fe5():
    prefix = '企业非正常户认定'
    df, train_num = utils.get_df(prefix, ['企业名称', '认定日期'])
    df[prefix] = 1
    df[prefix + '_认定日期'] = df[prefix + '_认定日期'].apply(utils.get_date)
    train_df, test_df = df[:train_num], df[train_num:]
    train_df.drop_duplicates('企业名称', inplace=True)
    test_df.drop_duplicates('企业名称', inplace=True)
    return prefix, train_df, test_df


def fe6():
    prefix = '许可资质年检信息'
    df, train_num = utils.get_df(prefix, ['企业名称', '年检结果', '年检事项名称', '年检日期'])
    df[prefix + '_年检结果'] = df[prefix + '_年检结果'].astype('str').apply(
        lambda x: '合格' if '合格' in x else '通过')
    df[prefix + '_年检事项名称'] = df[prefix + '_年检事项名称'].astype('str').apply(
        lambda x: x if '单位年检' == x or '对道路' in x else '其他')
    df = pd.get_dummies(df, prefix=['dummy_' + prefix + '_年检事项名称', 'dummy_' + prefix + '_年检结果'],
                        columns=[prefix + '_年检事项名称', prefix + '_年检结果'])
    del df['dummy_' + prefix + '_年检事项名称_其他']
    df[prefix + '_年检日期'] = df[prefix + '_年检日期'].astype('str').apply(
        lambda x: utils.get_date(x) if x != 'nan' else np.nan)
    raw_features = df.columns.values[1:]
    group = df.groupby('企业名称', as_index=False)
    for f in raw_features:
        if 'dummy' in f:
            df = pd.merge(df, utils.get_agg(group, f, ['sum', 'mean']), on='企业名称', how='left')
    df = pd.merge(df, group['企业名称'].agg({prefix + '_count': 'count'}), on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_年检日期', ['max', 'min', 'mean', 'diff_mean']),
                  on='企业名称', how='left')
    for f in raw_features:
        del df[f]
    train_df, test_df = df[:train_num], df[train_num:]
    train_df.drop_duplicates('企业名称', inplace=True)
    test_df.drop_duplicates('企业名称', inplace=True)
    return prefix, train_df, test_df


def fe7():
    prefix = '分支机构信息'
    df, train_num = utils.get_df(prefix, ['企业名称', '分支机构状态', '分支成立时间', '分支死亡时间'])
    df[prefix + '_分支机构状态'] = df[prefix + '_分支机构状态'].apply(lambda x: 1 if 1 == x else 0)
    df = pd.get_dummies(df, prefix=['dummy_' + prefix + '_分支机构状态'], columns=[prefix + '_分支机构状态'])
    df[prefix + '_分支成立时间'] = df[prefix + '_分支成立时间'].astype('str').apply(
        lambda x: utils.get_date(x) if x != 'nan' and x != '1899/12/30' else np.nan)
    df[prefix + '_分支死亡时间'] = df[prefix + '_分支死亡时间'].astype('str').apply(
        lambda x: utils.get_date(x) if x != 'nan' else np.nan)
    df[prefix + '_分支成立时间_分支死亡时间_diff']\
        = df[prefix + '_分支成立时间'] - df[prefix + '_分支死亡时间']
    raw_features = df.columns.values[1:]
    group = df.groupby('企业名称', as_index=False)
    for f in raw_features:
        if 'dummy' in f:
            df = pd.merge(df, utils.get_agg(group, f, ['sum', 'mean']), on='企业名称', how='left')
    df = pd.merge(df, group['企业名称'].agg({prefix + '_count': 'count'}), on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_分支成立时间', ['max', 'min', 'mean', 'diff_mean']),
                  on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_分支死亡时间', ['max', 'min', 'mean', 'diff_mean']),
                  on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_分支成立时间_分支死亡时间_diff', ['max', 'min', 'mean']),
                  on='企业名称', how='left')
    for f in raw_features:
        del df[f]
    train_df, test_df = df[:train_num], df[train_num:]
    train_df.drop_duplicates('企业名称', inplace=True)
    test_df.drop_duplicates('企业名称', inplace=True)
    return prefix, train_df, test_df


def fe8():
    prefix = '企业税务登记信息'
    df, train_num = utils.get_df(prefix, ['企业名称', '审核时间', '登记注册类型', '审核结果'])
    df[prefix + '_审核时间'] = df[prefix + '_审核时间'].apply(utils.get_date)
    df[prefix + '_审核结果'] = df[prefix + '_审核结果'].apply(
        lambda x: x if '江苏省苏州地方税务局' == x or '开业' == x or '正常' == x else '其他')
    f_pairs = [[prefix + '_登记注册类型', prefix + '_审核结果']]
    df = utils.get_ratio(df, f_pairs)
    train_df, test_df = df[:train_num], df[train_num:]
    train_df.drop_duplicates('企业名称', keep='last', inplace=True)
    test_df.drop_duplicates('企业名称', keep='last', inplace=True)
    for f in [prefix + '_审核结果', prefix + '_登记注册类型']:
        label_dict = dict(zip(train_df[f].unique(), range(train_df[f].nunique())))
        train_df[f] = train_df[f].map(label_dict).fillna(-1).astype('int16')
        test_df[f] = test_df[f].map(label_dict).fillna(-1).astype('int16')
    return prefix, train_df, test_df


def fe9():
    def date_proc(x1, x2):
        if 'nan' == x1 and 'nan' == x2:
            return np.nan
        if x1 != 'nan':
            return utils.get_date(x1)
        return utils.get_date(x2)
    def cert_name(x):
        if '高新技术' in x:
            return '高新技术'
        if '建筑施工' in x:
            return '建筑施工'
        return '其他'
    prefix = '资质登记（变更）信息'
    df, train_num = utils.get_df(prefix, ['企业名称', '资质名称', '资质生效期', '资质截止期', '认定日期'])
    df[prefix + '_资质名称'] = df[prefix + '_资质名称'].astype('str').apply(cert_name)
    df = pd.get_dummies(df, prefix=['dummy_' + prefix + '_资质名称'], columns=[prefix + '_资质名称'])
    df[prefix + '_资质生效期'] = df[[prefix + '_资质生效期', prefix + '_认定日期']].astype('str').apply(
        lambda x: date_proc(x[prefix + '_资质生效期'], x[prefix + '_认定日期']), axis=1)
    df[prefix + '_资质截止期'] = df[prefix + '_资质截止期'].astype('str').apply(
        lambda x: utils.get_date(x) if x != 'nan' and x[:4] != '1950' else np.nan)
    df[prefix + '_资质生效期_资质截止期_diff'] = df[prefix + '_资质生效期'] - df[prefix + '_资质截止期']
    raw_features = df.columns.values[1:]
    group = df.groupby('企业名称', as_index=False)
    for f in raw_features:
        if 'dummy' in f:
            df = pd.merge(df, utils.get_agg(group, f, ['sum', 'mean']), on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_资质生效期', ['max', 'min', 'mean']),
                  on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_资质截止期', ['min']), on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_资质生效期_资质截止期_diff', ['min']),
                  on='企业名称', how='left')
    for f in raw_features:
        del df[f]
    train_df, test_df = df[:train_num], df[train_num:]
    train_df.drop_duplicates('企业名称', inplace=True)
    test_df.drop_duplicates('企业名称', inplace=True)
    return prefix, train_df, test_df


def fe10():
    prefix = '企业表彰荣誉信息'
    df, train_num = utils.get_df(prefix, ['企业名称', '认定日期'])
    df[prefix + '_认定日期'] = df[prefix + '_认定日期'].apply(utils.get_date)
    group = df.groupby('企业名称', as_index=False)
    df = pd.merge(df, group['企业名称'].agg({prefix + '_count': 'count'}), on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_认定日期', ['max']), on='企业名称', how='left')
    for f in ['认定日期']:
        del df[prefix + '_' + f]
    train_df, test_df = df[:train_num], df[train_num:]
    train_df.drop_duplicates('企业名称', inplace=True)
    test_df.drop_duplicates('企业名称', inplace=True)
    return prefix, train_df, test_df


def fe11():
    prefix = '双打办打击侵权假冒处罚案件信息'
    df, train_num = utils.get_df(prefix, ['企业名称', '提供日期'])
    df[prefix + '_提供日期'] = df[prefix + '_提供日期'].astype('str').apply(
        lambda x: utils.get_date(x) if x != 'nan' else np.nan)
    group = df.groupby('企业名称', as_index=False)
    df = pd.merge(df, group['企业名称'].agg({prefix + '_count': 'count'}), on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_提供日期', ['max']), on='企业名称', how='left')
    for f in ['提供日期']:
        del df[prefix + '_' + f]
    train_df, test_df = df[:train_num], df[train_num:]
    train_df.drop_duplicates('企业名称', inplace=True)
    test_df.drop_duplicates('企业名称', inplace=True)
    return prefix, train_df, test_df


def fe12():
    prefix = '法人行政许可注（撤、吊）销信息'
    df, train_num = utils.get_df(prefix, ['企业名称', '注（撤、吊）销批准日期'])
    df[prefix + '_注（撤、吊）销批准日期'] = df[prefix + '_注（撤、吊）销批准日期'].apply(utils.get_date)
    group = df.groupby('企业名称', as_index=False)
    df = pd.merge(df, group['企业名称'].agg({prefix + '_count': 'count'}), on='企业名称', how='left')
    df = pd.merge(df, utils.get_agg(group, prefix + '_注（撤、吊）销批准日期', ['max']), on='企业名称', how='left')
    for f in ['注（撤、吊）销批准日期']:
        del df[prefix + '_' + f]
    train_df, test_df = df[:train_num], df[train_num:]
    train_df.drop_duplicates('企业名称', inplace=True)
    test_df.drop_duplicates('企业名称', inplace=True)
    return prefix, train_df, test_df


def fe():
    label_df11 = pd.read_csv(open('data/round1/train/失信被执行人名单.csv', encoding='utf-8'))
    label_df12 = pd.read_csv(open('data/round2/train/失信被执行人名单.csv', encoding='utf-8'))
    label_df1 = pd.concat([label_df11, label_df12], axis=0, ignore_index=True)
    label_df1['label1'] = 1
    label_df21 = pd.read_csv(open('data/round1/train/双公示-法人行政处罚信息.csv', encoding='utf-8'))
    label_df22 = pd.read_csv(open('data/round2/train/双公示-法人行政处罚信息.csv', encoding='utf-8'))
    label_df22.drop(label_df22.columns.values[1:], axis=1, inplace=True)
    label_df2 = pd.concat([label_df21, label_df22], axis=0, ignore_index=True)
    label_df2['label2'] = 1

    print('fe1')
    prefix1, train_df1, test_df1 = fe1()
    print('fe2')
    prefix2, train_df2, test_df2 = fe2()
    print('fe3')
    prefix3, train_df3, test_df3 = fe3()
    print('fe4')
    prefix4, train_df4, test_df4 = fe4()
    print('fe5')
    prefix5, train_df5, test_df5 = fe5()
    print('fe6')
    prefix6, train_df6, test_df6 = fe6()
    print('fe7')
    prefix7, train_df7, test_df7 = fe7()
    print('fe8')
    prefix8, train_df8, test_df8 = fe8()
    print('fe9')
    prefix9, train_df9, test_df9 = fe9()
    print('fe10')
    prefix10, train_df10, test_df10 = fe10()
    print('fe11')
    prefix11, train_df11, test_df11 = fe11()
    print('fe12')
    prefix12, train_df12, test_df12 = fe12()

    test_df = pd.merge(test_df1, test_df2, on='企业名称', how='left')
    test_df = pd.merge(test_df, test_df3, on='企业名称', how='left')
    test_df = pd.merge(test_df, test_df4, on='企业名称', how='left')
    test_df = pd.merge(test_df, test_df5, on='企业名称', how='left')
    test_df = pd.merge(test_df, test_df6, on='企业名称', how='left')
    test_df = pd.merge(test_df, test_df7, on='企业名称', how='left')
    test_df = pd.merge(test_df, test_df8, on='企业名称', how='left')
    test_df = pd.merge(test_df, test_df9, on='企业名称', how='left')
    test_df = pd.merge(test_df, test_df10, on='企业名称', how='left')
    test_df = pd.merge(test_df, test_df11, on='企业名称', how='left')
    test_df = pd.merge(test_df, test_df12, on='企业名称', how='left')
    test_df[prefix5] = test_df[prefix5].fillna(0)

    df = pd.merge(train_df1, train_df2, on='企业名称', how='left')
    df = pd.merge(df, train_df3, on='企业名称', how='left')
    df = pd.merge(df, train_df4, on='企业名称', how='left')
    df = pd.merge(df, train_df5, on='企业名称', how='left')
    df = pd.merge(df, train_df6, on='企业名称', how='left')
    df = pd.merge(df, train_df7, on='企业名称', how='left')
    df = pd.merge(df, train_df8, on='企业名称', how='left')
    df = pd.merge(df, train_df9, on='企业名称', how='left')
    df = pd.merge(df, train_df10, on='企业名称', how='left')
    df = pd.merge(df, train_df11, on='企业名称', how='left')
    df = pd.merge(df, train_df12, on='企业名称', how='left')
    df = pd.merge(df, label_df1, on='企业名称', how='left')
    df = pd.merge(df, label_df2, on='企业名称', how='left')
    df[prefix5] = df[prefix5].fillna(0)
    df['label1'] = df['label1'].fillna(0)
    df['label2'] = df['label2'].fillna(0)
    labels2 = df['label2'].values

    df[prefix1 + '_成立日期_' + prefix3
       + '_发证日期_diff'] = df[prefix1 + '_成立日期'] - df[prefix3 + '_发证日期']
    test_df[prefix1 + '_成立日期_' + prefix3
            + '_发证日期_diff'] = test_df[prefix1 + '_成立日期'] - test_df[prefix3 + '_发证日期']
    train_num = df.shape[0]
    df = pd.concat([df, test_df], axis=0, ignore_index=True)
    f_pairs = [[prefix1 + '_住所所在地省份', prefix3 + '_行政区划']]
    df = utils.get_ratio(df, f_pairs)
    del df['label1'], df['label2']
    test_df = df[train_num:]
    df = df[:train_num]
    train_names = df['企业名称']
    test_names = test_df['企业名称']
    del df['企业名称'], test_df['企业名称']
    return df, train_names, labels2, test_df, test_names
