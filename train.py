import shixin_fe, chufa_fe
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def train(df, labels, test_df, sub_df, predict_label):
    lgb = LGBMClassifier(learning_rate=0.05, n_estimators=5000, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.7, random_state=2018)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    val_auc_list = []
    for i, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        train_df, train_labels = df.iloc[train_idx], labels[train_idx]
        val_df, val_labels = df.iloc[val_idx], labels[val_idx]
        print('fold', i)
        lgb.fit(train_df, train_labels, eval_set=[(train_df, train_labels), (val_df, val_labels)],
                eval_metric='auc', early_stopping_rounds=100, verbose=False)
        val_auc_list.append(lgb.best_score_['valid_1']['auc'])
        sub_df[predict_label] += lgb.predict_proba(test_df, num_iteration=lgb.best_iteration_)[:, 1]
    sub_df[predict_label] /= 5
    return np.mean(val_auc_list)


print('失信特征...')
shixin_df, _, shixin_labels, shixin_test_df, test_names = shixin_fe.fe()
shixin_df['label'] = shixin_labels
shixin_df = pd.concat([shixin_df, shixin_df.sample(n=10000, axis=0, random_state=233)], axis=0, ignore_index=True)
shixin_labels = shixin_df['label']
del shixin_df['label']
print('\n处罚特征...')
chufa_df, _, chufa_labels, chufa_test_df, _ = chufa_fe.fe()
sub_df = pd.DataFrame({'EID': test_names,
                       'FORTARGET1': [0] * shixin_test_df.shape[0], 'FORTARGET2': [0] * shixin_test_df.shape[0]})


print('\n失信预测...')
shixin_val_auc_mean = train(shixin_df, shixin_labels, shixin_test_df, sub_df, 'FORTARGET1')
print('\n失信 val auc mean:', shixin_val_auc_mean)
print('\n处罚预测...')
chufa_val_auc_mean = train(chufa_df, chufa_labels, chufa_test_df, sub_df, 'FORTARGET2')
print('\n处罚 val auc mean:', chufa_val_auc_mean)

val_auc_mean = (shixin_val_auc_mean + chufa_val_auc_mean) / 2
print('\nval auc mean:', val_auc_mean)

sub_df.to_csv('result/compliance_assessment_{}.csv'.format(val_auc_mean), index=False)
