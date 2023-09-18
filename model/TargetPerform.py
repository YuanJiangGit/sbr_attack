# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function:

from model.AttackLTRWES import AttackLTRWES
from model.AttackFARSEC import AttackFARSEC
from dataprocess.LTRWESDataProcess import LTRWESParseData
from dataprocess.FARSECDataProcess import FARSEParseData

def LTRWES_evaluation():
    # project_best_classifier = {'ambari': 'LRCV', 'camel': 'MLP', 'derby': 'MLP', 'chromium': 'LR', 'wicket': 'MLPCV'}
    project = 'ambari'
    learning_method = 'LRCV'
    sbr_attack = AttackLTRWES(project, learning_method, 'HTPsAttackV1', 'LTRWES')
    sbr_attack.pretrain_model()
    dataPipline = LTRWESParseData()
    # load testing dataset
    df_test = dataPipline.load_data(project, tag='testing')
    df_test = df_test.sample(n=500, random_state=1)
    sbr_attack.pretrain_model_test(df_test)


def FARSEC_evaluation():
    # {'ambari': 'RF', 'camel': 'MLP', 'derby': 'NB', 'wicket': 'MLP'}
    project = 'chromium'
    sbr_attack = AttackFARSEC(project, 'NB', 'PWWSRen2019', 'FARSEC')
    _ = sbr_attack.load_sbr_words()
    sbr_attack.pretrain_model()
    dataPipline = FARSEParseData()
    # load testing dataset
    df_test = dataPipline.load_data(project, tag='testing')
    sbr_attack.pretrain_model_test(df_test)

def shuffle_test():
    project = 'ambari'
    for i in range(10):
        dataPipline = FARSEParseData()
        # load testing dataset
        df_train = dataPipline.load_data(project, tag='training')
        issue_id = df_train['issue_id']

if __name__ == '__main__':
    LTRWES_evaluation()
    # FARSEC_evaluation()
    # shuffle_test()