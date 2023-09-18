# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function: attack the state-of-the-art SBR detection model (LTRWES)

import os
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
import tensorflow as tf
from ResultLogger.MyLoggerManager import *
from model.SkLTRWESModelWrapper import SklearnModelWrapper

from model.AttackT import AttackT
from Config.ConfigT import MyConf
from dataprocess.LTRWESDataProcess import LTRWESParseData


class AttackLTRWES(AttackT):
    def __init__(self, project, model_name, recipe, model_type):
        super(AttackLTRWES, self).__init__(project, model_name, recipe, model_type)
        self.word2vec_model = None
        self.EMBEDDING_Dir = os.path.join('../resources/word2vec', '200features_5minwords_5context.model')

    def wordEmbedding(self):
        # sigle model predict
        word2vec_model = Word2Vec.load(self.EMBEDDING_Dir)
        self.word2vec_model=word2vec_model
        return word2vec_model

    def to_review_vector(self, words, model):
        array = np.array([model.wv[w] for w in words if w in model.wv])
        return pd.Series(array.mean(axis=0))

    def pretrain_model(self, defense=False, epoch=0, radio=None):
        '''
        1 load target model
        :return:
        '''
        word2vec_model = self.wordEmbedding()
        if defense:
            target_model_path = os.path.join(self.Defense_TargetModel_PATH,
                                             "epoch"+str(epoch), self.project + '_' + self.model_name+'_'+str(radio))
        else:
            target_model_path = os.path.join(self.TargetModel_PATH, self.project + '_' + self.model_name)
        print("target_model_path=", target_model_path)
        if os.path.exists(target_model_path):
            self.target_model._load(target_model_path)
        else:
            df_train = self.dataPipline.load_data(self.project, tag='training')
            x_train = df_train.apply(lambda x: self.to_review_vector(eval(x.summary) + eval(x.description), word2vec_model),
                                     axis=1)
            y_train = df_train.Security
            self.target_model._train(x_train, y_train)
            self.target_model._save(target_model_path)
        model = self.target_model.object.model
        return model

    def pretrain_model_test(self, df_test, attack=False):
        '''
        test target model
        :param df_test:
        :return:
        '''
        if self.word2vec_model==None:
            word2vec_model=self.wordEmbedding()
        else:
            word2vec_model=self.word2vec_model
            # clean textual fields of bug reports in df_test
        if attack:
            df_test = LTRWESParseData.clean_defense_pandas(df_test)
            x_test = df_test.apply(lambda x: self.to_review_vector(x.text, word2vec_model),
                                   axis=1)
        else:
            df_test = LTRWESParseData.clean_pandas(df_test)
            x_test = df_test.apply(lambda x: self.to_review_vector(x.summary + x.description, word2vec_model),
                                 axis=1)
        y_test = df_test.Security
        self.target_model._test(x_test, y_test)


    def modelAttack(self, defense=False, epoch=0, radio=None):
        config = MyConf('../Config/config.cfg')
        config.model = self.model_name
        config.recipe = self.recipe
        config.project = self.project
        config.model_type =self.model_type
        # determine whether attack results have been existed
        # if self.isExist(config):
        #     return

        # load models
        word2vec_model = self.wordEmbedding()
        target_model = self.pretrain_model(defense=defense, epoch=epoch, radio=radio)
        model_wrapper = SklearnModelWrapper(target_model, word2vec_model)
        # obtain attack
        attack= self.AttacK_Recipes[self.recipe].build(model_wrapper)
        print(attack)

        # load testing dataset
        df_test = self.dataPipline.load_data(self.project, tag='testing')
        df_train = self.dataPipline.load_data(self.project, tag='training')
        if test_target:
            self.pretrain_model_test(df_test)
            self.pretrain_model_test(df_train)

        num_examples = config.num_examples if len(df_test)>config.num_examples else 500
        if self.project == 'chromium':
            df_sbr = df_test[df_test['Security']==1]
            df_nsbr = df_test[df_test['Security']==0]
            samples = df_sbr.append(df_nsbr.sample(n=num_examples, random_state=1))
        else:
            samples = df_test.sample(n=num_examples, random_state=1)
        # df_test = df_test.dropna()
        sample_list = samples.apply(lambda x: (x['text'], int(x['Security'])), axis=1).to_list() # convert into the custom dataset
        # sample_list.pop(564)
        # sample_list.pop(280)
        num_examples=len(sample_list)

        # perform attack
        results_iterable = attack.attack_dataset(sample_list, indices=range(num_examples))

        attack_log_manager = parse_logger_from_args(config, train=False)
        for result in results_iterable:
            attack_log_manager.log_result(result)
            print()
            print()
            print(result.__str__(color_method='ansi'))

        attack_log_manager.log_summary()
        attack_log_manager.flush()


if __name__ == '__main__':
    tf.enable_eager_execution(
        config=None,
        device_policy=None,
        execution_mode=None
    )
    # recipes=['PWWSRen2019', 'TextFoolerJin2019', 'TextBuggerLi2018','DeepWordBugGao2018','HTPsAttackV4']  #'HTPsAttackV4' is our method
    recipes = ['PWWSRen2019']
    project_best_classifier = {'ambari': 'LRCV', 'camel': 'MLP', 'derby': 'MLP', 'chromium': 'LR', 'wicket': 'MLPCV'}
    epoch=0
    test_target = True
    for key, value in project_best_classifier.items():
        for recipe in recipes:
            sbr_attack = AttackLTRWES(key, value, recipe, 'LTRWES')
            sbr_attack.modelAttack()
            test_target=False
