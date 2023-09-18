# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function:

import pandas as pd
import numpy as np
from textattack.models.wrappers import ModelWrapper
from gensim.models.word2vec import Word2Vec
from Config.ConfigT import MyConf
from dataprocess.FARSECDataProcess import FARSEParseData
import os


class SklearnModelWrapper(ModelWrapper):
    """Loads a scikit-learn model and tokenizer (tokenizer implements
    `transform` and model implements `predict_proba`).

    May need to be extended and modified for different types of
    tokenizers.
    """
    def __init__(self, model, sbr_words):
        self.model = model
        # self.vocab=self.word2vec_model.vocab
        self.sbr_words=sbr_words


    def one_report(self, text, top_words):
        '''
        convert text into the binary form
        :param text:
        :param top_words:
        :return:
        '''
        dict = {term: 0 for term in top_words}
        # 0 就是issue_id
        for term in text:
            if term in top_words:
                dict[term] += 1
        return dict


    def make_data_by_topwords(self, word_list, top_words):
        '''
        combine dataset in pdList and convert each instance according to top_words
        :param text_input_list:
        :param top_words:
        :return:
        '''
        columns = top_words
        data = pd.DataFrame(columns=columns)
        report_var = self.one_report(word_list, top_words)
        data = data.append(report_var, ignore_index=True)
        return data

    def transform(self,text_input_list):
        text_vec_df = pd.DataFrame()
        for text in text_input_list:
            word_list= FARSEParseData.preprocess_br(text)
            sbr_words = self.sbr_words
            text_array=self.make_data_by_topwords(word_list, sbr_words)
            text_vec_df = text_vec_df.append(text_array,ignore_index=True)
        return text_vec_df

    def __call__(self, text_input_list):
        encoded_text_matrix = self.transform(text_input_list)
        return self.model.predict_proba(encoded_text_matrix)

    def get_grad(self, text_input):
        raise NotImplementedError()
