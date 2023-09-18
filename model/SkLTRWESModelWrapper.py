# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function:

import pandas as pd
import numpy as np
from textattack.models.wrappers import ModelWrapper
from Config.ConfigT import MyConf
from dataprocess.LTRWESDataProcess import LTRWESParseData


class SklearnModelWrapper(ModelWrapper):
    """Loads a scikit-learn model and tokenizer (tokenizer implements
    `transform` and model implements `predict_proba`).

    May need to be extended and modified for different types of
    tokenizers.
    """
    def __init__(self, model, word2vec_model):
        self.model = model
        self.word2vec_model = word2vec_model
        # self.vocab=self.word2vec_model.vocab
        self.config=MyConf('../Config/config.cfg')

    def to_review_vector(self, words):
        array = np.array([self.word2vec_model.wv[w] for w in words if w in self.word2vec_model.wv])
        return pd.Series(array.mean(axis=0))

    def transform(self,text_input_list):
        text_vec_df = pd.DataFrame()
        for text in text_input_list:
            word_list= LTRWESParseData.preprocess_br(text)
            text_array=self.to_review_vector(word_list)
            text_vec_df = text_vec_df.append(text_array,ignore_index=True)
        return text_vec_df

    def __call__(self, text_input_list):
        encoded_text_matrix = self.transform(text_input_list)
        return self.model.predict_proba(encoded_text_matrix)

    def get_grad(self, text_input):
        raise NotImplementedError()
