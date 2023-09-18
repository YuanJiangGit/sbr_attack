import json
import math
import os
from collections import defaultdict

from Utils.ExtractSecurityWord import ExtractSecurityWordUtil
from dataprocess.DataProcess import ParseData
import operator
import pandas as pd


class SplitSecurityWordUtil:
    def __init__(self, project, model_type):
        self.project = project
        self.model_type = model_type

    def load_corpus(self, df_train):
        '''
        加载语料库
        :param df_train:
        :return:
        '''
        self.SBR = df_train[df_train['Security'] == 1]
        self.NSBR = df_train[df_train['Security'] == 0]
        SBRs = self.SBR['summary'] + self.SBR['description']
        NSBRs = self.NSBR['summary'] + self.NSBR['description']
        sbr_corpus = []
        nsbr_corpus = []
        # 安全权重高的词位置
        self.sbr_path = "..\\resources\\sbr_words\\" + self.project + "_" + self.model_type
        # 非安全权重高的词位置
        self.nsbr_path = "..\\resources\\nsbr_words\\" + self.project + "_" + self.model_type

        for index in range(len(SBRs)):
            term_freq = defaultdict(int)
            SBR = SBRs.iloc[index]
            for term in str(SBR).split(' '):
                term_freq[term] += 1
            sbr_corpus.append(term_freq)

        for index in range(len(NSBRs)):
            term_freq = defaultdict(int)
            NSBR = NSBRs.iloc[index]
            for term in str(NSBR).split(' '):
                term_freq[term] += 1
            nsbr_corpus.append(term_freq)

        return sbr_corpus, nsbr_corpus

    def idf(self, corpus, term):
        '''
        计算term的idf值
        :param corpus:
        :param term:
        :return:
        '''
        N=len(corpus)
        contain_term_nsbrs=[NSBR for NSBR in corpus if term in NSBR]
        # 包含terms的sbr的个数
        N_t = len(contain_term_nsbrs)
        idf_t = math.log(N / (N_t + 1))
        return idf_t

    def get_idf_cache(self, corpus, top_words):
        '''
        计算corpus中top_words的idf值
        '''
        idf_cache = {term:self.idf(corpus, term) for term in top_words}
        return idf_cache

    def tf_idf(self, idf_value, term_freq, len):
        return term_freq / len * idf_value

    def tf_idf_pair(self, idf_cache, pair):
        '''
        计算每个缺陷报告的tf_idf值
        :param idf_cache:
        :param pair:
        :return:
        '''
        terms_tf_idf = {}
        for term, freq in pair.items():
            if idf_cache.__contains__(term):
                idf_value = idf_cache[term]
                tf_idf_value = self.tf_idf(idf_value, freq, len(pair))
                terms_tf_idf[term] = tf_idf_value
        return terms_tf_idf

    def split_security_words(self, sbr_corpus, nsbr_corpus, top_words):
        '''
        计算top_words在sbr_corpus和nsbr_corpus语料库中的权重比例，排序进行比较，将其拆分为
        安全权重高或者是非安全权重高的单词，分别存入文件中
        :param sbr_corpus:
        :param nsbr_corpus:
        :param top_words:
        :return:
        '''
        sbr_idf_cache = self.get_idf_cache(sbr_corpus, top_words)
        sbr_corpus_tf_idf = defaultdict(float)
        for pair in sbr_corpus:
            tf_idf_ = self.tf_idf_pair(sbr_idf_cache, pair)
            for term, tf_idf in tf_idf_.items():
                sbr_corpus_tf_idf[term] += tf_idf
        sbr_corpus_tf_idf = dict(sorted(sbr_corpus_tf_idf.items(), key=operator.itemgetter(1), reverse=True))
        # 计算权重和
        sbr_sum = sum(sbr_corpus_tf_idf.values())

        nsbr_idf_cache = self.get_idf_cache(nsbr_corpus, top_words)
        nsbr_corpus_tf_idf = defaultdict(float)
        for pair in nsbr_corpus:
            tf_idf_ = self.tf_idf_pair(nsbr_idf_cache, pair)
            for term, tf_idf in tf_idf_.items():
                nsbr_corpus_tf_idf[term] += tf_idf
        nsbr_corpus_tf_idf = dict(sorted(nsbr_corpus_tf_idf.items(), key=operator.itemgetter(1), reverse=True))
        # 计算权重和
        nsbr_sum = sum(nsbr_corpus_tf_idf.values())

        sbr_words = []
        nsbr_words = []
        common_keys = sbr_corpus_tf_idf.keys() & nsbr_corpus_tf_idf.keys()
        # 公有key按照权重比例区分
        if len(common_keys) != 0:
            for common_key in common_keys:
                sbr_key_value = sbr_corpus_tf_idf[common_key]
                nsbr_key_value = nsbr_corpus_tf_idf[common_key]
                if sbr_key_value / sbr_sum > nsbr_key_value / nsbr_sum:
                    sbr_words.append(common_key)
                else:
                    nsbr_words.append(common_key)
        # 非公有key直接加入对应列表
        sbr_keys = sbr_corpus_tf_idf.keys() - nsbr_corpus_tf_idf.keys()
        if len(sbr_keys) != 0:
            sbr_words.extend(sbr_keys)
        nsbr_keys = nsbr_corpus_tf_idf.keys() - sbr_corpus_tf_idf.keys()
        if len(nsbr_keys) != 0:
            nsbr_words.extend(nsbr_keys)
        return sbr_words, nsbr_words

    def write_word(self, sbr_words, nsbr_words):
        with open(self.sbr_path, 'w') as f:
            for sbr_word in sbr_words:
                f.write(sbr_word + ' ')
        with open(self.nsbr_path, 'w') as f:
            for nsbr_word in nsbr_words:
                f.write(nsbr_word + ' ')


if __name__ == '__main__':
    # FARSEC
    # project_best_classifier = {'ambari': 'RF', 'camel': 'MLP', 'derby': 'NB', 'chromium': 'NB', 'wicket': 'MLP'}
    # LTRWES
    # project_best_classifier = {'ambari': 'LRCV', 'camel': 'MLP', 'derby': 'MLP', 'chromium': 'LR', 'wicket': 'MLPCV',}
    project = 'wicket'
    learning_method = 'MLPCV'
    parse_data = ParseData()
    df_all = pd.read_csv(os.path.join(parse_data.dataset_dir, parse_data.dataset_dict[project]), encoding='utf-8')
    # split_security_words_util = SplitSecurityWordUtil(project, "FARSEC")
    split_security_words_util = SplitSecurityWordUtil(project, "LTRWES")
    sbr_corpus, nsbr_corpus = split_security_words_util.load_corpus(df_all)
    # 之前提取出的关键词
    # security_word_util = ExtractSecurityWordUtil(project, learning_method, 'FARSEC')
    security_word_util = ExtractSecurityWordUtil(project, learning_method, 'LTRWES')
    top_words = security_word_util.load_top_words()
    # 拆分关键词，按照tf-idf比较划分为安全相关和非安全相关的
    sbr_words, nsbr_words = split_security_words_util.split_security_words(sbr_corpus, nsbr_corpus, top_words)
    # 写入文件
    # split_security_words_util.write_word(sbr_words, nsbr_words)
    print(sbr_words, len(sbr_words))
    print(nsbr_words, len(nsbr_words))
