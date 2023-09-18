import os
from collections import defaultdict

import tensorflow as tf

from dataprocess.FARSECDataProcess import FARSEParseData
from dataprocess.LTRWESDataProcess import LTRWESParseData
from model.AttackFARSEC import AttackFARSEC
from model.AttackLTRWES import AttackLTRWES
from model.SkFARSECModelWrapper import SklearnModelWrapper as SkFARSECModelWrapper
from model.SkLTRWESModelWrapper import SklearnModelWrapper as SkLTRWESModelWrapper


class ExtractSecurityWordUtil:
    def __init__(self, project, learning_method, model_name):
        self.top_words_dir = '..\\resources\\top_words\\'
        self.top_words = None
        # 数据集名称
        self.project = project
        # 目标模型名字，FARSEC，LTRWES
        self.model_name = model_name
        # 机器学习方法
        self.learning_method = learning_method
        # 安全关键词存储文件位置信息
        self.top_file_path = self.top_words_dir + self.project + '_' + self.model_name

    def pre_initialization_data(self):
        # load models
        if self.model_name == 'FARSEC':
            sbr_attack = AttackFARSEC(self.project, self.learning_method, 'HTPsAttack', self.model_name)
            sbr_words = sbr_attack.load_sbr_words()
            # 训练好的目标模型，带权重的那种
            self.target_model = sbr_attack.pretrain_model()
            # 缺陷报告嵌入表示方法
            self.model_wrapper = SkFARSECModelWrapper(self.target_model, sbr_words)
            self.df_training = sbr_attack.dataPipline.load_data(self.project, tag='pre-training')
        elif self.model_name == 'LTRWES':
            sbr_attack = AttackLTRWES(self.project, self.learning_method, 'HTPsAttack', self.model_name)
            # 训练好的目标模型，带权重的那种
            self.target_model = sbr_attack.pretrain_model()
            # 缺陷报告嵌入表示方法
            self.model_wrapper = SkLTRWESModelWrapper(sbr_attack.target_model, sbr_attack.word2vec_model)
            df_training = sbr_attack.dataPipline.load_data(self.project, tag='pre-training')
            self.df_training = LTRWESParseData.clean_pandas(df_training)
        # 数据集样例，限制为从安全缺陷报告中提取关键词
        self.samples = self.df_training[self.df_training['Security'] == 1]

    def load_top_words(self):
        '''
        加载模型和数据库提取出来的安全关键词；
        如果有直接从文件返回；
        否则第一次访问进行提取
        :return:
        '''
        self.pre_initialization_data()
        with open(self.top_file_path, 'r+') as f:
            self.top_words = f.read().split()
        if len(self.top_words) == 0:
            self.extract_top_keywords()
        return self.top_words

    def extract_top_keywords(self):
        '''
        从安全缺陷报告中提取安全关键词，利用删除一个词查模型下降分数排序得到
        :return:
        '''
        top_word_dict = defaultdict(lambda: 0.0)
        top_word_frequency = defaultdict(lambda: 0)
        for report in self.samples.iterrows():
            report = report[1]
            origin_text = report['text'].lower()
            origin_score = self.target_model.predict_proba(self.model_wrapper.transform([origin_text]))[0][1]
            if report['Security'] == 1 and origin_score > 0.5:
                text_split = origin_text.split(' ')
                leave_one_texts = [
                    self.delete_word_at_index(i, origin_text) for i in range(len(text_split))
                ]
                if len(leave_one_texts) != 0:
                    score = self.target_model.predict_proba(self.model_wrapper.transform(leave_one_texts))
                    for i in range(len(score)):
                        if origin_score - score[i][1] > 0:
                            top_word_dict[text_split[i]] += origin_score - score[i][1]
                            top_word_frequency[text_split[i]] += 1
        # 计算每个词的平均下降分数
        top_word_dict = {key:(top_word_dict[key] / top_word_frequency[key]) for key in top_word_dict}
        top_word_dict = sorted(top_word_dict.items(), key=lambda d:d[1], reverse=True)
        # 去除不想要的单词，比如urls, alphnumeric, underscores, punctuation, stopwords and single characters.
        top_word_list = [one[0] for one in top_word_dict if FARSEParseData.unwanted_words(one[0])]
        print(top_word_dict)
        print(top_word_list, len(top_word_list))
        with open(self.top_file_path, 'w') as f:
            for word in top_word_list:
                f.write(word+" ")
        self.top_words = top_word_list

    def delete_word_at_index(self, i, origin_text):
        text_split = origin_text.split(' ')
        new_str = ' '
        new_text = new_str.join(text_split[:i] + text_split[i+1:])
        return new_text


if __name__ == '__main__':
    tf.enable_eager_execution(
        config=None,
        device_policy=None,
        execution_mode=None
    )
    # FARSEC
    # project_best_classifier = {'ambari': 'RF', 'camel': 'MLP', 'derby': 'NB', 'chromium': 'NB', 'wicket': 'MLP'}
    # security_FARSEC_word_util = ExtractSecurityWordUtil('ambari', 'MLP', 'FARSEC')
    # print(security_FARSEC_word_util.load_top_words())
    # LTRWES
    # project_best_classifier = {'ambari': 'LRCV', 'camel': 'MLP', 'derby': 'MLP', 'chromium': 'LR', 'wicket': 'MLPCV'}
    security_LTRWES_word_util = ExtractSecurityWordUtil('ambari', 'LRCV', 'LTRWES')
    print(security_LTRWES_word_util.load_top_words())

