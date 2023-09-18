# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function:
import os

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import nltk
import re
from dataprocess.DataProcess import ParseData


class FARSEParseData(ParseData):
    def __init__(self):
        super(FARSEParseData, self).__init__()
        self.training_dir= '../resources/FARSEC_training/'
        self.testing_dir = '../resources/FARSEC_testing/'
        self.training_origin_dir = "../resources/FARSEC_training/origin"
        self.top_words =None

    @staticmethod
    def clean_pandas(data):
        data.rename(columns=lambda x: x.strip(), inplace=True)
        data['summary'] = data.summary.apply(ParseData.dealNan)
        data['description'] = data.description.apply(ParseData.dealNan)
        # 对文本数据进行清洗
        data['summary'] = data['summary'].map(lambda x: FARSEParseData.preprocess_br(x))
        data['description'] = data['description'].map(lambda x: FARSEParseData.preprocess_br(x))
        return data

    @staticmethod
    def clean_defense_pandas(data):
        data.rename(columns=lambda x: x.strip(), inplace=True)
        data['text'] = data['text'].apply(ParseData.dealNan)
        # 对文本数据进行清洗
        data['text'] = data['text'].map(lambda x: FARSEParseData.preprocess_br(x))
        return data

    @staticmethod
    def is_unwanted(word):
        '''
        :param word: 需要判断的word
        :return: True represent the word is unwanted
        '''
        # 是否标点符号
        is_punctuation = re.search("(?i)[\w]+", word)
        if is_punctuation == None:
            return True
        # 是否字母数字
        is_alphanumeric = re.search("\d+", word)
        if is_alphanumeric:
            return True
        # 是否包含下划线
        is_underscore = re.search("_", word)
        if is_underscore:
            return True
        # 是否包含url
        is_url = re.match("//", word)
        if is_url:
            return True
        is_unletters = re.search("[^a-zA-Z]", word)
        if is_unletters:
            return True
        return False

    @staticmethod
    def unwanted_words(word):
        '''
        Removes unwanted keywords, such as urls, alphnumeric, underscores, punctuation, stopwords and single characters.
        :param word:
        :return: if word is meaningful, then return True else return False
        '''
        # Remove stop words
        stopwords = set(nltk.corpus.stopwords.words('english'))
        if word in stopwords:
            return False
        # Remove unwanted word
        if FARSEParseData.is_unwanted(word):
            return False
        # Remove the length of word less than 3
        if len(word) < 3:
            return False
        return True

    # Processing the text of BugReport
    @staticmethod
    def preprocess_br(raw_description):
        # print(raw_description)
        # 1. Tokenize
        current_desc_tokens = nltk.word_tokenize(raw_description)
        # 2. normalize (Change to lower case)
        lower_desc_tokens = [w.lower() for w in current_desc_tokens]
        # 3. unwanted words
        meaningful_words = [w for w in lower_desc_tokens if FARSEParseData.unwanted_words(w)]
        return meaningful_words

    def load_data(self, project, tag='training', radio=None,
                  baseline=False, method="HTPsAttackV1", test_accu=False,
                  model_name=None, epoch=0, all=True):
        if tag == 'training':
            for file in os.listdir(self.training_dir):
                if project in file:
                    train_path = os.path.join(self.training_dir, file)
                    df_train = pd.read_csv(train_path)
                    if project == "chromium":
                        df_sbr = df_train[df_train['Security'] == 1]
                        df_nsbr = df_train[df_train['Security'] == 0]
                        df_train = df_sbr.append(df_nsbr.sample(n=int(200), random_state=1))
                    else:
                        df_train = shuffle(df_train, random_state=1)
                    return df_train

        if tag == 'testing':
            if test_accu:
                # 原始测试集上的精度
                for file in os.listdir(self.testing_dir):
                    if project in file:
                        test_path = os.path.join(self.testing_dir, file)
                        df_test = pd.read_csv(test_path)
                        return df_test
            else:
                # 从测试集中用于生成对抗样本
                data_file = os.path.join(self.dataset_dir, self.dataset_dict[project])
                df_all = pd.read_csv(data_file, sep=',', encoding='ISO-8859-1')
                # specialized processing with chromium
                if project == 'chromium':
                    path = os.path.join(self.dataset_dir, 'Chromium2.csv')
                    if not os.path.exists(path):
                        df_all['summary'] = df_all.apply(lambda x: self.split_report(x.report, 'summary'), axis=1)
                        df_all['description'] = df_all.apply(lambda x: self.split_report(x.report, 'description'), axis=1)
                        df_all.to_csv(path, encoding='utf-8')
                    else:
                        df_all = pd.read_csv(path)
                df_test = pd.DataFrame(df_all, index=range(int(len(df_all) / 2), len(df_all)))
                # df_test = pd.DataFrame(df_all)
                df_test['text']=df_all.apply(lambda x: str(x['summary'])+' '+str(x['description']), axis=1)
                df_test['text'] = df_test.apply(lambda x: FARSEParseData.clip_longer_text(x['text']), axis=1)
                return df_test

        # 预处理-提取安全关键字
        if tag == 'pre-training':
            data_file = os.path.join(self.dataset_dir, self.dataset_dict[project])
            df_all = pd.read_csv(data_file, sep=',', encoding='ISO-8859-1')

            if project == 'chromium':
                path = os.path.join(self.dataset_dir, 'Chromium2.csv')
                if not os.path.exists(path):
                    df_all['summary'] = df_all.apply(lambda x: self.split_report(x.report, 'summary'), axis=1)
                    df_all['description'] = df_all.apply(lambda x: self.split_report(x.report, 'description'), axis=1)
                    df_all.to_csv(path, encoding='utf-8')
                else:
                    df_all = pd.read_csv(path)

            df_training = pd.DataFrame(df_all)
            df_training['text'] = df_all.apply(lambda x: str(x['summary']) + ' ' + str(x['description']), axis=1)
            df_training['text'] = df_training.apply(lambda x: FARSEParseData.clip_longer_text(x['text']), axis=1)
            return df_training

        # 防御——用于从训练集中生成的对抗样本中提取数据进行模型重训练
        if tag == 'defense':
            df_sbr, df_nsbr  = [], []
            for one in range(epoch+1):
                file_name = self.result_file(['FARSEC', project, model_name, 'HTPsAttackV1', '.csv'], epoch=one)
                csv_path = f"{self.defense_dataset_dir}/epoch{one}/{file_name}"
                df = pd.read_csv(csv_path)
                df = df[df['result_type'] == "Successful"]
                df["text"] = df.apply(lambda row: row["perturbed_text"].replace("[", "").replace("]", ""),
                                      axis=1)
                df['Security'] = df['ground_truth_output']
                temp_sbr = df[df['ground_truth_output'] == 1]
                temp_nsbr = df[df['ground_truth_output'] == 0]
                if len(df_sbr) == 0:
                    df_sbr = temp_sbr
                    df_nsbr = temp_nsbr[:int(len(temp_nsbr) * radio)]
                else:
                    df_sbr = df_sbr.append(temp_sbr)
                    if radio != 1.1:
                        df_nsbr = df_nsbr.append(temp_nsbr[:int(len(temp_nsbr) * radio)])
            if radio == 1.1:
                df_nsbr = []
            print("radio=", str(radio), "sbrs number=", str(len(df_sbr)),
                  "nsbrs number=", str(len(df_nsbr)), "===================\n")
            # if project == 'derby' and not all:
            #     file_name1 = "E:\\postgraduate\\research\\attack\\sbr_attack\\resources\\attack_results\\derby_temp\\train-attackderby.csv"
            #     df1 = pd.read_csv(file_name1)
            #     df1 = df1[df1['result_type'] == "Successful"]
            #     df1 = df1[df1['ground_truth_output'] == 0]
            #     df1["text"] = df1.apply(lambda row: row["perturbed_text"].replace("[", "").replace("]", ""),
            #                           axis=1)
            #     df1['Security'] = df1['ground_truth_output']
            #     df_nsbr = df_nsbr.append(df1)
            #     df_nsbr = df_nsbr.sample(n=int(len(df_nsbr) * radio), random_state=1)
            #     print("添加的NSBRs的对抗样本为", len(df_nsbr))
            #     return df_nsbr
            return df_sbr.append(df_nsbr)

        # 防御——用于获取可攻击的处理好的训练集样本
        if tag == 'adv-training':
            csv_path = os.path.join(self.training_origin_dir, self.dataset_dict[project])
            if project == 'chromium':
                csv_path = os.path.join(self.training_dir, "origin", "temp_chromium.csv")
            df = pd.read_csv(csv_path, sep=',')
            return df

        # 防御——检测重训练模型在对抗样本数据集上的检测精度，是否更加精确
        if tag == 'attack-test':
            if baseline:
                file_name = self.result_file(['FARSEC', project, method, '.csv'], dir=self.baseline_attack_dataset_dir)
                csv_path = f"{self.baseline_attack_dataset_dir}/{file_name}"
            else:
                if project == 'chromium':
                    csv_path = f"{self.attack_dataset_dir}/test/farsec_adv_onehot.csv"
                    df = pd.read_csv(csv_path, sep=',')
                    return df
                else:
                    file_name = self.result_file(['FARSEC', project, method, '.csv'],
                                                 dir=self.attack_dataset_dir+'/test')
                    csv_path = f"{self.attack_dataset_dir}/test/{file_name}"
            df = pd.read_csv(csv_path, sep=',')
            df = df[df['result_type'] == "Successful"]
            df["text"] = df.apply(lambda row: row["perturbed_text"].replace("[", "").replace("]", ""),
                                  axis=1)
            df['Security'] = df['ground_truth_output']
            return df

    @staticmethod
    def clip_longer_text(text):
        words=text.split(' ')
        return ' '.join(words[:200])

    @staticmethod
    def one_report(report, top_words, attack=False):
        '''
        convert report into the binary form
        :param report:
        :param top_words:
        :return:
        '''
        dict = {term: 0 for term in top_words}
        if attack:
            text = report['text']
        else:
            # 0 就是issue_id
            dict['issue_id'] = report[0]
            text = report['summary'] + report['description']
        for term in text:
            if term in top_words:
                dict[term] += 1
        dict['Security'] = report['Security']
        return dict

    @staticmethod
    def make_data_by_topwords(pdList, top_words, attack=False):
        '''
        combine dataset in pdList and convert each instance according to top_words
        :param pdList:
        :param top_words:
        :return:
        '''
        if attack:
            columns = top_words + ['Security']
        else:
            columns = ['issue_id'] + top_words + ['Security']
        data = pd.DataFrame(columns=columns)
        for corpus in pdList:
            for i in range(len(corpus)):
                report_var = FARSEParseData.one_report(corpus.iloc[i], top_words, attack)
                data = data.append(report_var, ignore_index=True)
        return data

    @staticmethod
    def process_train_data_to_tuplexy(df_train):
        df_train = df_train.dropna()
        df_train = df_train.astype('int64')
        x_train = df_train.iloc[:, 1:-1]
        y_train = df_train.Security
        return x_train, y_train

    @staticmethod
    def process_text_data_to_tuplexy(data, sbr_words, attack=False):
        # attack代表是在测试集\测试集的对抗样本上进行预处理
        # 在测试集上进行测试的是攻击精度/在对抗样本上测试的是防御精度
        if attack:
            data = FARSEParseData.clean_defense_pandas(data)
            df_data = FARSEParseData.make_data_by_topwords([data], sbr_words, attack)
            x_data = df_data.iloc[:, :-1]
        else:
            data = FARSEParseData.clean_pandas(data)
            df_data = FARSEParseData.make_data_by_topwords([data], sbr_words, attack)
            df_data['issue_id'] = df_data['issue_id'].astype('int64')
            x_data = df_data.iloc[:, 1:-1]
        df_data['Security'] = df_data['Security'].astype('float64')
        y_data = df_data.Security
        y_data.reset_index()
        return x_data, y_data
