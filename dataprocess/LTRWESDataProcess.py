# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function: process data
import os
import pandas as pd
import re
from dataprocess.DataProcess import ParseData
import nltk
from nltk.stem.snowball import SnowballStemmer

from evaluation.test import USEMetric

class LTRWESParseData(ParseData):
    def __init__(self):
        super(LTRWESParseData, self).__init__()
        self.training_dir= '../resources/LTRWES_training/'
        self.training_origin_dir = "../resources/LTRWES_training/origin"
        self.use_metric = USEMetric()

    @staticmethod
    def preprocess_br(raw_description):
        '''
        Processing the text of BugReport
        :param raw_description:
        :return:
        '''
        # 1. Remove \r
        current_desc = raw_description.replace('\r', ' ')
        # 2. Remove URLs
        current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                              current_desc)
        # 3. Remove Stack Trace
        start_loc = current_desc.find("Stack trace:")
        current_desc = current_desc[:start_loc]
        # 3.5 Remove Issue id : (only exists in chromium, others project can commented out)
        current_desc = re.sub(r'Issue (.*) : ', '', current_desc)
        # 4. Remove hex code
        current_desc = re.sub(r'(\w*)0x\w+', '', current_desc)
        # 5. Remove code snippet
        # current_desc=remove_code(current_desc)
        # 6. Change to lower case
        current_desc = current_desc.lower()
        # 7. only letters
        letters_only = re.sub("[^a-zA-Z\.]", " ", current_desc)
        current_desc = re.sub("\.(?!((c|h|cpp|py)\s+$))", " ", letters_only)
        # 8. Tokenize
        current_desc_tokens = nltk.word_tokenize(current_desc)
        # 9. Remove stop words
        stopwords = set(nltk.corpus.stopwords.words('english'))
        meaningful_words = [w for w in current_desc_tokens if not w in stopwords]
        # 10. Stemming
        snowball = SnowballStemmer("english")
        stems = [snowball.stem(w) for w in meaningful_words]
        return stems

    @staticmethod
    def clean_pandas(data):
        data['summary'] = data.summary.apply(ParseData.dealNan)
        data['description'] = data.description.apply(ParseData.dealNan)
        # 对文本数据进行清洗
        data['summary'] = data['summary'].map(lambda x: LTRWESParseData.preprocess_br(x))
        data['description'] = data['description'].map(lambda x: LTRWESParseData.preprocess_br(x))
        return data

    @staticmethod
    def clean_defense_pandas(data):
        data['text'] = data.text.apply(ParseData.dealNan)
        data['text'] = data['text'].map(lambda x: LTRWESParseData.preprocess_br(x))
        return data

    def load_data(self, project, tag='training',
                  model_name=None, radio=None,
                  baseline=False, method="HTPsAttackV1", epoch=0):
        # 训练——预处理好的训练集
        if tag == 'training':
            training_path=os.path.join(self.training_dir,project+'.csv')
            if os.path.exists(training_path):
                df_all = pd.read_csv(training_path,sep=",")
                return df_all

        # 训练——测试集
        if tag == 'testing':
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
            df_test['text']=df_test.apply(lambda x: LTRWESParseData.clip_longer_text(x['text']), axis=1)
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
            df_training['text'] = df_training.apply(lambda x: LTRWESParseData.clip_longer_text(x['text']), axis=1)
            return df_training

        # 防御——用于从训练集中生成的对抗样本中提取数据进行模型重训练
        if tag == 'defense':
            df_sbr, df_nsbr = [], []
            for one in range(epoch+1):
                file_name = self.result_file(['LTRWES', project, model_name, 'HTPsAttackV1', '.csv'], epoch=one)
                csv_path = f"{self.defense_dataset_dir}/epoch{one}/{file_name}"
                df = pd.read_csv(csv_path)
                df = df[df['result_type'] == "Successful"]
                df["text"] = df.apply(lambda row: row["perturbed_text"].replace("[", "").replace("]", ""),
                                               axis=1)
                df['Security'] = df['ground_truth_output']
                temp_sbr = df[df['ground_truth_output'] == 1]
                temp_nsbr = df[df['ground_truth_output'] == 0]
                print("radio=", str(radio), "epoch=", str(epoch), "this epoch all sbrs number=",
                      str(len(temp_sbr)), "nsbrs number=", str(len(temp_nsbr)), "===================\n")
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
            return df_sbr.append(df_nsbr)

        # 防御——用于获取可攻击的处理好的训练集样本
        if tag == 'adv-training':
            csv_path=os.path.join(self.training_origin_dir, self.dataset_dict[project])
            df = pd.read_csv(csv_path, sep=',')
            return df

        # 防御——检测重训练模型在对抗样本数据集上的检测精度，是否更加精确
        if tag == 'attack-test':
            if baseline:
                file_name = self.result_file(['LTRWES', project, method, '.csv'], dir=self.baseline_attack_dataset_dir)
                csv_path = f"{self.baseline_attack_dataset_dir}/{file_name}"
            else:
                file_name = self.result_file(['LTRWES', project, method, '.csv'],
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
        return ' '.join(words[:300])

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
        if LTRWESParseData.is_unwanted(word):
            return False
        # Remove the length of word less than 3
        if len(word) < 3:
            return False
        return True

    # def score_fun1(self, df_nsbr):
    #     new_df_nsbr = pd.DataFrame()
    #     for i, row in df_nsbr.iterrows():
    #         origin_grammer_error = self.language_tool.get_errors(row['original_text'])
    #         perturb_grammer_error = self.language_tool.get_errors(row['text'])
    #         row['grammer_error'] = perturb_grammer_error - origin_grammer_error
    #         # row['attack_sim'] = self.use_metric.calc_metric(row['original_text'], row['text'])
    #         new_df_nsbr = new_df_nsbr.append(row)
    #     # new_df_nsbr.sort_values(by=['grammer_error', 'attack_sim'], ascending=[True, False], inplace=True)
    #     new_df_nsbr.sort_values(by=['grammer_error'], ascending=[True], inplace=True)
    #     return new_df_nsbr
