# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function: Data process class
from abc import ABCMeta, abstractmethod
import re
import csv, os


class ParseData:
    def __init__(self):
        self.dataset_dir = '../resources/dataset/'
        self.dataset_dict = {
            # 'ambari': 'example.csv',
            'ambari': 'Ambari2.csv',
            'camel': 'Camel2.csv',
            'chromium': 'Chromium2.csv',
            'derby': 'Derby2.csv',
            'wicket': 'Wicket2.csv',
            'chromium_large': 'chromium_large.csv',
            'mozilla': 'mozilla_merge_update_process2.csv',
            'mozilla_whole': 'mozilla_whole.csv'
        }
        self.defense_dataset_dir = "../resources/attack_results/train-attack"
        self.attack_dataset_dir = "../resources/attack_results/final-htpsattack/final"
        self.baseline_attack_dataset_dir = "../resources/attack_results/现有方法"
        self.data = []

    @abstractmethod
    def load_data(self,project,tag):
        raise NotImplementedError()

    # 处理summary或者description没有任何内容的情况
    @staticmethod
    def dealNan(x):
        if type(x) == float or type(x) == list:
            x = ' '
        return x

    # specialized processing with chromium
    def split_report(self, raw_text, tag):
        text_list = re.split(r'[;?\n]\s*', raw_text)
        summary = text_list.pop(0)
        while (len(summary) < 35 and len(text_list) > 0):
            summary = summary + text_list.pop(0)
        if tag == 'summary':
            return summary
        else:
            return ' '.join(text_list)

    def result_file(self, train_info, dir="../resources/attack_results/train-attack", epoch=0):
        if dir == "../resources/attack_results/train-attack":
            files = os.listdir(dir+'/epoch'+str(epoch))
        else:
            files = os.listdir(dir)
        for file in files:
            tag = True
            for train in train_info:
                if train not in file:
                    tag = False
            if tag:
                return file
