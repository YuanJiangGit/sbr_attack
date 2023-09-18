# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function:
from model.SecurityBRClassifier import SecurityClassifier
import pickle


class TargetModel:
    def __init__(self,model_name):
        model_list=['RFCV', 'LRCV', 'KNNCV','LR', 'MLP', 'SVM', 'NB', 'RF','KNN' ,'NBCV','LRCV', 'SVMCV','MLPCV','RFCV', 'KNNCV']
        if model_name not in model_list:
            print('No this model')
        self.object = SecurityClassifier(model_name)

    def _train(self, x_train, y_train):
        self.object.train(x_train, y_train)

    def _save(self, store_name):
        pickle.dump(self.object, open(store_name, 'wb'))

    def _load(self,load_name):
        self.object = pickle.load(open(load_name, 'rb'))

    def _test(self,x_test, y_test):
        y_pred = self.object.predict_b(x_test)
        result = self.object.evaluate_b(y_test, y_pred)
        print(result)
        # count=0
        # skipped_lst=[]
        # for i in range(len(y_pred)):
        #     if not int(y_pred[i]) == y_test.iloc[i]:
        #         skipped_lst.append(i)
        #         print(f'The {i}-th bug report is skipped, whose predicted label is {y_pred[i]}, whose ground truth is {y_test.iloc[i]}')
        #         count+=1
        # print(f'the total of skipped is {count}')
        # print(skipped_lst)
