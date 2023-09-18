# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function:
import os
from abc import ABCMeta, abstractmethod

from htps.HTPsAttack import HTPsAttack
from htps.HTPsAttackV1 import HTPsAttackV1
from htps.HTPsAttackV2 import HTPsAttackV2
from htps.HTPsAttackV3 import HTPsAttackV3
from htps.HTPsAttackV4 import HTPsAttackV4
from model.TargetModel import TargetModel
from dataprocess.LTRWESDataProcess import LTRWESParseData
from dataprocess.FARSECDataProcess import FARSEParseData
from textattack.attack_recipes import PWWSRen2019, TextBuggerLi2018, TextFoolerJin2019, DeepWordBugGao2018, \
    HotFlipEbrahimi2017, GeneticAlgorithmAlzantot2018, CheckList2020, PSOZang2020, BAEGarg2019, CLARE2020


class AttackT:
    def __init__(self, project, model_name, recipe, model_type):
        self.AttacK_Recipes = {
            'PWWSRen2019': PWWSRen2019,
            'TextBuggerLi2018': TextBuggerLi2018,
            'TextFoolerJin2019': TextFoolerJin2019,
            'DeepWordBugGao2018': DeepWordBugGao2018,
            'HotFlipEbrahimi2017': HotFlipEbrahimi2017,
            'GeneticAlgorithmAlzantot2018': GeneticAlgorithmAlzantot2018,
            'CheckList2020': CheckList2020,
            'PSOZang2020': PSOZang2020,
            'HTPsAttack': HTPsAttack(project, model_type),
            'BAEGarg2019': BAEGarg2019,
            'CLARE2020': CLARE2020,
            'HTPsAttackV1': HTPsAttackV1(project, model_type),
            'HTPsAttackV2': HTPsAttackV2(project, model_type),
            'HTPsAttackV3': HTPsAttackV3(project, model_type),
            'HTPsAttackV4': HTPsAttackV4(project, model_type),
        }
        self.project = project
        self.model_name = model_name
        self.recipe = recipe
        # setting the path of models
        if model_type=='LTRWES':
            self.dataPipline = LTRWESParseData()
            self.TargetModel_PATH = '../resources/LTRWES_target_model/'
            self.Defense_TargetModel_PATH = '../resources/defense/LTRWES_target_model/'
        elif model_type=='FARSEC':
            self.dataPipline = FARSEParseData()
            self.TargetModel_PATH = '../resources/FARSEC_target_model/'
            self.Defense_TargetModel_PATH = '../resources/defense/FARSEC_target_model/'
        self.target_model = TargetModel(model_name)
        self.model_type = model_type

    def isExist(self,config):
        dir = config.log_to_txt
        files=os.listdir(dir)
        for file in files:
            tag=True
            for info in [config.model,config.project,config.recipe,config.model_type]:
                if info not in file:
                    tag =False
            if tag:
                return tag
        return False

    @abstractmethod
    def pretrain_model(self):
        raise NotImplementedError

    @abstractmethod
    def pretrain_model_test(self,df_test):
        raise NotImplementedError

    @abstractmethod
    def modelAttack(self):
        raise NotImplementedError