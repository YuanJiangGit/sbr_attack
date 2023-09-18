# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function: attack the state-of-the-art SBR detection model (FARSEC)
from ResultLogger.MyLoggerManager import *
from model.SkFARSECModelWrapper import SklearnModelWrapper
import tensorflow as tf
from model.AttackT import AttackT
from Config.ConfigT import MyConf
from dataprocess.FARSECDataProcess import FARSEParseData


class AttackFARSEC(AttackT):
    def __init__(self,project, model_name, recipe, model_type):
        super(AttackFARSEC, self).__init__(project, model_name, recipe, model_type)
        self.sbr_words_dir = '../resources/FARSEC_SBR_words'
        self.sbr_words = None

    def pretrain_model(self, defense=False, epoch=0, radio=None):
        if defense:
            # 重训练模型地址
            target_model_path = os.path.join(self.Defense_TargetModel_PATH,
                                             "epoch" + str(epoch),
                                             self.project + '_' + self.model_name + '_' + str(radio))
        else:
            # 原始模型地址
            target_model_path = os.path.join(self.TargetModel_PATH, self.project + '_' + self.model_name)
        print("target_model_path=", target_model_path)
        if os.path.exists(target_model_path):
            self.target_model._load(target_model_path)
        else:
            df_train = self.dataPipline.load_data(self.project, tag='training')
            x_train, y_train = FARSEParseData.process_train_data_to_tuplexy(df_train)
            self.target_model._train(x_train, y_train)
            self.target_model._save(target_model_path)
        model = self.target_model.object.model
        return model

    def pretrain_model_test(self, df_test, defense=False):
        x_test, y_test = FARSEParseData.process_text_data_to_tuplexy(df_test, self.load_sbr_words(), defense)
        print(len(df_test))
        self.target_model._test(x_test, y_test)

    def load_sbr_words(self):
        if self.sbr_words!=None:
            return self.sbr_words
        else:
            path=os.path.join(self.sbr_words_dir, self.project)
            with open(path,'r') as f:
                self.sbr_words = f.read().split()
            return self.sbr_words

    def modelAttack(self, defense=False, epoch=0, radio=None):
        # defense：false代表从原始模型进行攻击，true代表从重训练模型进行攻击，二者存储地址不一样，因此需加以区分
        # epoch：经过几轮训练集的对抗重训练
        # radio：筛选的radio%的NSBRs的增强训练的模型对应的radio
        config = MyConf('../Config/config.cfg')
        config.model = self.model_name
        config.recipe = self.recipe
        config.project = self.project
        config.model_type = self.model_type
        # determine whether attack results have been existed
        # if self.isExist(config):
        #     return
        sbr_words=self.load_sbr_words()
        # load models
        target_model = self.pretrain_model(defense, epoch=epoch, radio=radio)
        model_wrapper = SklearnModelWrapper(target_model,sbr_words)
        # obtain attack
        attack= self.AttacK_Recipes[self.recipe].build(model_wrapper)
        print(attack)

        # load testing dataset
        df_test = self.dataPipline.load_data(self.project, tag='testing')
        if test_target:
            self.pretrain_model_test(df_test)

        # the number of samples used to generate adversarial examples
        num_examples = config.num_examples if len(df_test)>config.num_examples else 500
        # samples
        if self.project == 'chromium':
            df_sbr = df_test[df_test['Security']==1]
            df_nsbr = df_test[df_test['Security']==0]
            samples = df_sbr.append(df_nsbr.sample(n=num_examples, random_state=1))
        else:
            samples = df_test.sample(n=num_examples, random_state=1)
        sample_list = samples.apply(lambda x: (x['text'], int(x['Security'])), axis=1).to_list() # convert into the custom dataset
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
    project_best_classifier = {'ambari': 'RF', 'wicket': 'MLP', 'camel': 'MLP', 'derby': 'NB', 'chromium': 'NB'}
    best_radios = {"ambari": 0.4, 'camel': 0.3, 'derby': 0.1, 'chromium': 0.1}
    test_target=True
    for key, value in project_best_classifier.items():
        for recipe in recipes:
            sbr_attack = AttackFARSEC(key, value, recipe, 'FARSEC')
            sbr_attack.modelAttack()
