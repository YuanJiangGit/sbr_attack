# -*- coding: utf-8 -*-
# @Author  :
# @Time    :
# @Function: evaluate the adversarial examples generated by each attack methods.

import os
import pandas as pd
from tqdm import notebook as tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import tensorflow as tf
import tensorflow_hub as hub
import torch
import math
import numpy as np

PYTORCH_DEVICE = 0
TF_DEVICE = 0
# torch.cuda.set_device(0)

class GPT2Metric:
    '''
    GPT2Metric measures the percent difference is perplexities of original text  𝑥  and adversarial example  𝑥𝑎𝑑𝑣 .
    '''
    def __init__(self):
        self._model = AutoModelForCausalLM.from_pretrained("gpt2")
        # self._model.to(device=f'cuda:{PYTORCH_DEVICE}') # jiangyuan
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    def perplexity(self, text):
        input_ids = self._tokenizer.encode(text)
        input_ids = input_ids[: self._tokenizer.model_max_length - 2]
        input_ids.insert(0, self._tokenizer.bos_token_id)
        input_ids.append(self._tokenizer.eos_token_id)
        input_ids = torch.tensor(input_ids)
        # input_ids = input_ids.to(device=f'cuda:{PYTORCH_DEVICE}') # jiangyuan
        with torch.no_grad():
            loss = self._model(input_ids, labels=input_ids)[0].item()

        perplexity = math.exp(loss)
        return perplexity

    def calc_metric(self, orig_text, new_text):
        orig_perplexity = self.perplexity(orig_text)
        new_perplexity = self.perplexity(new_text)
        return orig_perplexity, new_perplexity, (new_perplexity - orig_perplexity) / orig_perplexity


class USEMetric:
    '''
    USEMetric measures the Universal Sentence Encoder similarity between  𝑥  and  𝑥𝑎𝑑𝑣 .
    '''
    def __init__(self):
        # tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        tfhub_url = "D:\\downloads\\universal-sentence-encoder_4"
        with tf.device(f'/device:CPU:{TF_DEVICE}'):
            self._model = hub.load(tfhub_url)

    def encode(self, orig_text, new_text):
        with tf.device(f'/device:CPU:{TF_DEVICE}'):
            return self._model([orig_text, new_text]).numpy()

    def get_angular_sim(self, emb1, emb2):
        cos_sim = torch.nn.CosineSimilarity(dim=0)(emb1, emb2)
        return 1 - (torch.acos(cos_sim) / math.pi)

    def calc_metric(self, orig_text, new_text):
        orig_emb, new_emb = self.encode(orig_text, new_text)
        orig_emb = torch.tensor(orig_emb)
        new_emb = torch.tensor(new_emb)
        sim = self.get_angular_sim(orig_emb, new_emb).item()
        return sim


class PercentageOfWordsChanged:
    '''
    PercentageOfWordsChanged: measures the percentage of words swapped in  𝑥  to produce  𝑥𝑎𝑑𝑣 .
    '''
    def calc_metric(self, orig_text, new_text):
        # orig_words = set(orig_text.split())
        # new_words = set(new_text.split())
        # words_changed = new_words - (orig_words & new_words)
        # words_changed_num = 0
        # for one in new_words:
        #     if one in words_changed:
        #         words_changed_num += 1
        # return words_changed_num * 100 / len(orig_words)

        words_changed_num = 0
        orig_words = orig_text.split()
        new_words = new_text.split()
        for i in range(min(len(orig_words), len(new_words))):
            if orig_words[i] != new_words[i]:
                words_changed_num += 1
        return words_changed_num

class Evaluator:
    '''
    Evaluator: evaluator runs all three metrics for each sample and reports the average.
    '''
    def __init__(self):
        self.use_metric = USEMetric()
        self.gpt2_metric = GPT2Metric()
        self.percentageOfWordsChanged = PercentageOfWordsChanged()

    def evaluate(self, csv_file, all_successful):
        df = pd.read_csv(csv_file)
        df = df[df['result_type'] == "Successful"]

        total_sim = 0
        total_pp_diff = 0
        word_changed_percent = 0
        total_origin_pp_diff = 0
        total_attack_pp_diff = 0
        N = 0
        for i, row in df.iterrows():
            original_text = row["original_text"].replace("[", "").replace("]", "")
            if original_text not in all_successful:
                continue
            perturbed_text = row["perturbed_text"].replace("[", "").replace("]", "")
            sim = self.use_metric.calc_metric(original_text, perturbed_text)
            total_sim += sim
            origin_pp_diff, attack_pp_diff, pp_diff = self.gpt2_metric.calc_metric(original_text, perturbed_text)
            total_origin_pp_diff += origin_pp_diff
            total_attack_pp_diff += attack_pp_diff
            total_pp_diff += pp_diff
            word_changed_percent += row["perturbed_word_num"] / len(original_text.split(" "))
            # word_changed_percent += self.percentageOfWordsChanged.calc_metric(original_text, perturbed_text)
            N += 1

        return total_sim / N, total_origin_pp_diff / N, total_attack_pp_diff / N, total_pp_diff / N, word_changed_percent / N

    def evaluate_v1(self, csv_file, all_successful):
        df = pd.read_csv(csv_file)
        df = df[df['result_type'] == "Successful"]

        total_sim = 0
        total_pp_diff = 0
        word_changed_percent = 0
        total_origin_pp_diff = 0
        total_attack_pp_diff = 0
        total_insert_num = 0
        total_swap_num = 0
        N = 0
        for i, row in df.iterrows():
            original_text = row["original_text"].replace("[", "").replace("]", "")
            if original_text not in all_successful:
                continue
            perturbed_text = row["perturbed_text"].replace("[", "").replace("]", "")
            sim = self.use_metric.calc_metric(original_text, perturbed_text)
            total_sim += sim
            origin_pp_diff, attack_pp_diff, pp_diff = self.gpt2_metric.calc_metric(original_text, perturbed_text)
            total_origin_pp_diff += origin_pp_diff
            total_attack_pp_diff += attack_pp_diff
            total_pp_diff += pp_diff
            word_changed_percent += row["perturbed_word_num"] / len(original_text.split(" "))
            total_insert_num += row["insert_num"]
            total_swap_num += row["swap_num"]
            N += 1
        return total_sim / N, total_origin_pp_diff / N, total_attack_pp_diff / N, \
               total_pp_diff / N, word_changed_percent / N, total_insert_num/ N, total_swap_num / N

def result_file(dir, train_info):
    files=os.listdir(dir)
    for file in files:
        tag=True
        for train in train_info:
            if train not in file:
                tag=False
        if tag:
            return file

def main():
    evaluator = Evaluator()
    # projects = ['ambari', 'camel', 'derby', 'chromium']
    projects = ['ambari']
    # attack_models = ['PWWSRen2019', 'TextBuggerLi2018', 'TextFoolerJin2019', 'DeepWordBugGao2018']
    attack_models = ['HTPsAttackV1']
    model = "LTRWES"
    # RESULT_ROOT_DIR = "../resources/attack_results/现有方法"
    RESULT_ROOT_DIRs = [
                        # "../resources/attack_results/final-htpsattack/final/epoch0",
                        "../resources/attack_results/final-htpsattack/final/test",
                        "../resources/attack_results/final-htpsattack/final/epoch1"]
    # RESULT_ROOT_DIRs = ["../resources/attack_results/train-attack/epoch0",
    #                     "../resources/attack_results/train-attack/epoch1",
    #                     "../resources/attack_results/train-attack/epoch2",
    #                     ]

    all_successful_attacks = []
    # num_files = len(projects) * len(attack_models)
    num_files = len(RESULT_ROOT_DIRs)
    pbar = tqdm.tqdm(total=num_files, smoothing=0)
    for project in projects:
        all_successful = set()
        for am in attack_models:
            for RESULT_ROOT_DIR in RESULT_ROOT_DIRs:
                file_name=result_file(RESULT_ROOT_DIR, [model, project, am,'.csv'])
                csv_path = f"{RESULT_ROOT_DIR}/{file_name}"
                df = pd.read_csv(csv_path)
                df = df[df['result_type'] == "Successful"]
                df["original_text"] = df.apply(lambda row: row["original_text"].replace("[", "").replace("]", ""),
                                               axis=1)
                if len(all_successful) == 0:
                    all_successful = set(df["original_text"])
                else:
                    all_successful = all_successful.intersection(set(df["original_text"]))
                pbar.update(1)
            all_successful_attacks.append(all_successful)

    # num_files = len(projects) * len(attack_models)
    num_files = len(RESULT_ROOT_DIRs)
    pbar = tqdm.tqdm(total=num_files, smoothing=0)
    i = 0
    for project in projects:
        for am in attack_models:
            for RESULT_ROOT_DIR in RESULT_ROOT_DIRs:
                file_name = result_file(RESULT_ROOT_DIR, [model, project, am, '.csv'])
                csv_path = f"{RESULT_ROOT_DIR}/{file_name}"
                print(csv_path)
                all_successful = all_successful_attacks[i]
                avg_sim, avg_origin_pp_diff, avg_attack_pp_diff, avg_pp_diff, words_changed_percent = evaluator.evaluate(csv_path, all_successful)
                # avg_sim, avg_origin_pp_diff, avg_attack_pp_diff, avg_pp_diff, words_changed_percent, avg_insert_num, avg_swap_num = evaluator.evaluate_v1(csv_path, all_successful)
                print(
                    f"Word Changed Percent: {round(words_changed_percent, 2)} \t "
                    f"USE Sim: {round(avg_sim, 3)} \t "
                    f"Origin PP Diff: {round(avg_origin_pp_diff, 1)} \t "
                    f"Attack PP Diff: {round(avg_attack_pp_diff, 1)} \t "
                    f"PP Diff: {round(avg_pp_diff * 100, 1)} \t "
                    # f"avg_swap: {round(avg_swap_num, 1)} \t"
                    # f"avg_insert: {round(avg_insert_num, 1)}"
                )
                pbar.update(1)
        i += 1


if __name__ == '__main__':
    tf.enable_eager_execution(
        config=None,
        device_policy=None,
        execution_mode=None
    )
    main()