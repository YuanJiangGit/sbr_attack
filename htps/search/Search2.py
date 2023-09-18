import numpy as np
import pandas as pd
from textattack.attack_results import SuccessfulAttackResult
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod

# 主要的不同在于该搜索方法设定了不同的评分函数
from evaluation.test import USEMetric
from htps.CSVLoggerV1 import CSVLoggerV1
from htps.constrains.ValidatorsV1 import transformation_consists_of_word_swaps_and_deletions


class Search2(SearchMethod):
    def __init__(self, beam_width=1):
        self.beam_width = beam_width
        self.derby_train_adv_path = "E:\\postgraduate\\research\\attack\\sbr_attack\\resources\\attack_results\\train-attack"
        self.derby_train_adv = []
        self.logger = CSVLoggerV1(filename=self.derby_train_adv_path+'derby.csv', color_method="file")

    def _get_index_order(self, initial_text):
        """按照删除单词后的分数下降来对位置进行排序"""
        len_text = len(initial_text.words)
        leave_one_texts = [
            initial_text.delete_word_at_index(i) for i in range(len_text)
        ]
        leave_one_results, search_over = self.get_goal_results(leave_one_texts)
        index_scores = np.array([result.score for result in leave_one_results])
        index_order = (-index_scores).argsort()

        return index_order, search_over

    def score_fun1(self, results, cur_result):
        results = sorted(results, key=lambda x: -x.score)
        return results

    def score_fun2(self, results, cur_result):
        # 按照插入策略和替换策略的不同权重来计算分数并排序
        results = sorted(results, key=lambda x: -(x.score - cur_result.score) *
                                                x.attacked_text.perturbed_weight[x.attacked_text.attack_attrs["perturb_method"]])
        return results

    def score_fun3(self, results, cur_result=None):
        # 结合置信度下降程度，扰动量下降程度，置信度下降越多越好，扰动量下降越小越好
        # 采用无量纲的方式进行计算，此处选择线性函数归一化进行处理，通过记录max、min来将其映射到【0,1】范围内数字，再进行相减排序
        score_list = [one.score for one in results]
        sim_list = [one.attacked_text.attack_attrs['similarity_score'] for one in results]
        max_score, min_score = max(score_list), min(score_list)
        max_sim, min_sim = max(sim_list), min(sim_list)
        if max_score - min_score == 0 or max_sim - min_sim == 0:
            results = sorted(results, key=lambda x: -x.score)
        else:
            for x in results:
                score_percentage = (x.score - min_score) / (max_score - min_score)
                sim_percentage = (x.attacked_text.attack_attrs['similarity_score'] - min_sim) / (max_sim - min_sim)
                x.attacked_text.sorted_score = score_percentage + sim_percentage
            results = sorted(results, key=lambda x: -(x.attacked_text.sorted_score))
        return results

    def score_fun4(self, results):
        # TODO：按照insert和swap分开计算评分标准，然后分尅返回两个参数列表，分别从中取出优质样本混合进行后续行为
        insert_candidates, swap_candidates = [], []
        for x in results:
            if x.attacked_text.attack_attrs["perturb_method"] == "insert":
                insert_candidates.append(x)
            else:
                swap_candidates.append(x)
        if len(insert_candidates) != 0:
            insert_candidates = self.score_fun3(insert_candidates)
        if len(swap_candidates) != 0:
            swap_candidates = self.score_fun3(swap_candidates)
        return insert_candidates, swap_candidates

    def _perform_search(self, initial_result):
        '''
        对于贪心方法的beam-search的扩展，每次不是选择最优的那个样本进行迭代，而是保留beam-width个样本
        通过扩大了搜索空间来换取更优的样本
        设置评分函数，赋予插入和修改策略不同的权重
        :param initial_result:
        :return:
        '''
        attacked_text = initial_result.attacked_text
        # Sort words by order of importance
        try:
            index_order, search_over = self._get_index_order(attacked_text)
            # index_order = np.array([i for i in range(len(attacked_text.words))])
            search_over = False
            cur_result = initial_result
            beam = [initial_result.attacked_text]
            i = 0
            iteration_number = 0
            while i < len(index_order) and not search_over:
                potential_next_beam = []
                for text in beam:
                    transformed_text_candidates = self.get_transformations(
                        text,
                        original_text=initial_result.attacked_text,
                        indices_to_modify=[index_order[i]],
                    )
                    if len(transformed_text_candidates) == 0:
                        continue
                    potential_next_beam += transformed_text_candidates
                i += 1
                if len(potential_next_beam) == 0:
                    continue

                iteration_number += 1
                # 超过迭代次数，停止迭代，视为查找失败
                if iteration_number > 30:
                    return cur_result
                results, search_over = self.get_goal_results(potential_next_beam)
                results = [result for result in results if result.score > cur_result.score]
                # Skip swaps which don't improve the score
                if len(results) == 0:
                    continue
                # results = self.score_fun1(results, cur_result)
                results = self.score_fun2(results, cur_result)
                # results = self.score_fun3(results, cur_result)
                # 保留的beam-width个样本
                beam = [result.attacked_text for result in results[:self.beam_width]]
                cur_result = results[0]

                if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    # If we succeeded, return the index with best similarity.
                    max_similarity = -float("inf")
                    best_result = cur_result
                    for result in results:
                        if result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                            # success_result = SuccessfulAttackResult(
                            #     initial_result,
                            #     result,
                            # )
                            # self.logger.log_attack_result(success_result)
                            similarity_score = result.attacked_text.attack_attrs["similarity_score"]
                            if similarity_score > max_similarity:
                                max_similarity = similarity_score
                                best_result = result
                    # self.logger.flush()
                    return best_result
            return cur_result
        except:
            return initial_result


    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["beam_width"]
