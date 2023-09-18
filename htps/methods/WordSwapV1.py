"""
Word Swap
============================================
Word swap transformations act by replacing some words in the input. Subclasses can implement the abstract ``WordSwap`` class by overriding ``self._get_replacement_words``

"""
import random
import string

from htps.methods.TransformationV1 import TransformationV1


class WordSwapV1(TransformationV1):
    """An abstract class that takes a sentence and transforms it by replacing
    some of its words.

    letters_to_insert (string): letters allowed for insertion into words
    (used by some char-based transformations)
    """
    def __init__(self, letters_to_insert=None):
        self.letters_to_insert = letters_to_insert
        if not self.letters_to_insert:
            self.letters_to_insert = string.ascii_letters

    def _get_replacement_words(self, word):
        """Returns a set of replacements given an input word. Must be overriden
        by specific word swap transformations.

        Args:
            word: The input word to find replacements for.
        """
        raise NotImplementedError()

    def _get_random_letter(self):
        """Helper function that returns a random single letter from the English
        alphabet that could be lowercase or uppercase."""
        return random.choice(self.letters_to_insert)

    def _get_transformations(self, current_text, original_text, indices_to_modify, original_indices_to_modify):
        # current_text: 当前样本
        # original_text: 原始样本
        # indices_to_modify: 映射出来的变化的修改的位置索引，因为插入单词改变了索引位置
        # original_indices_to_modify: 原始的想要修改的位置，是一成不变的
        words = current_text.words
        transformed_texts = []
        for idx, i in enumerate(indices_to_modify):
            word_to_replace = words[i]
            replacement_words = self._get_replacement_words(word_to_replace)
            transformed_texts_idx = []
            for r in replacement_words:
                if r == word_to_replace:
                    continue
                new_text = current_text.replace_word_at_index(i, r)
                new_text.position_reflect = current_text.position_reflect.copy()
                new_text.attack_attrs["perturbed_num"] = current_text.attack_attrs["perturbed_num"]+1
                new_text.attack_attrs["perturb_method"] = 'swap'
                new_text.attack_attrs["swap_num"] = current_text.attack_attrs["swap_num"]+1
                new_text.attack_attrs["insert_num"] = current_text.attack_attrs["insert_num"]
                transformed_texts_idx.append(new_text)
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts
