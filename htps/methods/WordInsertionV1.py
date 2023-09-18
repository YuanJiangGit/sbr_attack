"""
Word Insertion
============================================
Word Insertion transformations act by inserting a new word at a specific word index.
For example, if we insert "new" in position 3 in the text "I like the movie", we get "I like the new movie".
Subclasses can implement the abstract ``WordInsertion`` class by overriding ``self._get_new_words``.
"""
from htps.methods.TransformationV1 import TransformationV1


class WordInsertionV1(TransformationV1):
    """A base class for word insertions."""

    def _get_new_words(self, current_text, index):
        """Returns a set of new words we can insert at position `index` of `current_text`
        Args:
            current_text (AttackedText): Current text to modify.
            index (int): Position in which to insert a new word
        Returns:
            list[str]: List of new words to insert.
        """
        raise NotImplementedError()

    def _get_transformations(self, current_text, original_text, indices_to_modify, original_indices_to_modify):
        # current_text: 当前样本
        # original_text: 原始样本
        # indices_to_modify: 映射出来的变化的修改的位置索引，因为插入单词改变了索引位置
        # original_indices_to_modify: 原始的想要修改的位置，是一成不变的
        words = current_text.words
        transformed_texts = []
        for idx, i in enumerate(indices_to_modify):
            # 找到当前样本想要修改的单词
            word_to_replace = words[i]
            replacement_words = self._get_new_words(current_text, i)
            transformed_texts_idx = []
            position_reflect_new = current_text.position_reflect.copy()
            for j in range(original_indices_to_modify[idx] + 1, len(original_text.words)):
                position_reflect_new[j] = position_reflect_new[j] + 1
            for r in replacement_words:
                if r == word_to_replace:
                    continue
                # 插入策略，扰动映射位置
                # 将改动的位置索引映射信息记录下来
                new_text = current_text.insert_text_after_word_index(i, r)
                new_text.position_reflect = position_reflect_new.copy()
                new_text.attack_attrs["perturbed_num"] = current_text.attack_attrs["perturbed_num"]+1
                new_text.attack_attrs["perturb_method"] = 'insert'
                new_text.attack_attrs["insert_num"] = current_text.attack_attrs["insert_num"]+1
                new_text.attack_attrs["swap_num"] = current_text.attack_attrs["swap_num"]
                transformed_texts_idx.append(new_text)
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts