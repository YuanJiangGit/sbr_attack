"""

Stopword Modification
--------------------------

"""

import nltk
import textattack
from flair.data import Sentence
from textattack.constraints import PreTransformationConstraint

from htps.constrains.ValidatorsV1 import transformation_consists_of_word_swaps_and_insertions


class StopwordModificationV1(PreTransformationConstraint):
    """A constraint disallowing the modification of stopwords."""

    def __init__(self, stopwords=None):
        if stopwords is not None:
            self.stopwords = set(stopwords)
        else:
            self.stopwords = set(nltk.corpus.stopwords.words("english"))

    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in ``current_text`` which are able to be
        modified."""
        non_stopword_indices = set()
        # for i, word in enumerate(current_text.words):
        #     if word not in self.stopwords:
        #         non_stopword_indices.add(i)
        for _, value in current_text.position_reflect.items():
            if current_text.words[value] not in self.stopwords:
                non_stopword_indices.add(value)
        tran_sentence = Sentence(
            current_text.text, use_tokenizer=textattack.shared.utils.words_from_text
        )
        textattack.shared.utils.flair_tag(tran_sentence)
        tran_word, tran_pos = textattack.shared.utils.zip_flair_result(
            tran_sentence
        )
        current_text.attack_attrs['pos'] = tran_pos
        return non_stopword_indices

    def check_compatibility(self, transformation):
        """The stopword constraint only is concerned with word swaps since
        paraphrasing phrases containing stopwords is OK.

        Args:
            transformation: The ``Transformation`` to check compatibility with.
        """
        return transformation_consists_of_word_swaps_and_insertions(transformation)
