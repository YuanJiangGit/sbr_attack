"""
Part of Speech Constraint
--------------------------
"""
from htps.constrains.ValidatorsV1 import transformation_consists_of_word_insertions
from htps.methods.WordSwapHowNetV1 import WordSwapHowNetV1
import pickle
from textattack.shared import utils
import flair
from flair.data import Sentence
from flair.models import SequenceTagger
import lru
import nltk
import textattack
from htps.constrains.ConstraintV1 import ConstraintV1
flair.device = textattack.shared.utils.device
stanza = textattack.shared.utils.LazyLoader("stanza", globals(), "stanza")


class POS(ConstraintV1):
    def __init__(
        self,
        tagger_type="nltk",
        tagset="universal",
        allow_verb_noun_swap=True,
        compare_against_original=True,
    ):
        super().__init__(compare_against_original)
        # Download synonym candidates bank if they're not cached.
        cache_path = utils.download_if_needed(
            "{}/{}".format(WordSwapHowNetV1.PATH, "word_candidates_sense.pkl")
        )

        # Actually load the files from disk.
        with open(cache_path, "rb") as fp:
            self.candidates_bank = pickle.load(fp)

        self.pos_dict = {"ADJ": "adj", "NOUN": "noun", "ADV": "adv", "VERB": "verb"}
        self.tagger_type = tagger_type
        self.tagset = tagset
        self.allow_verb_noun_swap = allow_verb_noun_swap

        self._pos_tag_cache = lru.LRU(2 ** 14)
        if tagger_type == "flair":
            if tagset == "universal":
                self._flair_pos_tagger = SequenceTagger.load("upos-fast")
            else:
                self._flair_pos_tagger = SequenceTagger.load("pos-fast")

        if tagger_type == "stanza":
            self._stanza_pos_tagger = stanza.Pipeline(
                lang="en", processors="tokenize, pos", tokenize_pretokenized=True
            )

    def clear_cache(self):
        self._pos_tag_cache.clear()

    def _get_pos(self, before_ctx, word, after_ctx, insert=False):
        context_words = before_ctx + [word] + after_ctx
        context_key = " ".join(context_words)
        if context_key in self._pos_tag_cache:
            word_list, pos_list = self._pos_tag_cache[context_key]
        else:
            if self.tagger_type == "nltk":
                word_list, pos_list = zip(
                    *nltk.pos_tag(context_words, tagset=self.tagset)
                )

            if self.tagger_type == "flair":
                context_key_sentence = Sentence(
                    context_key, use_tokenizer=textattack.shared.utils.words_from_text
                )
                self._flair_pos_tagger.predict(context_key_sentence)
                word_list, pos_list = textattack.shared.utils.zip_flair_result(
                    context_key_sentence
                )

            if self.tagger_type == "stanza":
                word_list, pos_list = textattack.shared.utils.zip_stanza_result(
                    self._stanza_pos_tagger(context_key), tagset=self.tagset
                )

            self._pos_tag_cache[context_key] = (word_list, pos_list)

        # idx of `word` in `context_words`
        assert word in word_list, "POS list not matched with original word list."
        word_idx = word_list.index(word)
        if insert:
            if word_idx+1 < len(pos_list):
                return pos_list[word_idx], pos_list[word_idx + 1]
            return pos_list[word_idx], True
        return pos_list[word_idx]

    def _check_constraint(self, transformed_text, reference_text, **kwargs):
        try:
            indices = kwargs['indices_to_modify']
        except KeyError:
            raise KeyError(
                "Cannot apply part-of-speech constraint without `newly_modified_indices`"
            )
        for i in indices:
            ref_word = reference_text.words[i]
            next_word = reference_text.words[min(i+1, len(reference_text.words)-1)]
            trans_i = transformed_text.position_reflect[i]
            insert_word = transformed_text.words[trans_i+1]
            before_ctx = reference_text.words[max(i - 5, 0): i]
            after_ctx = reference_text.words[
                            i + 1: min(i + 5, len(reference_text.words))
                        ]
            insert_pos = self._get_pos(before_ctx, insert_word, after_ctx)
            ref_pos = reference_text.attack_attrs['pos'][i]
            next_pos = reference_text.attack_attrs['pos'][min(i+1, len(reference_text.words)-1)]
            if not self._can_insert_pos(ref_pos, insert_pos, next_pos):
                return False
        return True

    def check_compatibility(self, transformation):
        return transformation_consists_of_word_insertions(transformation)

    def _can_insert_pos(self, pos_a, pos_b, pos_c):
        if pos_a == pos_b and pos_a == "NOUN":
            return True
        if pos_a == "VERB" and pos_b == "NOUN" and pos_c == "NOUN":
            return True
        return False
