"""
Word Embedding Distance
--------------------------
"""

from textattack.shared import AbstractWordEmbedding, WordEmbedding

from htps.constrains.ConstraintV1 import ConstraintV1
from htps.constrains.ValidatorsV1 import transformation_consists_of_word_swaps


class WordEmbeddingDistanceV1(ConstraintV1):
    """A constraint on word substitutions which places a maximum distance
    between the embedding of the word being deleted and the word being
    inserted.

    Args:
        embedding (obj): Wrapper for word embedding.
        include_unknown_words (bool): Whether or not the constraint is fulfilled if the embedding of x or x_adv is unknown.
        min_cos_sim (:obj:`float`, optional): The minimum cosine similarity between word embeddings.
        max_mse_dist (:obj:`float`, optional): The maximum euclidean distance between word embeddings.
        cased (bool): Whether embedding supports uppercase & lowercase (defaults to False, or just lowercase).
        compare_against_original (bool):  If `True`, compare new `x_adv` against the original `x`. Otherwise, compare it against the previous `x_adv`.
    """

    def __init__(
        self,
        embedding=WordEmbedding.counterfitted_GLOVE_embedding(),
        include_unknown_words=True,
        min_cos_sim=None,
        max_mse_dist=None,
        cased=False,
        compare_against_original=True,
    ):
        super().__init__(compare_against_original)
        self.include_unknown_words = include_unknown_words
        self.cased = cased

        if bool(min_cos_sim) == bool(max_mse_dist):
            raise ValueError("You must choose either `min_cos_sim` or `max_mse_dist`.")
        self.min_cos_sim = min_cos_sim
        self.max_mse_dist = max_mse_dist

        if not isinstance(embedding, AbstractWordEmbedding):
            raise ValueError(
                "`embedding` object must be of type `textattack.shared.AbstractWordEmbedding`."
            )
        self.embedding = embedding

    def get_cos_sim(self, a, b):
        """Returns the cosine similarity of words with IDs a and b."""
        return self.embedding.get_cos_sim(a, b)

    def get_mse_dist(self, a, b):
        """Returns the MSE distance of words with IDs a and b."""
        return self.embedding.get_mse_dist(a, b)

    def _check_constraint(self, transformed_text, reference_text, **kwargs):
        """Returns true if (``transformed_text`` and ``reference_text``) are
        closer than ``self.min_cos_sim`` or ``self.max_mse_dist``."""
        try:
            indices = kwargs['indices_to_modify']
        except KeyError:
            raise KeyError(
                "Cannot apply part-of-speech constraint without `newly_modified_indices`"
            )
        # indices_to_modify是初始样本对应的位置
        for i in indices:
            ref_word = reference_text.words[i]
            transformed_word = transformed_text.words[transformed_text.position_reflect[i]]

            if not self.cased:
                # If embedding vocabulary is all lowercase, lowercase words.
                ref_word = ref_word.lower()
                transformed_word = transformed_word.lower()

            try:
                ref_id = self.embedding.word2index(ref_word)
                transformed_id = self.embedding.word2index(transformed_word)
            except KeyError:
                # This error is thrown if x or x_adv has no corresponding ID.
                if self.include_unknown_words:
                    continue
                return False

            # Check cosine distance.
            if self.min_cos_sim:
                cos_sim = self.get_cos_sim(ref_id, transformed_id)
                if cos_sim < self.min_cos_sim:
                    return False
            # Check MSE distance.
            if self.max_mse_dist:
                mse_dist = self.get_mse_dist(ref_id, transformed_id)
                if mse_dist > self.max_mse_dist:
                    return False

        return True

    def check_compatibility(self, transformation):
        """WordEmbeddingDistance requires a word being both deleted and
        inserted at the same index in order to compare their embeddings,
        therefore it's restricted to word swaps."""
        return transformation_consists_of_word_swaps(transformation)

    def extra_repr_keys(self):
        """Set the extra representation of the constraint using these keys.

        To print customized extra information, you should reimplement
        this method in your own constraint. Both single-line and multi-
        line strings are acceptable.
        """
        if self.min_cos_sim is None:
            metric = "max_mse_dist"
        else:
            metric = "min_cos_sim"
        return [
            "embedding",
            metric,
            "cased",
            "include_unknown_words",
        ] + super().extra_repr_keys()
