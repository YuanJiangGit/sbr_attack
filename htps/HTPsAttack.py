from textattack.attack_recipes import AttackRecipe
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification

from htps.AttackV1 import AttackV1
from htps.constrains.StopwordModificationV1 import StopwordModificationV1
from htps.methods.CharacterMethod import CharacterMethod
from htps.methods.CompositeTransformationV1 import CompositeTransformationV1
from htps.methods.WordSwapV1 import WordSwapV1
from htps.search.GreedyBeamSearchV3 import GreedyBeamSearchV3


class InsertHtps(WordSwapV1):
    def __init__(
        self, top_words=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.top_words = top_words

    def _get_replacement_words(self, word):
        candidate_words = []

        for top_word in self.top_words:
            candidate_word = word + ' ' + top_word
            candidate_words.append(candidate_word)

        return candidate_words[:50]


class HTPsAttack(AttackRecipe):
    nsbr_top_words = []
    top_words = []

    def __init__(self, project, model_type):
        nsbr_top_words_path = '..\\resources\\nsbr_words\\' + project + '_' + model_type
        top_words_path = '..\\resources\\top_words\\' + project + '_' + model_type
        with open(nsbr_top_words_path, 'r+') as f:
            HTPsAttack.nsbr_top_words = f.read().split()
        with open(top_words_path, 'r+') as f:
            HTPsAttack.top_words = f.read().split()

    @staticmethod
    def build(model):
        transformation = CompositeTransformationV1(
            [
                CharacterMethod(),
            ]
        )
        # Universal Sentence Encoder with a minimum angular similarity of ε = 0.5.
        #
        # In the TextFooler code, they forget to divide the angle between the two
        # embeddings by pi. So if the original threshold was that 1 - sim >= 0.5, the
        # new threshold is 1 - (0.5) / pi = 0.840845057
        #
        constraints = [StopwordModificationV1()]
        use_constraint = UniversalSentenceEncoder(
            # threshold=0.7,
            # metric="cosine",
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        goal_function = UntargetedClassification(model)
        search_method = GreedyBeamSearchV3(beam_width=1)
        return AttackV1(goal_function, constraints, transformation, search_method)
