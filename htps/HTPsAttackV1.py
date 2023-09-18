from textattack.attack_recipes import AttackRecipe
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification

from htps.AttackV1 import AttackV1
from htps.constrains.POS import POS
from htps.constrains.StopwordModificationV1 import StopwordModificationV1
from htps.constrains.WordEmbeddingDistanceV1 import WordEmbeddingDistanceV1
from htps.methods.CompositeTransformationV1 import CompositeTransformationV1
from htps.methods.WordInsertionV1 import WordInsertionV1
from htps.methods.WordSwapEmbeddingV1 import WordSwapEmbeddingV1
from htps.methods.WordSwapHowNetThenEmbed import WordSwapHowNetThenEmbed
from htps.methods.WordSwapHowNetV1 import WordSwapHowNetV1
from htps.methods.WordSwapWordNetV1 import WordSwapWordNetV1
from htps.search.GreedyBeamSearchV3 import GreedyBeamSearchV3


class InsertHtps(WordInsertionV1):
    def __init__(
        self, top_words=None, num=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.top_words = top_words
        self.insert_num_limit = num

    def _get_new_words(self, current_text, index):
        if self.insert_num_limit != None and current_text.attack_attrs["insert_num"] >= self.insert_num_limit:
            return []
        return self.top_words[:50]

class HTPsAttackV1(AttackRecipe):
    nsbr_top_words = []
    top_words = []

    def __init__(self, project, model_type):
        nsbr_top_words_path = '..\\resources\\nsbr_words\\' + project + '_' + model_type
        top_words_path = '..\\resources\\top_words\\' + project + '_' + model_type
        with open(nsbr_top_words_path, 'r+') as f:
            HTPsAttackV1.nsbr_top_words = f.read().split()
        with open(top_words_path, 'r+') as f:
            HTPsAttackV1.top_words = f.read().split()

    @staticmethod
    def build(model):
        transformation = CompositeTransformationV1([
            # WordSwapHowNetV1(max_candidates=50),
            WordSwapHowNetThenEmbed(max_candidates=50),
            # WordSwapEmbeddingV1(max_candidates=50),
            # WordSwapWordNetV1(),
            InsertHtps(top_words=HTPsAttackV1.nsbr_top_words),
        ])
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost",
             "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any",
             "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at",
             "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between",
             "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't",
             "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty",
             "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former",
             "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here",
             "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however",
             "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just",
             "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more",
             "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't",
             "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing",
             "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others",
             "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't",
             "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere",
             "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence",
             "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those",
             "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until",
             "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever",
             "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
             "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why",
             "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd",
             "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
        constraints = [StopwordModificationV1(stopwords=stopwords)]

        # Minimum word embedding cosine similarity of 0.5.
        # (The paper claims 0.7, but analysis of the released code and some empirical
        # results show that it's 0.5.)
        # constraints.append(WordEmbeddingDistanceV1(min_cos_sim=0.5))

        # Only replace words with the same part of speech (or nouns with verbs)
        constraints.append(POS())

        # Universal Sentence Encoder with a minimum angular similarity of ε = 0.5.
        # In the TextFooler code, they forget to divide the angle between the two
        # embeddings by pi. So if the original threshold was that 1 - sim >= 0.5, the
        # new threshold is 1 - (0.5) / pi = 0.840845057
        use_constraint = UniversalSentenceEncoder(
            threshold=0.8,
            metric="cosine",
            compare_against_original=True,
        )
        constraints.append(use_constraint)

        goal_function = UntargetedClassification(model)
        search_method = GreedyBeamSearchV3(beam_width=1)
        return AttackV1(goal_function, constraints, transformation, search_method)
