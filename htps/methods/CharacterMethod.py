"""
Word Swap by Neighboring Character Swap
============================================
"""

import numpy as np

from htps.methods.WordSwapV1 import WordSwapV1


class CharacterMethod(WordSwapV1):
    """Transforms an input by replacing its words with a neighboring character
    swap.

    Args:
        random_one (bool): Whether to return a single word with two characters
            swapped. If not, returns all possible options.
        skip_first_char (bool): Whether to disregard perturbing the first
            character.
        skip_last_char (bool): Whether to disregard perturbing the last
            character.
    """

    def __init__(
        self, random_one=True, skip_first_char=False, skip_last_char=False, **kwargs
    ):
        super().__init__()
        self.random_one = random_one
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char
        self.homos = {
            "-": "˗",
            "9": "৭",
            "8": "Ȣ",
            "7": "𝟕",
            "6": "б",
            "5": "Ƽ",
            "4": "Ꮞ",
            "3": "Ʒ",
            "2": "ᒿ",
            "1": "l",
            "0": "O",
            "'": "`",
            "a": "ɑ",
            "b": "Ь",
            "c": "ϲ",
            "d": "ԁ",
            "e": "е",
            "f": "𝚏",
            "g": "ɡ",
            "h": "հ",
            "i": "і",
            "j": "ϳ",
            "k": "𝒌",
            "l": "ⅼ",
            "m": "ｍ",
            "n": "ո",
            "o": "о",
            "p": "р",
            "q": "ԛ",
            "r": "ⲅ",
            "s": "ѕ",
            "t": "𝚝",
            "u": "ս",
            "v": "ѵ",
            "w": "ԝ",
            "x": "×",
            "y": "у",
            "z": "ᴢ",
        }

    def _get_replacement_words(self, word):
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 1 if self.skip_first_char else 0
        end_idx = (len(word) - 2) if self.skip_last_char else (len(word) - 1)

        if start_idx >= end_idx:
            return []

        if self.random_one:
            i = np.random.randint(start_idx, end_idx)
            # 交换
            candidate_word = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
            candidate_words.append(candidate_word)
            # 替换
            candidate_word = word[:i] + self._get_random_letter() + word[i + 1 :]
            candidate_words.append(candidate_word)
            # 删除
            candidate_word = word[:i] + word[i + 1:]
            candidate_words.append(candidate_word)
            # 插入
            candidate_word = word[:i] + self._get_random_letter() + word[i:]
            candidate_words.append(candidate_word)
            # 视觉替换
            if word[i] in self.homos:
                repl_letter = self.homos[word[i]]
                candidate_word = word[:i] + repl_letter + word[i + 1:]
                candidate_words.append(candidate_word)
        else:
            for i in range(start_idx, end_idx):
                candidate_word = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
                candidate_words.append(candidate_word)
                candidate_word = word[:i] + self._get_random_letter() + word[i + 1:]
                candidate_words.append(candidate_word)
                candidate_word = word[:i] + word[i + 1:]
                candidate_words.append(candidate_word)
                candidate_word = word[:i] + self._get_random_letter() + word[i:]
                candidate_words.append(candidate_word)
                if word[i] in self.homos:
                    repl_letter = self.homos[word[i]]
                    candidate_word = word[:i] + repl_letter + word[i + 1:]
                    candidate_words.append(candidate_word)
        return candidate_words

    @property
    def deterministic(self):
        return not self.random_one

    def extra_repr_keys(self):
        return super().extra_repr_keys() + ["random_one"]
