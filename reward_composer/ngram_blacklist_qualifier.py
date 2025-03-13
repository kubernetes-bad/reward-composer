import json
import re
from typing import List

import spacy

from .qualifiers import QualifierInput, Qualifier

nlp = spacy.load("en_core_web_sm")


class NgramBlacklistQualifier(Qualifier):
    def __init__(self, blacklist_path: str, n_min: int = 2, n_max: int = 4):
        """
        Qualifier that checks if completion does not have any blacklisted ngrams.

        :param blacklist_path: Path to JSON file containing an array of blacklisted ngrams.
        :param n_min: Minimum length of ngrams to check (2-grams, 3-grams, etc).
        :param n_max: Maximum length of ngrams to check.
        """
        super().__init__(name="ngram_blacklist")
        with open(blacklist_path, 'r') as f:
            self.blacklist = json.load(f)
        self.n_min = n_min
        self.n_max = n_max

    @staticmethod
    def text_to_tokens(text: str) -> List[str]:
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        doc = nlp(cleaned_text)
        tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
        return tokens

    @staticmethod
    def generate_ngrams(tokens: List[str], n: int):
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def make_ngrams(self, text: str):
        tokens = NgramBlacklistQualifier.text_to_tokens(text)
        ngrams = []
        for n in range(self.n_min, self.n_max + 1):
            ngrams.extend(NgramBlacklistQualifier.generate_ngrams(tokens, n))
        return ngrams

    def __call__(self, completion: str, context: QualifierInput = None) -> bool:
        completion = completion.lower()
        ngrams = self.make_ngrams(completion)
        return not any(word in ngrams for word in self.blacklist)
