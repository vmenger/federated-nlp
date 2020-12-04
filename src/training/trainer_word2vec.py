from training.trainer_base import BaseModelTrainer
from collections import defaultdict
import itertools

from gensim.models import Word2Vec

from typing import Generator

class Word2VecModelTrainer(BaseModelTrainer):

    def __init__(self):
        self.model = None

    def setup_model(self):
        pass

    def train(self, input_data: Generator):
        self.model = Word2Vec(sentences=input_data)

    def save_model(self, path: str):
        self.model.save(path)

    def restrict_model(self):
        pass

class HashedWord2Vec:

    def __init__(self, model: Word2Vec, token_to_hash: dict):
        self.model = model
        self.token_to_hash = token_to_hash
        self.hash_to_token = defaultdict(lambda: "UNKNOWN")
        self.hash_to_token.update({v: k for k, v in token_to_hash.items()})

    def most_similar(self, token: str):

        m = self.model.wv.most_similar(self.token_to_hash[token])

        return [(self.hash_to_token[hash], similarity) for hash, similarity in m]

