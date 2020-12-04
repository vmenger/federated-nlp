import numpy as np

from itertools import cycle
from typing import Generator


class MetaGenerator:

    def __init__(self):
        self.sentence_counts = []
        self.generators = []
        self.locked = False

    def register_generator(self, sentence_count: int, generator: Generator):

        if self.locked:
            raise ValueError("This metagenerator is already locked")

        self.sentence_counts.append(sentence_count)
        self.generators.append(cycle(generator))

    def lock(self):
        self.locked = True

    def __iter__(self):

        if not self.locked:
            raise ValueError("Must lock metagenerator before starting generation")

        probs = np.array(self.sentence_counts) / sum(self.sentence_counts)

        for i in range(sum(self.sentence_counts)):
            yield next(np.random.choice(
                self.generators,
                p=probs
            ))
