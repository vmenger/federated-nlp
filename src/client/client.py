import pickle
import hashlib
import os

from abc import ABC, abstractmethod
from typing import Generator, Optional


class ClientBase(ABC):

    @abstractmethod
    def num_sentences(self) -> int:
        pass

    @abstractmethod
    def get_train_generator(self) -> Generator:
        pass

    @abstractmethod
    def persist_hashes(self):
        pass


class Client:

    def __init__(self, name: str, filename: str):
        self.name = name
        self.filename = filename
        self.hash_tokens = {}

    def num_sentences(self):
        with open(self.filename) as file:
            return len(file.readlines())

    def _hash_token(self, token: str):
        h = hashlib.sha256(token.encode('utf-8')).hexdigest()
        if token not in self.hash_tokens:
            self.hash_tokens[token] = h
        return h

    def _process_line(self, line: str):
        return [self._hash_token(token) for token in line[:-1].split(" ")]

    def get_train_generator(self, maxrows: Optional[int] = None):

        num_returned_rows = 0

        with open(self.filename, 'r', encoding='utf-8') as file:

            for line in file.readlines():

                num_returned_rows += 1

                yield self._process_line(line)

                if num_returned_rows == maxrows:
                    break

    def persist_hashes(self, path: str):

        with open(os.path.join(path, f"hashed_tokens_{self.name}.pickle"), 'wb') as outfile:
            pickle.dump(self.hash_tokens, outfile)