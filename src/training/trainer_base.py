from abc import ABC, abstractmethod

from typing import Generator, Set

class BaseModelTrainer(ABC):

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def train(self, input_data: Generator):
        pass

    @abstractmethod
    def save_model(self, path: str):
        pass

    @abstractmethod
    def restrict_model(self, tokens: Set[str]):
        pass

