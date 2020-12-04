from training.trainer_base import BaseModelTrainer
from training.input_data import MetaGenerator
from client.client import Client

class Federator:

    def __init__(self):
        self.model_trainer = None
        self.clients = []

    def register_model_trainer(self, model_trainer: BaseModelTrainer):
        self.model_trainer = model_trainer

    def register_client(self, client: Client):
        self.clients.append(client)

    def run(self):

        if self.model_trainer is None:
            raise ValueError("No model trainer set")

        if len(self.clients) == 0:
            raise ValueError("No clients yet")

        metagenerator = MetaGenerator()

        for c in self.clients:
            metagenerator.register_generator(c.num_sentences(), c.get_train_generator())

        metagenerator.lock()

        self.model_trainer.setup_model()
        self.model_trainer.train(metagenerator)
        self.model_trainer.save_model("../models/test.w2v")

        for c in self.clients:
            c.persist_hashes("../data")