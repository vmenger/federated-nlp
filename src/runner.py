from training.trainer_word2vec import Word2VecModelTrainer
from federator import Federator

from client.client import Client

if __name__ == '__main__':

    f = Federator()

    model = Word2VecModelTrainer()
    f.register_model_trainer(model)

    client1 = Client(name="moby_dick", filename="../data/moby_dick.txt")
    client2 = Client(name="the_brothers_karamazov", filename="../data/the_brothers_karamazov.txt")
    f.register_client(client1)
    f.register_client(client2)

    f.run()