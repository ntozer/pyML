from models import NaiveBayes
from data_handler import DataHandler


class BaggingNB:
    def __init__(self, examples, targets, num_attributes):
        self.targets = targets
        self.examples = examples
        self.num_attributes = num_attributes
        self.nb_classifiers = []

    def train(self, num_classifiers=50):
        bagged_datasets = DataHandler.create_bagged_datasets(num_classifiers, self.examples, self.targets)
        for bagged_dataset in bagged_datasets:
            naive_bayes = NaiveBayes(bagged_dataset[0], bagged_dataset[1], self.num_attributes)
            naive_bayes.train()
            self.nb_classifiers.append(naive_bayes)

    def classify(self, attributes):
        votes = {}
        for target in set(self.targets):
            votes[target] = 0
        for id3 in self.nb_classifiers:
            votes[id3.classify(attributes)] += 1
        max_key = self.targets[0]
        for key in votes:
            if votes[key] > votes[max_key]:
                max_key = key
        return max_key
