from models import ID3
from data_handler import DataHandler
from math import sqrt
import random


class RandomForest:
    def __init__(self, examples, targets):
        self.examples = examples
        self.targets = targets
        self.forest = []

    def train(self, forest_size=50, tree_depth=10):
        self.forest = []
        bagged_datasets = DataHandler.create_bagged_datasets(forest_size, self.examples, self.targets)
        for bagged_dataset in bagged_datasets:
            examples = bagged_dataset[0]
            targets = bagged_dataset[1]
            num_attributes = len(examples[0])
            num_chosen_attr = int(sqrt(num_attributes))
            while num_chosen_attr > len(examples[0]):
                DataHandler.rm_column(examples, random.randint(1, len(examples[0])-1))
            id3 = ID3(examples, targets)
            id3.train(tree_depth)
            self.forest.append(id3)

    def classify(self, attributes):
        votes = {}
        for target in set(self.targets):
            votes[target] = 0
        for id3 in self.forest:
            votes[id3.classify(attributes)] += 1
        max_key = self.targets[0]
        for key in votes:
            if votes[key] > votes[max_key]:
                max_key = key
        return max_key
