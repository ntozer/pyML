import csv
import copy
import random


class DataHandler:
    def __init__(self):
        self.dataset = []
        self.examples = []
        self.targets = []

    def import_data(self, filename):
        self.dataset = []
        with open('data/{}'.format(filename)) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                for idx in range(len(row)):
                    row[idx] = int(row[idx])
                self.dataset.append(row)

    def export_data(self, filename):
        with open('data/prepared/{}'.format(filename), 'w') as file:
            writer = csv.writer(file)
            writer.writerows(self.dataset)

    def count_occurrences(self, column):
        occurrences = {}
        for row in self.dataset:
            if row[column] not in occurrences.keys():
                occurrences[row[column]] = 0
            occurrences[row[column]] += 1
        return occurrences

    def create_targets(self, target_col):
        self.targets = []
        self.examples = copy.deepcopy(self.dataset)
        for row in self.examples:
            self.targets.append(row[target_col])
            del row[target_col]

    def view_abnormal_rows(self):
        for row in self.dataset:
            for val in row:
                if val == '?':
                    print(row)

    @staticmethod
    def create_bagged_datasets(num_bags, examples, targets):
        N = len(examples)
        bagged_datasets = []
        for i in range(num_bags):
            bagged_examples = []
            bagged_targets = []
            for n in range(N):
                j = random.randint(1, N - 1)
                bagged_examples.append(examples[j])
                bagged_targets.append(targets[j])
            bagged_datasets.append((bagged_examples, bagged_targets))
        return bagged_datasets

    @staticmethod
    def column(matrix, i):
        return [row[i] for row in matrix]

    @staticmethod
    def rm_column(matrix, i):
        for row in matrix:
            del row[i]

    @staticmethod
    def index_from_probs(probabilities):
        choice = random.random()
        prob_sum = 1
        for i in range(len(probabilities)):
            if (prob_sum - probabilities[i]) < choice:
                return i
            prob_sum += -probabilities[i]

    @staticmethod
    def create_weighted_bag(examples, targets, probs):
        N = len(examples)
        bagged_examples = []
        bagged_targets = []
        for n in range(N):
            i = DataHandler.index_from_probs(probs)
            bagged_examples.append(examples[i])
            bagged_targets.append(targets[i])
        return bagged_examples, bagged_targets

    @staticmethod
    def shuffle_dataset(dataset):
        for i in range(len(dataset)):
            j = random.randint(0, len(dataset)-1)
            temp = dataset[i]
            dataset[i] = dataset[j]
            dataset[j] = temp
