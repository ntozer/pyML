import copy
import math
from data_handler import DataHandler


class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def classify_set(self, test_set):
        classifications = []
        for test in test_set:
            classifications.append(self.model.classify(test))
        return classifications

    @staticmethod
    def compute_accuracy(classifications, targets):
        p = classifications
        y = targets

        correct = 0
        incorrect = 0
        for i in range(len(y)):
            if p[i] == y[i]:
                correct += 1
            else:
                incorrect += 1

        accuracy = correct / (correct + incorrect)

        return accuracy

    def k_fold_cross_val(self, k=5):
        model = copy.deepcopy(self.model)
        examples = copy.deepcopy(model.examples)
        targets = copy.deepcopy(model.targets)
        dataset = []
        # retrieve and shuffle dataset
        for i in range(len(examples)):
            dataset.append(examples[i])
            dataset[i].append(targets[i])
        DataHandler.shuffle_dataset(dataset)

        # divide dataset into k parts
        fold_len = math.ceil(len(dataset) / k)
        folds = []
        for i in range(k):
            base = i * fold_len
            limit = (i + 1) * fold_len
            folds.append(dataset[base:limit])

        # allow each fold to be the cross validation set once
        training_sets = [[] for i in range(k)]
        test_sets = []
        for i in range(k):
            for j in range(k):
                if i == j:
                    test_sets.append(folds[j])
                else:
                    training_sets[i] += folds[j]

        accuracy = []
        for i in range(k):
            training = copy.deepcopy(training_sets[i])
            test = copy.deepcopy(test_sets[i])
            training_targets = DataHandler.column(training_sets[i], -1)
            test_targets = DataHandler.column(test_sets[i], -1)
            DataHandler.rm_column(training, -1)
            DataHandler.rm_column(test, -1)

            # train the model on training set i
            self.model.targets = training_targets
            self.model.examples = training
            self.model.train()

            # test model on test set i
            classifications = self.classify_set(test)

            # get accuracy
            accuracy.append(ModelEvaluator.compute_accuracy(classifications, test_targets))

        avg_accuracy = sum(accuracy)/len(accuracy)

        return avg_accuracy

    def n_time_k_cross_fold(self, n, k):
        accuracy = []
        for i in range(n):
            accuracy.append(self.k_fold_cross_val(k))

        mean = sum(accuracy) / len(accuracy)

        std_dev = 0
        for val in accuracy:
            std_dev += (mean - val) ** 2
        std_dev = (1 / len(accuracy) * std_dev) ** 0.5

        return mean, std_dev

    # model functions
    def training_error(self):
        incorrect = 0
        wrong_preds = []
        num_examples = len(self.model.examples)
        predictions = []
        for i in range(num_examples):
            prediction = self.model.classify(self.model.examples[i])
            if prediction != self.model.targets[i]:
                predictions.append(0)
                wrong_preds.append((self.model.examples[i], self.model.targets[i]))
                incorrect += 1
            else:
                predictions.append(1)
        return float(incorrect) / float(num_examples), predictions
