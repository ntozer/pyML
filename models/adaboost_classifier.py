from models import ID3
from models.id3_classifier import ID3Node
from data_handler import DataHandler
from model_evaluator import ModelEvaluator
from math import log, exp


class AdaBoost:
    def __init__(self, examples, targets):
        self.examples = examples
        self.targets = targets
        self.stumps = []
        self.coefficients = []

    @staticmethod
    def error(targets, attribute_vec, weights):
        error = 0
        count_dict = {}
        clf_dict = {}

        for attribute in set(attribute_vec):
            count_dict[attribute] = {}
            for target in set(targets):
                count_dict[attribute][target] = 0

        for attribute in set(attribute_vec):
            for i in range(len(attribute_vec)):
                if attribute_vec[i] == attribute:
                    count_dict[attribute][targets[i]] += 1

        for attribute in set(attribute_vec):
            max_value = 0
            max_key = None
            for key, value in count_dict[attribute].items():
                if value > max_value:
                    max_key = key
                    max_value = value
            clf_dict[attribute] = max_key

        for i in range(len(attribute_vec)):
            if clf_dict[attribute_vec[i]] != targets[i]:
                error += weights[i]

        return error

    @staticmethod
    def train_stump(node, weights):
        # identifies the best attribute for the current node
        min_idx = 0
        min_error = AdaBoost.error(node.targets, DataHandler.column(node.examples, min_idx), weights)
        for i in range(len(node.examples[0])):
            if AdaBoost.error(node.targets, DataHandler.column(node.examples, i), weights) < min_error:
                min_idx, min_error = i, AdaBoost.error(node.targets, DataHandler.column(node.examples, i), weights)
        node.attribute = min_idx

        # creates a dictionary of children associated with attribute values
        for attr_val in set(DataHandler.column(node.examples, min_idx)):
            node.children[attr_val] = ID3Node(parent=node)

        idx = 0
        for row in node.examples:
            node.children[row[min_idx]].examples.append(row.copy())
            node.children[row[min_idx]].targets.append(node.targets[idx])
            idx += 1

        for key, child in node.children.items():
            DataHandler.rm_column(child.examples, node.attribute)

    def train(self, num_stumps=50):
        self.stumps = []
        self.coefficients = []
        weights = [1/len(self.examples) for example in self.examples]
        for n in range(num_stumps):
            eps = 10**-14

            # getting bagged dataset to train id3 stump on
            stump = ID3(self.examples, self.targets)
            AdaBoost.train_stump(stump.root, weights)

            # getting the error and AdaBoost coefficient
            model_eval = ModelEvaluator(stump)
            pred_results = model_eval.training_error()[1]
            err = 0
            for i in range(len(pred_results)):
                if pred_results[i] == 0:
                    err += weights[i]
            coeff = 0.5 * log((1 - err + eps) / (err + eps))

            # updating weights
            for i in range(len(pred_results)):
                if pred_results[i] == 1:
                    weights[i] = weights[i] * exp(-1 * coeff)
                elif pred_results[i] == 0:
                    weights[i] = weights[i] * exp(coeff)

            # normalizing weights
            total_weight = sum(weights)
            for i in range(len(weights)):
                weights[i] = weights[i] / total_weight

            # adding coefficients and stump to lists
            self.stumps.append(stump)
            self.coefficients.append(coeff)

    def classify(self, attributes):
        votes = {}
        for target in set(self.targets):
            votes[target] = 0
        for i in range(len(self.stumps)):
            votes[self.stumps[i].classify(attributes)] += self.coefficients[i]
        max_key = self.targets[0]
        for key in votes:
            if votes[key] > votes[max_key]:
                max_key = key
        return max_key
