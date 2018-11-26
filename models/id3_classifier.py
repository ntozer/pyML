from math import log2
from data_handler import DataHandler


class ID3Node:
    def __init__(self, examples=None, targets=None, parent=None):
        self.examples = examples if examples else []
        self.targets = targets if targets else []
        self.parent = parent
        self.children = {}
        self.attribute = None


class ID3:
    def __init__(self, examples, targets):
        self.examples = examples
        self.targets = targets
        self.root = ID3Node(examples, targets)

    @staticmethod
    def entropy(targets):
        target_counts = {}
        for target in targets:
            if target not in target_counts.keys():
                target_counts[target] = 0
            target_counts[target] += 1

        entropy = 0
        for target in set(targets):
            target_prob = target_counts[target] / len(targets)
            entropy -= target_prob * log2(target_prob)
        return entropy

    @staticmethod
    def gain(targets, attribute_vec):
        if len(targets) != len(attribute_vec):
            return -1
        gain = ID3.entropy(targets)
        for attribute in set(attribute_vec):
            attr_targets = []
            for i in range(len(attribute_vec)):
                if attribute_vec[i] == attribute:
                    attr_targets.append(targets[i])
            gain -= ID3.entropy(attr_targets) * len(attr_targets) / len(targets)
        return gain

    def train(self, depth=10):
        ID3.train_helper(self.root, depth)

    @staticmethod
    def train_helper(node, depth):
        if depth == 0 or len(set(node.targets)) == 0:
            return

        # identifies the best attribute for the current node
        min_idx = 0
        min_gain = ID3.gain(node.targets, DataHandler.column(node.examples, min_idx))
        for i in range(len(node.examples[0])):
            if ID3.gain(node.targets, DataHandler.column(node.examples, i)) < min_gain:
                min_idx, min_gain = i, ID3.gain(node.targets, DataHandler.column(node.examples, i))
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
            ID3.train_helper(child, depth-1)

    def classify(self, attributes):
        return ID3.classify_helper(self.root, attributes.copy())

    @staticmethod
    def classify_helper(node, attributes):
        if node.attribute is None or attributes[node.attribute] not in node.children.keys():
            max_estimate = 0
            classification = None
            for estimate in set(node.targets):
                if node.targets.count(estimate) > max_estimate:
                    max_estimate = node.targets.count(estimate)
                    classification = estimate
            return classification

        else:
            attribute = attributes[node.attribute]
            del attributes[node.attribute]
            return ID3.classify_helper(node.children[attribute], attributes)
