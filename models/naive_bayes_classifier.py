from data_handler import DataHandler


class NaiveBayes:
    def __init__(self, examples, targets):
        self.num_instances = 0
        self.num_attributes = 0
        self.targets = targets
        self.examples = examples
        self.target_map = {}
        self.attribute_map = {}
        self.col_attr_count = {}
        self.target_set = set(targets)

    def train(self):
        self.target_map = {}
        self.attribute_map = {}
        self.col_attr_count = {}
        self.num_instances = len(self.examples)
        self.num_attributes = len(self.examples[0])

        for target in self.targets:
            self.target_map[target] = self.targets.count(target)
            target_examples = [self.examples[i] for i in range(len(self.examples)) if target == self.targets[i]]
            example_t = list(map(list, zip(*target_examples)))
            for i in range(len(example_t)):
                self.col_attr_count[i] = len(set(example_t[i]))
                for val in set(example_t[i]):
                    self.attribute_map[(target, i, val)] = example_t[i].count(val)

    def classify(self, attributes):
        estimates = []
        for target in self.target_set:
            estimate = 1
            for i in range(len(attributes)):
                occurrences = 0
                if (target, i, attributes[i]) in self.attribute_map.keys():
                    occurrences = self.attribute_map[(target, i, attributes[i])]
                estimate *= (occurrences + 1) / (self.num_instances + self.col_attr_count[i])
            estimate *= (self.target_map[target] if target in self.target_map.keys() else 0) / self.num_instances
            estimates.append(estimate)

        max_idx = 0
        for i in range(len(self.target_set)):
            if estimates[i] > estimates[max_idx]:
                max_idx = i
        return (list(self.target_set))[max_idx]

    def classify_set(self, examples):
        classifications = []
        for row in examples:
            classifications.append(self.classify(row))
        return classifications
