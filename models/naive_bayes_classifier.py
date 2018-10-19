from data_handler import DataHandler


class NaiveBayes:
    def __init__(self, targets, examples, num_attributes):
        self.target_set = set(targets)
        self.num_instances = len(examples)
        self.targets = targets
        self.examples = examples
        self.target_map = {}
        self.attribute_map = {}
        for target in self.target_set:
            self.target_map[target] = 0
            self.attribute_map[target] = {}
            for i in range(num_attributes):
                self.attribute_map[target][i] = {}
                for attribute in set(DataHandler.column(examples, i)):
                    self.attribute_map[target][i][attribute] = 0

    def train(self):
        for target in self.target_set:
            for i in range(self.num_instances):
                if self.targets[i] == target:
                    self.target_map[target] += 1
                    for j in range(len(self.examples[i])):
                        self.attribute_map[target][j][self.examples[i][j]] += 1

    def classify(self, attributes):
        classifications = []
        for target in self.target_set:
            classification = 1
            for i in range(len(attributes)):
                classification *= self.attribute_map[target][i][attributes[i]] / self.num_instances
            classification *= self.target_map[target] / self.num_instances
            classifications.append(classification)

        max_idx = 0
        for i in range(len(self.target_set)):
            if classifications[i] > classifications[max_idx]:
                max_idx = i

        return (list(self.target_set))[max_idx]
