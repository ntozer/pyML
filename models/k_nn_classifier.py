class kNN:
    def __init__(self, examples, targets):
        self.examples = examples
        self.targets = targets
        self.k = None

    @staticmethod
    def compute_hamming(vector1, vector2):
        distance = 0
        if len(vector1) != len(vector2):
            print('Hamming distance not computable, vectors of unequal length')
            return
        for i in range(len(vector1)):
            if vector1[i] != vector2[i]:
                distance += 1
        return distance

    @staticmethod
    def furthest_neighbor(neighbors):
        max_idx = 0
        for i in range(len(neighbors)):
            if neighbors[i][1] > neighbors[max_idx][1]:
                max_idx = i
        return max_idx

    def train(self, k=10):
        self.k = k
        pass

    def classify(self, attributes):
        nearest = []
        for i in range(len(self.examples)):
            dist = kNN.compute_hamming(self.examples[i], attributes)
            if len(nearest) < self.k:
                nearest.append((i, dist))
            else:
                idx = kNN.furthest_neighbor(nearest)
                if dist < nearest[idx][1]:
                    nearest[idx] = (i, dist)

        votes = {}
        for target in set(self.targets):
            votes[target] = 0
        for neighbor in nearest:
            votes[self.targets[neighbor[0]]] += 1
        max_key = self.targets[0]
        for key in votes:
            if votes[key] > votes[max_key]:
                max_key = key

        return max_key
