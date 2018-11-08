class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    # TODO: k-cross fold validation
    def k_cross_val(self):
        pass

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
