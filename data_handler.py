import csv
import copy


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
                    # TODO currently hardcoded to remove '?' from breastcancer data, need to implement bagging
                    if row[idx] == '?':
                        row[idx] = 0
                    row[idx] = int(row[idx])
                self.dataset.append(row)

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
    def column(matrix, i):
        return [row[i] for row in matrix]

    @staticmethod
    def rm_column(matrix, i):
        for row in matrix:
            del row[i]
