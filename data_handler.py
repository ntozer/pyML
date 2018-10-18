import csv

class DataHandler:
    def __init__(self):
        pass

    def csv2matrix(self, filename):
        data_matrix = []
        with open('data/{}'.format(filename)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                data_matrix.append(row)
        return data_matrix
