from data_handler import DataHandler
from model_evaluator import ModelEvaluator
from models import *


def build_model(model_name, data_handler):
    if model_name == 'NaiveBayes':
        return NaiveBayes(data_handler.examples, data_handler.targets)
    if model_name == 'BaggingNB':
        return BaggingNB(data_handler.examples, data_handler.targets)
    if model_name == 'ID3':
        return ID3(data_handler.examples, data_handler.targets)
    if model_name == 'AdaBoost':
        return AdaBoost(data_handler.examples, data_handler.targets)
    if model_name == 'RandomForest':
        return RandomForest(data_handler.examples, data_handler.targets)
    if model_name == 'kNN':
        return kNN(data_handler.examples, data_handler.targets)


def run_model(model_name, data):
    dh = DataHandler()
    dh.import_data(data)
    dh.create_targets(-1)
    model = build_model(model_name, dh)
    m_eval = ModelEvaluator(model)
    acc, std = m_eval.n_time_k_cross_fold(10, 5)
    print('Accuracy: {}\nStandard Deviation: {}\n'.format(acc, std))
