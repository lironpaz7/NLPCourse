import os
import numpy as np
import torch
from gensim import downloader
from sklearn.metrics import f1_score
from utils import read_data_model2
from model2 import Q2NN


def run_test(test_path, predictions_path):
    """
    Running the evaluation process with the given weights and features and produces a prediction file
    :param test_path: path to the test file
    :param predictions_path: path to store the model predictions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'current device: {device}')

    if not os.path.exists('embeds.pkl'):
        print('Downloading glove embedding model...')
        embeds = downloader.load(f"glove-twitter-200")
        torch.save(embeds, 'embeds.pkl')
        print('Finished downloading glove embedding model...')
    else:
        print('Loading glove embedding model...')
        embeds = torch.load('embeds.pkl')

    print('Loading model...')
    model = torch.load('weights.pkl')
    test = False
    if 'test' in test_path:
        test = True

    X_test, y_test, X_real = read_data_model2(test_path, embeds, return_words=True, test=test)
    X_test = X_test.to(device)
    y_test = y_test.detach().numpy()
    y_pred = model(X_test).cpu()
    y_pred = torch.round(y_pred).detach().numpy()
    if not test:
        f1 = f1_score(y_true=y_test, y_pred=y_pred)
        print(f'F1 Score: {f1 :.3f}')
    print(f'Writing results to {predictions_path}...')
    with open(predictions_path, 'w') as f:
        for w, y in zip(X_real, y_pred):
            tag = 'O' if int(y[0]) == 0 else '1'
            f.write(f"{w}\t{tag}\n")


if __name__ == '__main__':
    path = 'data/test.untagged'
    print(f'Running evaluation on {path}...')
    run_test(path, 'comp.tagged')
    print('Finished!')
