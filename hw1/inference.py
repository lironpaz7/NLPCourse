from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm


def q_and_e_func(weight_vector, features, sentence, idx, p_tag_list, pp_tag_list):
    """
    Creates matrix of probabilities from histories generated from pp_tag_list, p_tag_list and all labels.
    :param weight_vector:
    :param features:
    :param sentence: the given list of words
    :param idx: index (k-1)
    :param p_tag_list:
    :param pp_tag_list:
    :return: matrix of probabilities in the format of: pp_tag_list, p_tag_list, |Y|
    """
    rows, cols = [], []
    cnt = 0
    # generate the previous and next words
    pp_word = '*' if idx - 2 < 0 else sentence[idx - 2]
    p_word = '*' if idx - 1 < 0 else sentence[idx - 1]
    n_word = '*' if idx + 1 >= len(sentence) else sentence[idx + 1]
    nn_word = '*' if idx + 2 >= len(sentence) else sentence[idx + 2]
    c_word = sentence[idx]
    for pp in pp_tag_list:  # S[k-2]
        for p in p_tag_list:  # S[k-1]
            for curr_tag in features.feature_statistics.tags:
                feat = represent_input_with_features((pp_word, p_word, c_word, n_word, nn_word, pp, p, curr_tag),
                                                     features.feature_to_idx)
                cols.extend(feat)
                rows.extend((cnt + np.zeros_like(feat)).tolist())
                cnt += 1
    # Optimizations using sparse matrix
    mat = csr_matrix((np.ones_like(cols), (rows, cols)), shape=(cnt, features.n_total_features), dtype=bool)
    s = np.exp(mat @ weight_vector).reshape(len(pp_tag_list), len(p_tag_list),
                                                 len(features.feature_statistics.tags))
    return s / (s.sum(axis=2).reshape(len(pp_tag_list), len(p_tag_list), -1))


def memm_viterbi(sentence, pre_trained_weights, feature2id, B=2):
    """
    Viterbi algorithm implementation as we have seen in class with optimization of q calculation
    :param sentence: List of words
    :param pre_trained_weights: Pretrained vector of weights of features
    :param feature2id: Include features statistics and build features vector from given history
    :param B: Beam search parameter. Default is 2
    :return:
    """
    Sk = list(feature2id.feature_statistics.tags)
    n = len(sentence)
    pi, bp = {(0, '*', '*'): 1}, {}
    S = {-1: ['*'], 0: ['*']}
    for k in range(1, n + 1):
        probabilities = defaultdict(int)
        q_mul_e = q_and_e_func(pre_trained_weights, feature2id, sentence, k - 1, S[k - 1], S[k - 2])
        for iu, u in enumerate(S[k - 1]):
            for iv, v in enumerate(Sk):
                W = np.array([pi[(k - 1, w, u)] * q_mul_e[iw, iu, iv] for iw, w in enumerate(S[k - 2])])
                pi[(k, u, v)] = W.max()
                bp[(k, u, v)] = S[k - 2][W.argmax()]

        # Beam Search Implementation
        for p, v in sorted([(pi[(k, u, v)], v) for v in Sk for u in S[k - 1]], reverse=True)[:B * 2]:
            probabilities[v] += p
        S[k] = sorted(probabilities, key=lambda x: probabilities[x], reverse=True)[:B]
    # init y - that is the vector of final labels
    y = [''] * n
    _, u, v = max({key: pi[key] for key in pi if key[0] == n}, key=lambda key: pi[key])  # (n, u, v)
    y[n - 1] = v
    if (n - 2) >= 0:
        y[n - 2] = u
    # dynamic algorithm to set the right label
    for k in range(n - 3, -1, -1):
        y[k] = bp[(k + 3, y[k + 1], y[k + 2])]
    return np.array(y[2:])


def confusion_matrix_top_k(labels, confusion_matrix, k=10):
    """
    Prints the confusion matrix top k elements
    :param labels: Word labels
    :param confusion_matrix: Confusion matrix (numpy array)
    :param k: top k labels to display
    """
    n = len(labels)
    confusion_matrix_copy = confusion_matrix.copy()
    confusion_matrix_copy[np.arange(n), np.arange(n)] = 0
    wrongs_idx = confusion_matrix_copy.sum(axis=1).argsort()[-k:].tolist()
    wrong_cols = [labels[idx_wrong] for idx_wrong in wrongs_idx]
    wrongs_idx.reverse()
    wrong_cols.reverse()
    print(pd.DataFrame(confusion_matrix[wrongs_idx, :][:, wrongs_idx].astype(int), wrong_cols, wrong_cols))


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)
    output_file = open(predictions_path, "a+")
    labels = list(feature2id.feature_statistics.tags)
    n = len(feature2id.feature_statistics.tags)
    label2id = {label: i for i, label in enumerate(labels)}
    confusion_matrix = np.zeros((n, n))
    dic_counts_errors = defaultdict(int)
    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        preds = memm_viterbi(sentence, pre_trained_weights, feature2id)
        sentence = sentence[2:]
        tags = sen[1][2:]
        for i in range(len(preds)):
            if i > 0:
                output_file.write(" ")
            tag, pred, word = tags[i], preds[i], sentence[i]
            output_file.write(f"{word}_{pred}")
            if tag != pred:
                dic_counts_errors[word] += 1
            if tagged:
                confusion_matrix[label2id[tag], label2id[pred]] += 1
        output_file.write("\n")
    if tagged:
        # prints the confusion matrix and calculates the accuracy
        confusion_matrix_top_k(labels, confusion_matrix, k=10)
        accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()
        print(f'Accuracy: {accuracy}')
    output_file.close()
