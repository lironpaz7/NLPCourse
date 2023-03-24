import re

from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple

WORD = 0
TAG = 1


class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        feature_dict_list = ["f100", 'f101', 'f102', 'f103', 'f104', 'f105', 'f106',
                             'f107',
                             'feature_words_with_capital_letters',
                             'feature_words_with_numbers',
                             'feature_words_with_hyphens',
                             'feature_words_with_capitals_only',
                             'feature_nnword_ctag',
                             'feature_ppword_ctag',
                             'feature_pword_ptag_ctag',
                             'feature_ppword_pword_ctag',
                             'feature_nnword_nword_ctag',
                             'f100_lower',
                             'f101_lower',
                             'f101_lower_complete',
                             'f102_lower',
                             'f102_lower_complete',
                             'upper_tag_exist',
                             'hyphen_tag_exist',
                             'digit_tag_exist',
                             'cXXxx_dd',
                             'nXXxx_dd',
                             'pXXxx_dd']  # the feature classes used in the code
        self.feature_rep_dict = {fd: defaultdict(int) for fd in feature_dict_list}
        '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the test

    def features_lower_cases(self, cword, ctag, pword, nword):
        cword = cword.lower()
        pword = pword.lower()
        nword = nword.lower()
        # feature 100
        self.feature_rep_dict['f100_lower'][(cword, ctag)] += 1

        # feature 101 + 102
        for i in range(1, min(5, len(cword) + 1)):
            self.feature_rep_dict['f101_lower'][(cword[-i:], ctag)] += 1
            self.feature_rep_dict['f101_lower_complete'][(cword[:-i], ctag)] += 1
            self.feature_rep_dict['f102_lower'][(cword[:i], ctag)] += 1
            self.feature_rep_dict['f102_lower_complete'][(cword[i:], ctag)] += 1

        # feature 106
        self.feature_rep_dict['f106'][(pword, ctag)] += 1

        # feature 107
        self.feature_rep_dict['f107'][(nword, ctag)] += 1

    def calculate_f_functions(self, split_words, word_idx, pp_word, p_word, cur_word, pp_tag, p_tag, cur_tag):
        # f100
        self.feature_rep_dict["f100"][(cur_word, cur_tag)] += 1

        # f101
        for i in range(1, min(5, len(cur_word) + 1)):
            self.feature_rep_dict['f101'][(cur_word[-i:], cur_tag)] += 1

        # f102
        for i in range(1, min(5, len(cur_word) + 1)):
            self.feature_rep_dict['f102'][(cur_word[:i], cur_tag)] += 1

        # f103

        self.feature_rep_dict['f103'][(pp_tag, p_tag, cur_tag)] += 1

        # f104

        self.feature_rep_dict['f104'][(p_tag, cur_tag)] += 1

        # f105

        self.feature_rep_dict['f105'][cur_tag] += 1

        # f106
        self.feature_rep_dict['f106'][(p_word, cur_tag)] += 1

        # f107
        new_word = split_words[word_idx + 1].split('_')[0] if word_idx + 1 < len(split_words) else '*'

        self.feature_rep_dict['f107'][(new_word, cur_tag)] += 1

        # Capital Letters
        if bool(re.search(r'[A-Z]', cur_word)):
            self.feature_rep_dict['feature_words_with_capital_letters'][(cur_word, cur_tag)] += 1

        # Numbers
        if bool(re.search(r'\d', cur_word)):
            self.feature_rep_dict['feature_words_with_numbers'][(cur_word, cur_tag)] += 1
            self.feature_rep_dict['digit_tag_exist'][cur_tag] += 1

        # More features
        nnew_word = split_words[word_idx + 2].split('_')[0] if word_idx + 2 < len(split_words) else '*'

        self.feature_rep_dict['feature_ppword_ctag'][(pp_word, cur_tag)] += 1

        self.feature_rep_dict['feature_nnword_ctag'][(nnew_word, cur_tag)] += 1

        self.feature_rep_dict['feature_nnword_ctag'][(nnew_word, cur_tag)] += 1

        self.feature_rep_dict['feature_pword_ptag_ctag'][(p_word, p_tag, cur_tag)] += 1

        self.features_lower_cases(cur_word, cur_tag, p_word, new_word)

        self.feature_rep_dict['cXXxx_dd'][(translate_to_pattern(cur_word), cur_tag)] += 1
        self.feature_rep_dict['nXXxx_dd'][(translate_to_pattern(new_word), cur_tag)] += 1
        self.feature_rep_dict['pXXxx_dd'][(translate_to_pattern(cur_word), cur_tag)] += 1
        self.feature_rep_dict['feature_ppword_pword_ctag'][(pp_word, p_word, cur_tag)] += 1
        self.feature_rep_dict['feature_nnword_nword_ctag'][(pp_word, p_word, cur_tag)] += 1

    def get_word_tag_pair_count(self, file_path) -> None:
        """
            Extract out of text all word/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                # initialize history
                p_word, pp_word, p_tag, pp_tag = '*', '*', '*', '*'
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    self.tags.add(cur_tag)
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1

                    # calculate f100-f107:
                    self.calculate_f_functions(split_words, word_idx, pp_word, p_word, cur_word, pp_tag, p_tag, cur_tag)

                    # update the word tags and current
                    pp_word, p_word, pp_tag, p_tag = p_word, cur_word, p_tag, cur_tag

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0])
                    self.histories.append(history)


def translate_to_pattern(word: str):
    def trans_letter(c: str):
        if c.isupper():
            return 'X'
        if c.islower():
            return 'x'
        if c.isdigit():
            return 'd'
        return c

    return ''.join([trans_letter(ch) for ch in word])


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.feature_to_idx = {
            "f100": defaultdict(int),
            # spelling features for prefixes/suffixes of length <= 4
            'f101': defaultdict(int),
            'f102': defaultdict(int),

            # Contextual Features
            'f103': defaultdict(int),
            'f104': defaultdict(int),
            'f105': defaultdict(int),
            'f106': defaultdict(int),
            'f107': defaultdict(int),

            # Custom Features
            'feature_words_with_capital_letters': defaultdict(int),
            'feature_words_with_numbers': defaultdict(int),

            'feature_words_with_hyphens': defaultdict(int),
            'feature_words_with_capitals_only': defaultdict(int),

            'feature_nnword_ctag': defaultdict(int),
            'feature_ppword_ctag': defaultdict(int),
            'feature_pword_ptag_ctag': defaultdict(int),
            'feature_ppword_pword_ctag': defaultdict(int),
            'feature_nnword_nword_ctag': defaultdict(int),

            'f100_lower': defaultdict(int),
            'f101_lower': defaultdict(int),
            'f101_lower_complete': defaultdict(int),
            'f102_lower': defaultdict(int),
            'f102_lower_complete': defaultdict(int),

            'upper_tag_exist': defaultdict(int),
            'hyphen_tag_exist': defaultdict(int),
            'digit_tag_exist': defaultdict(int),

            'cXXxx_dd': defaultdict(int),
            'nXXxx_dd': defaultdict(int),
            'pXXxx_dd': defaultdict(int),
        }
        self.represent_input_with_features = OrderedDict()
        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx
        """
        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)


def append_mandatory_features(history, features, dic_features):
    """
    get history and feature list and append all the features that we were asked to implement and exist in the
    given history.
    """
    # print(history)
    ppword, pword, cword, nword, nnword, pptag, ptag, ctag = history
    add_to_list((cword, ctag), 'f100', features, dic_features)  # feature 100
    # feature 101 + 102
    for i in range(1, min(6, len(cword) + 1)):
        add_to_list((cword[-i:], ctag), 'f101', features, dic_features)  # feature 101
        add_to_list((cword[:i], ctag), 'f102', features, dic_features)  # feature 102

    add_to_list((pptag, ptag, ctag), 'f103', features, dic_features)  # feature 103
    add_to_list((ptag, ctag), 'f104', features, dic_features)  # feature 104
    add_to_list(ctag, 'f105', features, dic_features)  # feature 105
    add_to_list((pword, ctag), 'f106', features, dic_features)  # feature 106
    add_to_list((nword, ctag), 'f107', features, dic_features)  # feature 107
    # feature words with capital letters and digits
    add_to_list((cword, ctag), 'feature_words_with_capital_letters', features, dic_features)
    add_to_list((cword, ctag), 'feature_words_with_numbers', features, dic_features)
    add_to_list(ctag, 'digit_tag_exist', features, dic_features)


def append_lower_cases_features(history, features, dict_of_dicts):
    """
    get history and feature list and append all the features that we were asked to implement and exist in the
    given history.
    """
    ppword, pword, cword, nword, nnword, pptag, ptag, ctag = history
    cword = cword.lower()
    pword = pword.lower()
    nword = nword.lower()
    add_to_list((cword, ctag), 'f100_lower', features, dict_of_dicts)  # feature 100
    # feature 101 + 102
    for i in range(1, min(5, len(cword) + 1)):
        add_to_list((cword[-i:], ctag), 'f101_lower', features, dict_of_dicts)  # feature 101
        add_to_list((cword[:-i], ctag), 'f101_lower_complete', features, dict_of_dicts)  # feature 101
        add_to_list((cword[:i], ctag), 'f102_lower', features, dict_of_dicts)  # feature 102
        add_to_list((cword[i:], ctag), 'f102_lower_complete', features, dict_of_dicts)  # feature 102

    add_to_list((pword, ctag), 'f106', features, dict_of_dicts)  # feature 106
    add_to_list((nword, ctag), 'f107', features, dict_of_dicts)  # feature 107


def append_custom_features(history, features, dict_of_dicts):
    append_lower_cases_features(history, features, dict_of_dicts)
    pp_word, p_word, cur_word, new_word, nnew_word, pptag, p_tag, cur_tag = history

    add_to_list((pp_word, cur_tag), 'feature_ppword_ctag', features, dict_of_dicts)
    if nnew_word is not None:
        add_to_list((nnew_word, cur_tag), 'feature_nnword_ctag', features, dict_of_dicts)
    add_to_list((p_word, p_tag, cur_tag), 'feature_pword_ptag_ctag', features, dict_of_dicts)
    add_to_list((translate_to_pattern(cur_word), cur_tag), 'cXXxx_dd', features, dict_of_dicts)
    add_to_list((translate_to_pattern(new_word), cur_tag), 'nXXxx_dd', features, dict_of_dicts)
    add_to_list((translate_to_pattern(cur_word), cur_tag), 'pXXxx_dd', features, dict_of_dicts)
    add_to_list((pp_word, p_word, cur_tag), 'feature_ppword_pword_ctag', features, dict_of_dicts)
    add_to_list((pp_word, p_word, cur_tag), 'feature_nnword_nword_ctag', features, dict_of_dicts)


def add_to_list(hist: tuple, feature_name: str, features: list, dic_features):
    if hist in dic_features[feature_name]:
        features.append(dic_features[feature_name][hist])


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]]) \
        -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features that are relevant to the given history
    """
    features = []
    if len(history) == 7:
        cword, ctag, pword, ptag, ppword, pptag, nword = history
        history = (ppword, pword, cword, nword, None, pptag, ptag, ctag)
        append_mandatory_features(history, features, dict_of_dicts)
        append_custom_features(history, features, dict_of_dicts)
    else:
        append_mandatory_features(history, features, dict_of_dicts)
        append_custom_features(history, features, dict_of_dicts)
    return features


def preprocess_train(train_path, threshold):
    # Statistics
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for dict_key in feature2id.feature_to_idx:
        print(dict_key, len(feature2id.feature_to_idx[dict_key]))
    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences
