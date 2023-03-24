import random
import numpy as np
import torch
from gensim import downloader
from torch.utils.data.dataset import Dataset
from collections import defaultdict
import matplotlib.pyplot as plt

from hw3.constants import *

torch.manual_seed(28)
torch.cuda.manual_seed(28)
random.seed(28)
np.random.seed(28)


def get_vocabs(paths):
    """
        Extract vocabs from given datasets.
        :param paths:
        :param file_paths: a list with a full path for all corpuses
            Return:
              - words vocabulary : list
              - pos_tags vocabulary : list
    """
    word_vocab, pos_vocab = set(), set()
    word_vocab.add("ROOT")
    pos_vocab.add("ROOT")
    for path in paths.values():
        with open(path) as f:
            for line in f:
                if line == "\n":
                    continue
                splited_words = line.split("\t")
                word = splited_words[1]
                pos_tag = splited_words[3]
                word_vocab.add(word)
                pos_vocab.add(pos_tag)
    return list(word_vocab), list(pos_vocab)


def get_word_dict(paths):
    word_dict = defaultdict(int)
    word_dict["ROOT"] = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                if line == "\n":
                    word_dict["ROOT"] += 1
                    continue
                splited_words = line.split("\t")
                word = splited_words[1]
                word_dict[word] += 1
    return word_dict


def get_embeddings_from_glove(word_vocabulary, path_to_glove='glove-twitter-200', random_size=200):
    encoder_model = downloader.load(path_to_glove)
    glove_embeddings = [np.float32(np.random.normal(size=random_size))]
    for word in word_vocabulary:
        if word in encoder_model.key_to_index:
            glove_embeddings.append(encoder_model[word])
        else:
            glove_embeddings.append(np.float32(np.random.normal(size=random_size)))
    return np.stack(glove_embeddings, axis=0)


class PosDataReader:
    def __init__(self, file_path, word_vocabulary, pos_vocabulary):
        self.file_path = file_path
        self.word_vocabulary = word_vocabulary
        self.pos_vocabulary = pos_vocabulary
        self.sentences = []
        self.headers = []
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file_path, 'r') as f:
            cur_sentence = [("ROOT", "ROOT")]
            curr_headers = []
            for line in f:
                if line == "\n":
                    self.sentences.append(cur_sentence)
                    self.headers.append(curr_headers)
                    cur_sentence = [("ROOT", "ROOT")]
                    curr_headers = []
                    continue
                splited_word = line.split("\t")
                cur_word = splited_word[1]
                cur_pos = splited_word[3]
                header = splited_word[6]
                cur_sentence.append((cur_word, cur_pos))
                curr_headers.append(int(header))

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


class PosDataset(Dataset):
    def __init__(self, file_path: str, word_vocabulary, pos_vocabulary):
        super().__init__()
        self.datareader = PosDataReader(file_path, word_vocabulary, pos_vocabulary)
        self.word_idx_mappings = self.init_word_embeddings(self.datareader.word_vocabulary)
        self.pos_idx_mappings = self.init_pos_vocab(self.datareader.pos_vocabulary)
        self.sentences_dataset = self.convert_sentences_to_dataset()
        self.headers = self.datareader.headers

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, sentence_len = self.sentences_dataset[index]
        header = self.headers[index]
        return word_embed_idx, pos_embed_idx, sentence_len, header

    @staticmethod
    def __build_word_embeds(vocab):
        mapping = {UNK: 0}
        for index, word in enumerate(vocab, start=1):
            mapping[word] = index
        return mapping

    @staticmethod
    def init_word_embeddings(word_vocabulary):
        return PosDataset.__build_word_embeds(word_vocabulary)

    @staticmethod
    def init_pos_vocab(pos_vocabulary):
        pos_idx_mappings = {UNK: 0}
        for i, pos in enumerate(sorted(pos_vocabulary)):
            pos_idx_mappings[str(pos)] = int(i + 1)
        return pos_idx_mappings

    def __convert_to_dict(self, sentence_word_idx_list, sentence_pos_idx_list, sentence_len_list):
        d = {ind: samp for ind, samp in
             enumerate(zip(sentence_word_idx_list, sentence_pos_idx_list, sentence_len_list))}
        return d

    def convert_sentences_to_dataset(self):
        sentence_word_idx_list, sentence_pos_idx_list, sentence_len_list = [], [], []
        for _, sentence in enumerate(self.datareader.sentences):
            words_idx_list, pos_idx_list = [], []
            for word, pos in sentence:
                words_idx_list.append(self.word_idx_mappings.get(word))
                pos_idx_list.append(self.pos_idx_mappings.get(pos))
            sentence_len = len(words_idx_list)
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_len_list.append(sentence_len)
        return self.__convert_to_dict(sentence_word_idx_list, sentence_pos_idx_list, sentence_len_list)


class CompDataset(Dataset):
    def __init__(self, file_path, word_idx_mapping, pos_idx_mapping, path='data/comp.unlabeled'):
        self.file_path = file_path
        self.word_idx_mapping = word_idx_mapping
        self.pos_idx_mapping = pos_idx_mapping
        self.sentences = []
        self.sentences_word_idx = []
        self.sentences_pos_idx = []
        self.path = path
        self.__readData__()

    def __readData__(self):
        with open(self.path, 'r') as file:
            sentence = [('ROOT', 'ROOT')]
            word_idx = [self.word_idx_mapping['ROOT']]
            pos_idx = [self.pos_idx_mapping['ROOT']]
            for line in file:
                if line != '\n':
                    words = line.split("\t")
                    c_word, c_tag = words[1], words[3]
                    sentence.append((c_word, c_tag))
                    if c_word in self.word_idx_mapping.keys():
                        word_idx.append(self.word_idx_mapping[c_word])
                    else:
                        word_idx.append(self.word_idx_mapping[UNK])
                    if c_tag in self.pos_idx_mapping.keys():
                        pos_idx.append(self.pos_idx_mapping[c_tag])
                    else:
                        pos_idx.append(self.pos_idx_mapping[UNK])
                else:
                    self.sentences.append(sentence)
                    self.sentences_word_idx.append(word_idx)
                    self.sentences_pos_idx.append(pos_idx)
                    sentence = [('ROOT', 'ROOT')]
                    word_idx = [self.word_idx_mapping['ROOT']]
                    pos_idx = [self.pos_idx_mapping['ROOT']]
                    continue

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences_word_idx[index], self.sentences_pos_idx[index], self.sentences[index]


def plot_one(train_, test_, title='', ylabel='', xlabel=''):
    plt.plot(train_, c="green", label="Train")
    plt.plot(test_, c="red", label="Test")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_graphs(stats: dict):
    print('Generating plots...')
    plot_one(stats['train_acc'], stats['test_acc'], title='UAS over Epochs', xlabel='Epochs', ylabel='UAS')
    plot_one(stats['train_loss'], stats['test_loss'], title='Loss over Epochs', xlabel='Epochs', ylabel='Loss')
