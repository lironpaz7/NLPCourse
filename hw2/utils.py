import numpy as np
import pandas as pd
import torch
from gensim import downloader
from torch.utils.data import Dataset
from tqdm import tqdm
from constants import TOKEN
import gensim.downloader as api


def read_data(path: str):
    """
    Reads the data in the given path and returns list of lists which contains the words. Also return a list containing the labels for each words
    :param path: path to the file
    :return: List[List[str]] -> List of sentences (each sentence is a list of words), List[int] -> The labels for each word
    """
    with open(path, encoding='utf-8') as f:
        sentences, labels = [], []
        words = []
        for line in f.readlines():
            line = line.replace('\n', '')
            if line == '':
                # new sentence
                sentences.append(words[:])
                words = []
            else:
                word, label = line.split('\t')
                label = 1 if label != 'O' else 0  # fixing the label
                words.append(word.lower())
                labels.append(label)
        return sentences, labels


def word_2_vec(word, embeds):
    word = word.lower()
    if word in embeds:
        vector = embeds[word]
    elif 'http' in word:
        vector = embeds['<url>']
    elif '@' in word:
        vector = embeds['<user>']
    else:
        vector = embeds['<unknown>']

    return vector


def down_sample(X_test: torch.Tensor, y_test: torch.Tensor, biggest_class: int = 0, leave: int = 10000):
    """
    Down samole the given data to make a more balanced data set
    :param X_test: Tensor
    :param y_test: Tensor
    :param biggest_class: class with the biggest num of samples
    :param leave: final number of samples to leave
    :return: X_test and y_test after down sampling to the given ratio
    """
    X_new, y_new = [], []
    X_test: list = X_test.tolist()
    y_test: list = y_test.tolist()
    running_num = 0
    for x, y in zip(X_test, y_test):
        if y == biggest_class:
            if running_num < leave:
                X_new.append(x)
                y_new.append(y)
                running_num += 1
        else:
            X_new.append(x)
            y_new.append(y)
    return torch.Tensor(np.array(X_new)), torch.Tensor(np.array(y_new))


def read_data_model2(path: str, embeds, return_words=False, test=False):
    """
    Reads the data in the given path and returns list of lists which contains the words. Also return a list containing the labels for each words
    :param test: If true then no tags are expected
    :param return_words: return the real words as well
    :param embeds: Glove words embedding
    :param path: path to the file
    :return: List[List[str]] -> List of sentences (each sentence is a list of words), List[int] -> The labels for each word
    """
    print('Preparing data for model...')
    words, labels, real_words = [], [], []
    with open(path) as f:
        for line in f.readlines():
            line_splitted = line.strip().split('\t')
            if not test:
                if len(line_splitted) == 2:
                    # we have a word and a label
                    word, label = line_splitted
                    real_words.append(word)
                    label = 1 if label != 'O' else 0  # fixing the label
                    words.append(word_2_vec(word, embeds))
                    labels.append(label)
            else:
                # we are in test mode so only words are expected with no tags at all
                if len(line_splitted) == 1:
                    word = line_splitted[0]
                    real_words.append(word)
                    words.append(word_2_vec(word, embeds))

    print('Finished preparing data for model...')
    if return_words:
        return torch.Tensor(np.array(words)), torch.Tensor(np.array(labels)), real_words
    else:
        return torch.Tensor(np.array(words)), torch.Tensor(np.array(labels))


def read_and_arrange_data(path: str) -> pd.DataFrame:
    """
    Reads the data in the given path and returns dataframe where each row is a sentence is it's labels
    :param path: path to the file
    :return: Dataframe of sentences and corresponding labels
    """
    with open(path, encoding='utf-8') as f:
        sentences, all_labels = [], []
        words, labels = [], []
        for line in f.readlines():
            line_splitted = line.strip().split('\t')
            if len(line_splitted) == 2:
                word, label = line_splitted
                label = '1' if label != 'O' else '0'  # fixing the label
                words.append(word.lower())
                labels.append(label)
            else:
                # finished sentence
                sentences.append(' '.join(words))
                all_labels.append(' '.join(labels))
                words, labels = [], []
        df = pd.DataFrame({
            'text': sentences, 'labels': all_labels
        })
        return df


def generate_context(sentences, method='pn'):
    result = []
    for sentence in sentences:
        sentence = [TOKEN] + sentence + [TOKEN]
        new_sentence = []
        n = len(sentence)
        for i in range(n):
            if method == 'pn':
                if i == 0 or i == n - 1:
                    continue
                new_word = sentence[i - 1] + ' ' + sentence[i] + ' ' + sentence[i + 1]
                new_sentence.append(new_word)
        result.append(new_sentence)
    return result


def vectorize_data(sentences, model, n=100):
    """
    Vectorizing the data by creating a matrix of vectors from the Word2Vec model
    :param sentences: List for Lists -> each list contains the words of a single sentence
    :param model: Word2Vec model (pretrained)
    :param n: If the word doesn't appear in the vocabulary we will generate an n-size uniform vector
    :return: vectorized data where each word represented as a vector from the given model
    """
    X = []
    for sentence in sentences:
        for word in sentence:
            if word in model:
                vec = model[word]
            else:
                # the word doesn't appear in the vocabulary, so we have to generate a new array to represent the word
                vec = [np.random.uniform() for _ in range(n)]
            X.append(vec)
    return X


def get_predict(X, model):
    predict = []
    for word_vec in tqdm(X, desc='Predicting...', ncols=100):
        p = model.predict([word_vec])
        predict.append(p[0])
    return predict


class WordsDataSet(Dataset):
    def __init__(self, file_path, n=100):
        self.sentences, self.labels = read_data(path=file_path)
        model_w2v = api.load(f"glove-twitter-{n}")
        self.sentences = vectorize_data(self.sentences, model_w2v, n=n)

    def __getitem__(self, item):
        sen = torch.FloatTensor(self.sentences[item]).squeeze()
        data = {"input_ids": sen, "labels": self.labels[item]}
        return data

    def __len__(self):
        return len(self.sentences)


def align_label(texts, labels, tokenizer, labels_to_ids, label_all_tokens):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, labels_to_ids, label_all_tokens):
        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i),
                                padding='max_length', max_length=512, truncation=True, return_tensors="pt") for i in
                      txt]
        self.labels = [align_label(i, j, tokenizer, labels_to_ids, label_all_tokens) for i, j in zip(txt, lb)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels


def output_save(words, predictions, path):
    predicted = []
    for pred in predictions:
        if pred == 0:
            predicted.append('O')
        else:
            predicted.append('I')
    f = open(path, 'w', encoding='utf-8')
    i = 0
    for w in words:
        f.write(f"{w}\t{predicted[i]}\n")
        i += 1
    f.close()


def load_data(path, glove):
    with open(path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    sentences = [sen.strip().lower() for sen in sentences]
    sentences = [sen.split() for sen in sentences if sen]

    X_vectors = []
    y_labels = []
    for sen in sentences:
        if sen[0] not in glove.key_to_index:
            continue
        vec = glove[sen[0]]
        X_vectors.append(vec)
        y_labels.append(0 if sen[1] == 'O' else 1)
    X_vectors = np.asarray(X_vectors)
    y_labels = np.asarray(y_labels)

    return X_vectors, y_labels


def load_dev_data(path, glove):
    with open(path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    sentences = [sen.strip().lower() for sen in sentences]
    sentences = [sen.split() for sen in sentences if sen]

    X_vectors = []
    y_labels = []
    for sen in sentences:
        if sen[0] not in glove.key_to_index:
            X_vectors.append(np.random.normal(0, 1, 200))
            y_labels.append(0)
        else:
            vec = glove[sen[0]]
            X_vectors.append(vec)
            y_labels.append(0 if sen[1] == 'O' else 1)
    X_vectors = np.asarray(X_vectors)
    y_labels = np.asarray(y_labels)

    return X_vectors, y_labels
