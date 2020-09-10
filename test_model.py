from collections import Counter
import pandas as pd
import numpy as np
from math import ceil
import torch
from sklearn.utils import shuffle
from Neural_Architecture import LSTM_architecture
import argparse

def createArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=argparse.FileType('r'))
    return parser

def read_data():
    n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
    data_positive = pd.read_csv('./dataset/positive.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])
    data_negative = pd.read_csv('./dataset/negative.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])

    sample_size = 50000
    texts_withoutshuffle = np.concatenate((data_positive['text'].values[:sample_size],
                           data_negative['text'].values[:sample_size]), axis=0)
    labels_withoutshuffle = np.asarray([1] * sample_size + [0] * sample_size)
    assert len(texts_withoutshuffle) == len(labels_withoutshuffle)
    texts,labels = shuffle(texts_withoutshuffle, labels_withoutshuffle, random_state=0)

    return texts, labels

texts, labels = read_data()

def tokenize():
    punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
    all_texts = 'separator'.join(texts)
    all_texts = all_texts.lower()
    all_text = ''.join([c for c in all_texts if c not in punctuation])
    texts_split = all_text.split('separator')
    all_text = ' '.join(texts_split)
    words = all_text.split()
    return words

def get_vocabulary():
    words = tokenize()
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    return vocab, vocab_to_int

def tokenize_text(test_text):
    punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
    test_text = test_text.lower()
    test_text = ''.join([c for c in test_text if c not in punctuation])
    test_words = test_text.split()
    new_text = []
    for word in test_words:
        if (word[0] != '@') & ('http' not in word) & (~word.isdigit()):
            new_text.append(word)
    test_ints = []
    _, vocab_to_int = get_vocabulary()
    mas_to_int = []
    for word in new_text:
        if word in vocab_to_int:
            mas_to_int.append(vocab_to_int[word])
    test_ints.append(mas_to_int)

    return test_ints


def add_pads(texts_ints, seq_length):
    features = np.zeros((len(texts_ints), seq_length), dtype=int)

    for i, row in enumerate(texts_ints):
        if len(row) == 0:
            continue
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features

def predict(net, test_text, sequence_length=30):
    net.eval()
    test_ints = tokenize_text(test_text)
    seq_length = sequence_length
    features = add_pads(test_ints, seq_length)
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)
    h = net.init_hidden_state(batch_size)
    output, h = net(feature_tensor, h)

    pred = torch.round(output.squeeze())
    pos_prob = output.item()

    if (pred.item() == 1):
        result = "Позитивное сообщение"
    else:
        result = "Негативное сообщение"

    return result, pos_prob

if __name__ == "__main__":
    parser = createArgParser()
    args = parser.parse_args()
    assert args.i != None, "Для запуска тестирования без веб-приложения приложите текстовый файл с помощью ключа -i"
    file_name = args.i.name
    f = open(file_name, 'r')
    data = f.readlines()
    input_text = ''.join(data)
    f.close()

    vocab_size = 194345
    output_size = 1
    embedding_dim = 200
    hidden_dim = 128
    number_of_layers = 2
    model = LSTM_architecture(vocab_size, output_size, embedding_dim, hidden_dim, number_of_layers)
    model.load_state_dict(torch.load("model", map_location=torch.device('cpu')))
    seq_length = 30

    type_of_tonal, pos_prob = predict(model, input_text, seq_length)
    if type_of_tonal == "Негативное сообщение":
        prob = ceil((1 - pos_prob) * 100)
    else:
        prob = ceil(pos_prob * 100)
    print("Окраска - {}, вероятность = {}%".format(type_of_tonal, prob))