# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from Neural_Architecture import LSTM_architecture

### Считывание данных

n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
data_positive = pd.read_csv('./dataset/positive.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])
data_negative = pd.read_csv('./dataset/negative.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])

### Формирование сбалансированного датасета

sample_size = 40000
reviews_withoutshuffle = np.concatenate((data_positive['text'].values[:sample_size],
                           data_negative['text'].values[:sample_size]), axis=0)
labels_withoutshuffle = np.asarray([1] * sample_size + [0] * sample_size)

assert len(reviews_withoutshuffle) == len(labels_withoutshuffle)
from sklearn.utils import shuffle
reviews,labels = shuffle(reviews_withoutshuffle, labels_withoutshuffle, random_state=0)

### Токенизация

def tokenize():
  punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
  all_reviews = 'separator'.join(reviews)
  all_reviews = all_reviews.lower()
  all_text = ''.join([c for c in all_reviews if c not in punctuation])
  texts_split = all_text.split('separator')
  all_text = ' '.join(texts_split)
  words = all_text.split()
  return words, texts_split

words, texts_split = tokenize()

new_reviews = []
for review in texts_split:
    review = review.split()
    new_text = []
    for word in review:
        if (word[0] != '@') & ('http' not in word) & (~word.isdigit()):
            new_text.append(word)
    new_reviews.append(new_text)

counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii - 1 for ii, word in enumerate(vocab, 1)}
reviews_ints = []
for review in new_reviews:
    reviews_ints.append([vocab_to_int[word] for word in review])

def add_pads(reviews_ints, seq_length):
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        if len(row) == 0:
            continue
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features

### Разделение на обучающую, валидационную и тестовую выборки

features = add_pads(reviews_ints, seq_length=30)
split_frac = 0.8 # 80% на обучающую выборку

split_idx = int(len(features)*split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = labels[:split_idx], labels[split_idx:]
test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
batch_size = 50
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

### Определение режима: GPU или CPU

train_gpu=torch.cuda.is_available()

###Выбор гиперпараметров и инициализация сети

vocab_size = len(vocab_to_int)+1
output_size = 1
embedding_dim = 100
hidden_dim = 128
n_layers = 2
model = LSTM_architecture(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

### Обучение модели

epochs = 4 #оптимальное количество эпох для того, чтобы модель достаточно обучилась, но не переобучилась
counter = 0
batch_num = 100
clip = 5
if (train_gpu):
    model.cuda()
num_correct = 0
model.train()
for e in range(epochs):
    h = model.init_hidden_state(batch_size)
    for inputs, labels in train_loader:
        num_correct = 0
        counter += 1
        if(train_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()
        h = tuple([each.data for each in h])
        model.zero_grad()
        output, h = model.forward(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if counter % batch_num == 0:
            val_h = model.init_hidden_state(batch_size)
            val_losses = []
            model.eval()
            for inputs, labels in valid_loader:
                val_h = tuple([each.data for each in val_h])
                if(train_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()
                output, val_h = model(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())

                #accuracy
                pred = torch.round(output.squeeze())
                correct_tensor = pred.eq(labels.float().view_as(pred))
                correct = np.squeeze(correct_tensor.numpy()) if not train_gpu else np.squeeze(correct_tensor.cpu().numpy())
                num_correct += np.sum(correct)
                valid_acc = num_correct/len(valid_loader.dataset)

            model.train()
            print("Epoch: {} ;".format(e+1),
                  "Batch Number: {};".format(counter),
                  "Train Loss: {:.4f} ;".format(loss.item()),
                  "Valid Loss: {:.4f} ;".format(np.mean(val_losses)),
                  "Valid Accuracy: {:.4f}".format(valid_acc))

# PATH = "model"
# torch.save(model.state_dict(), PATH)