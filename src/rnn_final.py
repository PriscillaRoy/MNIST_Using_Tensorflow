
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys


def preds_fn(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


#getting the file
File_path = get_file('alad10.txt', origin='/Users/priscilla/Downloads/alad10.txt')
data = open(File_path).read().lower() # converting to lower case
print('Length of File', len(data))

sort_text = sorted(list(set(data)))
print('total chars:', len(sort_text))
dict_keys = dict((c, i) for i, c in enumerate(sort_text))
dict_keys_reverse = dict((i, c) for i, c in enumerate(sort_text))

# cutting the text
maximum_length = 40
cut_value = 3
combine_total_text = []
next_chars = []
for i in range(0, len(data) - maximum_length, cut_value):
    combine_total_text.append(data[i: i + maximum_length])
    next_chars.append(data[i + maximum_length])
print('Sequence', len(combine_total_text))


X = np.zeros((len(combine_total_text), maximum_length, len(sort_text)), dtype=np.bool)
y = np.zeros((len(combine_total_text), len(sort_text)), dtype=np.bool)
for i, sentence in enumerate(combine_total_text):
    for t, char in enumerate(sentence):
        X[i, t, dict_keys[char]] = 1
    y[i, dict_keys[next_chars[i]]] = 1

#add layer
model = Sequential()
model.add(LSTM(128, input_shape=(maximum_length, len(sort_text))))
model.add(Dense(len(sort_text)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)



for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=128,
              epochs=1)

    start_index = random.randint(0, len(data) - maximum_length - 1)

    for difference in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('Diff : ', difference)

        text = ''
        sentence = data[start_index: start_index + maximum_length]
        text += sentence
        print( sentence )
        sys.stdout.write(text)

        for i in range(400):
            x = np.zeros((1, maximum_length, len(sort_text)))
            for t, char in enumerate(sentence):
                x[0, t, dict_keys[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = preds_fn(preds, difference)
            next_char = dict_keys_reverse[next_index]

            text += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()