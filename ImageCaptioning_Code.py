import numpy as np
from numpy import array
import pandas as pd
import string
import os
import glob
import pickle
from PIL import Image
from pickle import load
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout
from keras.layers.merge import add
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.text import Tokenizer

sys_path = "." #Useful with Colab path

def load_doc(filename):
    file = open(filename, 'r', encoding='cp437')
    text = file.read()
    file.close()
    return text[14:]


filename = sys_path + "/VizWiz_Data_train1/annotations_train1.txt"
doc = load_doc(filename)
print(doc[:300])

test_filename = sys_path + "/VizWiz_Data_val/annotations_val.txt"
test_doc = load_doc(test_filename)
print(test_doc[:300])


def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping


train_descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(train_descriptions))

test_descriptions = load_descriptions(test_doc)
print('Loaded: %d ' % len(test_descriptions))


def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = 'startseq ' + ' '.join(desc) + ' endseq'


clean_descriptions(train_descriptions)
clean_descriptions(test_descriptions)

images = sys_path + '/VizWiz_Data_train1/train1/'
train_img = glob.glob(images + '*.jpg')

test_images = sys_path + '/VizWiz_Data_val/val/val/'
test_img = glob.glob(test_images + '*.jpg')

################################################ Image Pickles #########################################################
def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

model = InceptionV3(weights='imagenet')

model_new = Model(model.input, model.layers[-2].output)

def encode(image):
    image = preprocess(image)
    fea_vec = model_new.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

encoding_train = {}
for img in train_img:
    encoding_train[img[len(images):]] = encode(img)

with open(sys_path + "/Pickle/encoded_train_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)

encoding_test = {}
for img in test_img:
    encoding_test[img[len(test_images):]] = encode(img)

with open(sys_path + "/Pickle/encoded_test_images.pkl", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)
########################################################################################################################


train_features = load(open(sys_path + "/Pickle/encoded_train_images.pkl", "rb"))
print('Photos: train=%d' % len(train_features))

test_features = load(open(sys_path + "/Pickle/encoded_test_images.pkl", "rb"))
print('Photos: test=%d' % len(test_features))

all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
len(all_train_captions)

all_test_captions = []
for key, val in test_descriptions.items():
    for cap in val:
        all_test_captions.append(cap)
len(all_test_captions)

word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

test_word_count_threshold = 10
test_word_counts = {}
test_nsents = 0
for test_sent in all_test_captions:
    test_nsents += 1
    for test_w in test_sent.split(' '):
        test_word_counts[test_w] = test_word_counts.get(test_w, 0) + 1

test_vocab = [test_w for test_w in test_word_counts if test_word_counts[test_w] >= test_word_count_threshold]
print('preprocessed words %d -> %d' % (len(test_word_counts), len(test_vocab)))

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

test_ixtoword = {}
test_wordtoix = {}

test_ix = 1
for test_w in test_vocab:
    test_wordtoix[test_w] = test_ix
    test_ixtoword[test_ix] = test_w
    test_ix += 1

vocab_size = len(ixtoword) + 1
print(vocab_size)

test_vocab_size = len(test_ixtoword) + 1
print(test_vocab_size)

def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

test_max_length = max_length(test_descriptions)
print('Description Length Test: %d' % test_max_length)

max_length = max_length(train_descriptions)
print('Description Length Train: %d' % max_length)

def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key, desc_list in descriptions.items():
            n += 1
            photo = photos[key + '.jpg']
            for desc in desc_list:
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n = 0

glove_dir = sys_path + '/glove.6B.200d.txt'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 200

embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=2)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


ans_out = pd.DataFrame(columns=['Model', 'Accuracy', 'BELU1', 'BELU2'])

for j in range(8): # j=1-8 for 8 different models
    ans_out.loc[j + 1, 'Model'] = j + 1
    if j == 0:
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        model.layers[2].set_weights([embedding_matrix])
        model.layers[2].trainable = False

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif j == 1:
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.2)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(0.2)(se1)
        se3 = LSTM(256)(se2)
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        model.layers[2].set_weights([embedding_matrix])
        model.layers[2].trainable = False

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif j == 2:
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        fe3 = Dense(128, activation='relu')(fe2)
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256, return_sequences=True)(se2)
        se4 = LSTM(128)(se3)
        decoder1 = add([fe3, se4])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        model.layers[2].set_weights([embedding_matrix])
        model.layers[2].trainable = False

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif j == 3:
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.2)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        fe3 = Dense(128, activation='relu')(fe2)
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(0.2)(se1)
        se3 = LSTM(256, return_sequences=True)(se2)
        se4 = LSTM(128)(se3)
        decoder1 = add([fe3, se4])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        model.layers[2].set_weights([embedding_matrix])
        model.layers[2].trainable = False

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif j == 4:
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        fe3 = Dense(128, activation='relu')(fe2)
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256, return_sequences=True)(se2)
        se4 = LSTM(128)(se3)
        decoder1 = add([fe3, se4])
        decoder2 = Dense(256, activation='relu')(decoder1)
        decoder3 = Dense(128, activation='relu')(decoder2)
        outputs = Dense(vocab_size, activation='softmax')(decoder3)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        model.layers[2].set_weights([embedding_matrix])
        model.layers[2].trainable = False

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif j == 5:
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.2)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        fe3 = Dense(128, activation='relu')(fe2)
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(0.2)(se1)
        se3 = LSTM(256, return_sequences=True)(se2)
        se4 = LSTM(128)(se3)
        decoder1 = add([fe3, se4])
        decoder2 = Dense(256, activation='relu')(decoder1)
        decoder3 = Dense(128, activation='relu')(decoder2)
        outputs = Dense(vocab_size, activation='softmax')(decoder3)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        model.layers[2].set_weights([embedding_matrix])
        model.layers[2].trainable = False

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif j == 6:
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        fe3 = Dense(128, activation='relu')(fe2)
        fe4 = Dense(64, activation='relu')(fe3)
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256, return_sequences=True)(se2)
        se4 = LSTM(128, return_sequences=True)(se3)
        se5 = LSTM(64)(se4)
        decoder1 = add([fe4, se5])
        decoder2 = Dense(256, activation='relu')(decoder1)
        decoder3 = Dense(128, activation='relu')(decoder2)
        outputs = Dense(vocab_size, activation='softmax')(decoder3)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        model.layers[2].set_weights([embedding_matrix])
        model.layers[2].trainable = False

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif j == 7:
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.2)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        fe3 = Dense(128, activation='relu')(fe2)
        fe4 = Dense(64, activation='relu')(fe3)
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
        se2 = Dropout(0.2)(se1)
        se3 = LSTM(256, return_sequences=True)(se2)
        se4 = LSTM(128, return_sequences=True)(se3)
        se5 = LSTM(64)(se4)
        decoder1 = add([fe4, se5])
        decoder2 = Dense(256, activation='relu')(decoder1)
        decoder3 = Dense(128, activation='relu')(decoder2)
        outputs = Dense(vocab_size, activation='softmax')(decoder3)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        model.layers[2].set_weights([embedding_matrix])
        model.layers[2].trainable = False

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 50
    number_pics_per_bath = 32
    steps = len(train_descriptions) // number_pics_per_bath

    for i in range(epochs):
        generator = data_generator(train_descriptions, train_features, wordtoix, max_length, number_pics_per_bath)
        test_generator = data_generator(test_descriptions, test_features, test_wordtoix, test_max_length,
                                        number_pics_per_bath)
        history = model.fit(generator, epochs=1, steps_per_epoch=steps, validation_data=test_generator,
                            validation_steps=len(test_descriptions) // number_pics_per_bath, verbose=2)

    _, acc = model.evaluate(test_generator, steps=len(test_descriptions) // number_pics_per_bath, verbose=2)
    print('> %.3f' % (acc * 100.0))
    ans_out.loc[j + 1, 'Accuracy'] = acc * 100.0

    model.save_weights(sys_path + '/model_weights/model_' + str(j + 1) + '_50.h5')

    model.load_weights(sys_path + '/model_weights/model_' + str(j + 1) + '_50.h5')

    with open(sys_path + "/Pickle/encoded_test_images.pkl", "rb") as encoded_pickle:
        encoding_test = load(encoded_pickle)

    actual, predicted = list(), list()

    for key in tqdm(list(test_descriptions.keys())):
        captions = test_descriptions[key]
        y_pred = greedySearch(encoding_test[key + '.jpg'].reshape((1, 2048)))
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        actual.append(actual_captions)
        predicted.append(y_pred)

    # BLEU score
    print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    ans_out.loc[j + 1, 'BELU1'] = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    ans_out.loc[j + 1, 'BELU2'] = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))

ans_out.to_csv('./Model_Ans.csv')


############################# Example Caption Outcomes ################################
model.load_weights(sys_path + '/model_weights/model_7_50.h5')
images = sys_path + '/VizWiz_Data_val/val/val/'
with open(sys_path + "/Pickle/encoded_test_images.pkl", "rb") as encoded_pickle:
    encoding_test = load(encoded_pickle)

z = 305
pic = list(encoding_test.keys())[z]
image = encoding_test[pic].reshape((1,2048))
x = plt.imread(images+pic)
plt.imshow(x)
plt.show()
print("Outcome:", greedySearch(image))
print("Actual:")
print(test_descriptions[pic[:-4]][0][9:-7])
print(test_descriptions[pic[:-4]][1][9:-7])
print(test_descriptions[pic[:-4]][2][9:-7])
print(test_descriptions[pic[:-4]][3][9:-7])
print(test_descriptions[pic[:-4]][4][9:-7])
######################################################################################
