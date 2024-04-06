import SampleBiLSTM
from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np
import contractions # pip install contractions to fix contractions like you're to you are
from cleantext import clean # pip install clean-text to remove emojis
import re
import unicodedata
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, confusion_matrix

from keras.layers import Dropout, Dense, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.layers import Attention

# load model 
model = load_model('model.02-0.15.hdf5')

# check model info 
model.summary()

suicide_data = pd.read_csv("../Dataset/Suicide_Detection.csv")
print(len(suicide_data))

#small_dataset = pd.DataFrame()
list = []
suicide_class = 0
non_suicide_class = 0
for index in range(len(suicide_data)):
    df = suicide_data.loc[index]
    if (suicide_class < 50000) & (df['class'] == 'suicide'):
        suicide_class = suicide_class + 1
        list.append(df.squeeze())
    elif (non_suicide_class < 50000) & (df['class'] == 'non-suicide'):
        non_suicide_class = non_suicide_class + 1
        list.append(df.squeeze())

small_dataset = pd.DataFrame(list, columns=['text', 'class'])
print(small_dataset)

suicide_text = small_dataset['text']

lst = [suicide_data]
del suicide_data
del lst

def remove_emoji(text_data, suicide_text_removed_emoji):
    for text in text_data:
        suicide_text_removed_emoji.append(clean(text, no_emoji=True))

def remove_non_ascii_words(text_data, suicide_text_removed_non_ascii):
    for text in text_data:
        ascii_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        suicide_text_removed_non_ascii.append(ascii_text)

def remove_long_numbers(text_data, suicide_text_removed_long_numbers):
    for text in text_data:
        text = re.sub(r'[0-9]{10,}', '', text)
        suicide_text_removed_long_numbers.append(text)

def fix_contractions(text_data, suicide_text_fixed_contractions):
    contractions_fixed = []
    for text in text_data:
        sentences = sent_tokenize(text)
        contractions_fixed.clear()
        for line in sentences:
            contractions_fixed.append(contractions.fix(line))
        suicide_text_fixed_contractions.append("".join(contractions_fixed))

def convert_to_lower_case(text_data, suicide_text_lower):
    for text in text_data:
        suicide_text_lower.append(text.lower())

def remove_punctuation(text_data, suicide_text_no_punctuation):
    for text in text_data:
        text = re.sub(r'[^\w\s]', ' ', text)
        suicide_text_no_punctuation.append(text)

def remove_miscellaneous(text_data, suicide_remove_miscellaneous):
    chars = "\/*_{}[]()#+-!$';<>|:%=¸”&‚"
    for text in text_data:
        for c in chars:
            text = text.replace(c, " ")
        text = text.replace("filler", "'")
        text = text.replace("  ", " ")
        suicide_remove_miscellaneous.append(text)

def tokenize_text(text_data, suicide_tokenize_text):
    for text in text_data:
        sentences = sent_tokenize(text)

        sentence_tokens = []
        for line in sentences:
            tokens = (word_tokenize(line))
            sentence_tokens.extend(tokens)
            #sentence_tokens.append(" ".join(tokens))

        suicide_tokenize_text.append(sentence_tokens)
        #suicide_tokenize_text.append(" ".join(sentence_tokens))

def remove_stop_words(text_data, suicide_removed_stop_words):
    stop_words=set(stopwords.words('english'))
    for text in text_data:
        post = []
        for word in text:
            if word not in stop_words:
                post.append(word)

        #suicide_removed_stop_words.append(post)
        suicide_removed_stop_words.append(" ".join(post))

def lemmatize_verbs(text_data, suicide_text_lemmatized):
    lemmatizer = WordNetLemmatizer()
    for text in text_data:
        post = []
        for word in text:
            lemmatized_word = lemmatizer.lemmatize(word, pos='v')
            post.append(lemmatized_word)

        suicide_text_lemmatized.append(" ".join(post))     

def preprocess_text():
    suicide_text_removed_non_ascii = []
    remove_non_ascii_words(suicide_text, suicide_text_removed_non_ascii)

    suicide_text_removed_long_numbers = []
    remove_long_numbers(suicide_text_removed_non_ascii, suicide_text_removed_long_numbers)

    suicide_text_fixed_contractions = []
    fix_contractions(suicide_text_removed_long_numbers, suicide_text_fixed_contractions)

    suicide_text_lower = []
    convert_to_lower_case(suicide_text_fixed_contractions, suicide_text_lower)

    suicide_text_no_punctuation = []
    remove_punctuation(suicide_text_lower, suicide_text_no_punctuation)

    suicide_remove_miscellaneous = []
    remove_miscellaneous(suicide_text_no_punctuation, suicide_remove_miscellaneous)

    # Tokenize words
    suicide_tokenize_text = []
    tokenize_text(suicide_remove_miscellaneous, suicide_tokenize_text)
    #print(suicide_tokenize_text)

    suicide_removed_stop_words = []
    remove_stop_words(suicide_tokenize_text, suicide_removed_stop_words)

    #suicide_text_lemmatized = []
    #lemmatize_verbs(suicide_removed_stop_words, suicide_text_lemmatized)

    return suicide_removed_stop_words
    #return suicide_text_lemmatized


def encode_label():
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(small_dataset['class'])
    #print(suicide_data.loc[94])

def prepare_input_embeddings(X_train, X_test):
    suicide_text = np.concatenate((X_train, X_test), axis=0)
    keras_tokenizer = Tokenizer(num_words=TOKENIZER_NUM_WORDS)

    # Update the internal vocabulary
    keras_tokenizer.fit_on_texts(suicide_text)
    # Transform each post to sequence of integers
    word_sequences = keras_tokenizer.texts_to_sequences(suicide_text)

    word_index = keras_tokenizer.word_index
    print("Length of word index = ", len(word_index))

    # Pad sequences to equal length
    padded_word_sequence = pad_sequences(word_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    #print(padded_word_sequence.shape)

    # Separate glove embeddings for tain and test data
    train_sequence = padded_word_sequence[0:len(X_train), ]
    test_sequence = padded_word_sequence[len(X_train):, ]

    # create glove embedding dictionary
    embeddings_dictionary = {}
    filename = "../glove.6B/glove.6B." + str(GLOVE_EMBEDDING_DIMENSION) + "d.txt"
 
    with open(filename, encoding="utf8") as glove_file:
        for line in glove_file:
            embedding = line.split()
            key = embedding[0]
            embed_vector = np.asarray(embedding[1:], dtype='float32')
            embeddings_dictionary[key] = embed_vector
        
    glove_file.close()

    # Create embedding matrix from word_index and embeddings_dictionary
    word_index_length = len(word_index) + 1
    #print("Size of matrix = ", word_index_length*GLOVE_EMBEDDING_DIMENSION)
    embedding_matrix = np.arange(word_index_length*GLOVE_EMBEDDING_DIMENSION, dtype=float).reshape(word_index_length, GLOVE_EMBEDDING_DIMENSION)

    for word, index in word_index.items():
        if word in embeddings_dictionary.keys():
            embedding_matrix[index] = embeddings_dictionary.get(word)
        else:
            #print(word, " not found")
            embedding_matrix[index] = np.zeros(GLOVE_EMBEDDING_DIMENSION, dtype='float32')

    #print(np.shape(embedding_matrix)[0])
    #print(np.shape(embedding_matrix)[1])

    return (train_sequence, test_sequence, embedding_matrix)


def get_eval_report(labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1 = (2*(precision*recall))/(precision+recall)
    return {
        "mcc": mcc,
        "true positive": tp,
        "true negative": tn,
        "false positive": fp,
        "false negative": fn,
        "pricision" : precision,
        "recall" : recall,
        "F1" : f1,
        "accuracy": (tp+tn)/(tp+tn+fp+fn)
    }

def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds)

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string], '')
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()



small_dataset['text'] = preprocess_text()
#print(small_dataset['text'])
small_dataset['class'] = encode_label()
X = small_dataset['text']
Y = small_dataset['class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

MAX_SEQUENCE_LENGTH = small_dataset['text'].apply(lambda x: len(x.split())).max()
print("Max length = ", MAX_SEQUENCE_LENGTH)

TOKENIZER_NUM_WORDS = 100000
GLOVE_EMBEDDING_DIMENSION = 50
X_train_sequence, X_test_sequence, embedding_matrix = prepare_input_embeddings(X_train, X_test)


predicted = model.predict(X_test_sequence)[1]
cm=confusion_matrix(Y_test, predicted)
print(cm)