import pandas as pd
import numpy as np
import contractions # pip install contractions to fix contractions like you're to you are
#from cleantext import clean # pip install clean-text to remove emojis
import re
import unicodedata
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from transformers import DistilBertTokenizerFast
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments

suicide_data = pd.read_csv("../Dataset/Suicide_Detection.csv")
print(len(suicide_data))

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
#print(small_dataset)

suicide_text = small_dataset['text'].copy()
lst = [suicide_data]
del suicide_data
del lst


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

def preprocess_text():
    suicide_text_removed_non_ascii = []
    remove_non_ascii_words(suicide_text, suicide_text_removed_non_ascii)

    suicide_text_removed_long_numbers = []
    remove_long_numbers(suicide_text_removed_non_ascii, suicide_text_removed_long_numbers)

    suicide_text_fixed_contractions = []
    fix_contractions(suicide_text_removed_long_numbers, suicide_text_fixed_contractions)

    suicide_text_lower = []
    convert_to_lower_case(suicide_text_fixed_contractions, suicide_text_lower)

    suicide_remove_miscellaneous = []
    remove_miscellaneous(suicide_text_lower, suicide_remove_miscellaneous)

    return suicide_remove_miscellaneous

def encode_label():
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(small_dataset['class'])


small_dataset['text'] = preprocess_text()
small_dataset['class'] = encode_label()

X=small_dataset['text'].tolist()
y=small_dataset['class'].tolist()

X_train, X_testVal, y_train, y_testVal = train_test_split(X, y, test_size = 0.20, random_state = 0)
X_val, X_test, y_val, y_test = train_test_split(X_testVal, y_testVal, test_size = 0.5, random_state = 0)


#print(y_val)
suicide_class = 0
for index in range(len(y_val)):
    suicide_class = suicide_class + y_val[index]
print(suicide_class)

#print(y_test)
suicide_class = 0
for index in range(len(y_test)):
    suicide_class = suicide_class + y_test[index]
print(suicide_class)


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(X_train, truncation=True, padding=True)
val_encodings = tokenizer(X_val, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    y_val
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
))

training_args = TFTrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    eval_steps = 10,
)

with training_args.strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = TFTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()

trainer.evaluate(val_dataset)

trainer.predict(test_dataset)

trainer.predict(test_dataset)[1].shape

output=trainer.predict(test_dataset)[1]

cm=confusion_matrix(y_test, output)
print(cm)

trainer.save_model('DistilBERT_model')