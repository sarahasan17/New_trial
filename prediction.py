import joblib
import tensorflow
import numpy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer

from model import tokenizer, max_sequence_len
import numpy as np
def predict(data):
    seed_text = pd.DataFrame(data)[0]
    model = joblib.load('model.sav')
    next_words = 25
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list,
                                        verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

    seed_text += " " + output_word
    l=seed_text.split(' ')
    for i in range(0,len(l)-5,5):
        print(l[i],' ',l[i+1],' ',l[i+2],' ',l[i+3],' ',l[i+4])
