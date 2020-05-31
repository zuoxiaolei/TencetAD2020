'''
__project_ = 'tencentAD2020'
__author__ = zuoxiaolei
__time__ = '2020/5/31 17:04'
__description = ''
'''
import logging

import pandas as pd
from gensim.corpora import Dictionary
from gensim.sklearn_api import LdaTransformer, TfIdfTransformer
import joblib
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def train_lda_model():
    data = pd.read_csv("corpus/part-00000-8274d92c-217e-4ce7-80c7-50c52a899545-c000.csv", header=None)
    data.columns = ["sentence"]
    sentence = data["sentence"].values.tolist()
    sentence = map(lambda x: x.split(), sentence)
    dct = Dictionary.load('data/lda_dict')
    # dct = Dictionary(sentence)
    # dct.save('./data/lda_dict')
    sentence = list(map(lambda x: dct.doc2bow(x), sentence))
    model = LdaTransformer(num_topics=100,
                              id2word=dct,
                              random_state=1)
    model.fit(sentence)
    joblib.dump(model, './data/lda.model')


def train_lda_model():
    data = pd.read_csv("corpus/part-00000-8274d92c-217e-4ce7-80c7-50c52a899545-c000.csv", header=None)
    data.columns = ["sentence"]
    sentence = data["sentence"].values.tolist()
    sentence = map(lambda x: x.split(), sentence)
    dct = Dictionary.load('data/lda_dict')
    # dct = Dictionary(sentence)
    # dct.save('./data/lda_dict')
    sentence = list(map(lambda x: dct.doc2bow(x), sentence))
    model = TfIdfTransformer(dictionary=dct, id2word=dct)
    model.fit(sentence)
    joblib.dump(model, './data/lda.model')

if __name__ == '__main__':
    train_lda_model()
