from nltk.corpus import reuters
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np
import random
import pickle

from glob import glob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from scipy import spatial, sparse

NER_WEIGHT = 10
NUM_TOPICS = 10
NUM_WORDS = 50
stopwords = stopwords.words("english")


def load_similar_data(config={}):

    documents = reuters.fileids()
    test = [d for d in documents if d.startswith('test/')]
    train = [d for d in documents if d.startswith('training/')]
    total = test + train
    raw = {}
    raw['text'] = [reuters.raw(doc_id) for doc_id in total]
    raw['label'] = [reuters.categories(doc_id) for doc_id in total]
    labels = reuters.categories()
    docs = {'text' : [], 'label' : []}
    for label in labels:
        texts = list(filter(lambda x: raw['label'][x[0]] == [label], enumerate(raw['text'])))

        docs['text'] = docs['text'] + list(map(lambda y: y[1], texts))
        docs['label'] = docs['label'] + list(map(lambda y: label, texts))
    return pd.DataFrame(docs)

def load_reuters_ner():
    return pd.read_pickle('reuters_ner_result.pkl')

def compute_similarity(adict, fpath):
    pass

def preprocess(df_data):
    res = []
    for raw_text in df_data['text']:
        res.append(word_tokenize(raw_text)) # TODO: what if we do this on other library spacey?because it is said this does well
    print('tokenization done')
    df_data['tokenized_body'] = res
    # save the output
    df_data.to_pickle('reuters_preprocess_result.pkl')
    return df_data

def tokenize(word_list):
    res = [None] * len(word_list)
    for i, body in enumerate(word_list):
        res[i] = word_tokenize(body)
    return res


def make_dict(df):
    # make df with ner count
    tokenized_with_ner_count = []
    for token_list in df:
        # add ner with promoted frequency
        total_list = token_list
        # filter out stopwords
        tokens = list(filter(lambda x: x not in stopwords, total_list))
        tokenized_with_ner_count.append(tokens)
    return Dictionary(tokenized_with_ner_count)

def bow2vec(dictionary, texts, clen):
    tlen = texts.shape[0]
    sp = sparse.dok_matrix((tlen, clen + 1))
    for i in range(tlen):
        bow = dictionary.doc2bow(texts.iloc[i])
        wordSum = sum(map(lambda x: x[1], bow))
        for wordIdx, wordCount in bow:
            sp[i, wordIdx]  = float(wordCount) / wordSum
    return sp

def compute_dist(reuters, o_result, load = True):
    if load:
        with open('innerdist.pickle', 'rb') as f:
            data = pickle.load(f)
        with open('outerdist.pickle', 'rb') as f:
            odata = pickle.load(f)
        return data, odata
    labelset = set(reuters.label.tolist())
    dictionary = make_dict(reuters['neuroner_tokenized'].append(o_result['neuroner_tokenized'], ignore_index = True))
    clen = max(dictionary.keys())
    print(clen)
    reutersv = bow2vec(dictionary, reuters['neuroner_tokenized'], clen)
    targetv = bow2vec(dictionary, o_result['neuroner_tokenized'], clen)
    innerdist = {}
    outerdist = {}
    tolen = o_result.shape[0]
    oindexarr = []
    if tolen <= 50:
        oindexarr = list(range(tolen))
    else:
        oindexarr = [random.choice(list(range(tolen))) for i in range(50)]
        tolen = 50
    for l in labelset:
        rel = reuters[reuters.label == l]
        tlen = rel.shape[0]
        indexarr = []
        if tlen <= 20:
            indexarr = list(range(tlen))
        else:
            indexarr = [random.choice(list(range(tlen))) for i in range(20)]
            tlen = 20
        sdt = None
        for i in range(tlen):
            for j in range(tlen):
                av = reutersv[rel.iloc[indexarr[i]].name,:]
                bv = reutersv[rel.iloc[indexarr[j]].name,:]
                dt = ((av - bv) * (av - bv).transpose())[0,0]
                if sdt is None or dt > sdt:
                    sdt = dt
        innerdist[l] = sdt
        sodt = None
        for i in range(tlen):
            for j in range(tolen):
                av = reutersv[rel.iloc[indexarr[i]].name,:]
                bv = targetv[o_result.iloc[oindexarr[j]].name,:]
                dt = ((av - bv) * (av - bv).transpose())[0,0]
                if sodt is None or dt < sodt:
                    sodt = dt
        outerdist[l] = sodt
        print("%s done" % l)
    with open('innerdist.pickle', 'wb') as f:
        pickle.dump(innerdist, f)
    with open('outerdist.pickle', 'wb') as f:
        pickle.dump(outerdist, f)
    return innerdist, outerdist

def LDA_with_neuroner(ner_weight, df):
    # make df with ner count
    tokenized_with_ner_count = []
    for ner_list, token_list in zip(df['neuroner_list'], df['neuroner_tokenized']):
        # add ner with promoted frequency
        total_list = token_list + ner_list * ner_weight
        # filter out stopwords
        tokens = list(filter(lambda x: x not in stopwords and x not in "\.,()" and '\\xe2' not in x, total_list))
        tokenized_with_ner_count.append(tokens)
    df['tokenized_with_ner_count'] = tokenized_with_ner_count
    dictionary = Dictionary(df['tokenized_with_ner_count'])
    df['body_vector'] = df['tokenized_with_ner_count'].apply(
        dictionary.doc2bow)
    corpus = [vector for vector in df['body_vector']]

    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        chunksize=2000,
        alpha='auto',
        iterations=400,
        num_topics=NUM_TOPICS,
        passes=20,
        eval_every=None
    )
    return model



if __name__ == '__main__':
    reuters = load_reuters_ner()
    o_result = pd.read_pickle('neuroner_result.pkl')
    innerdist, outerdist = compute_dist(reuters, o_result, True)
    p = []
    for i in innerdist.keys():
        p.append((innerdist[i] + outerdist[i], i))
    p.sort()
    df = pd.DataFrame([], columns = reuters.columns)
    c = 0
    for i in p:
        if (reuters[reuters.label == i[1]].shape[0] > 5):
            print(i[1])
            df = df.append(reuters[reuters.label == i[1]])
            c += 1
        if c == 10:
            break
    #df.to_pickle('reuters_top10.pkl')

    #model = LDA_with_neuroner(NER_WEIGHT, df)
    #dictionary = model.id2word
    #print(df['tokenized_body'])
    #tc = [dictionary.doc2bow(text) for text in df['neuroner_tokenized']]
    #topic_assign = []
    #for ins in tc:
    #    vector = model.get_document_topics(ins)
    #    idx = np.argmax(list(map(lambda x: x[1], vector)))
    #    topic_assign.append(vector[idx][0])
    #x = df['label'].tolist()
    #res = []
    #for i in range(len(x)):
    #    res.append((x[i], topic_assign[i]))
    #topics = model.show_topics(num_topics=NUM_TOPICS, num_words=NUM_WORDS)
    #data = []
    #for t in topics:
    #    data.append(str(t[0]) + '\n' + t[1])

    #data = list(map(lambda x: str(x.encode('utf-8')), data))
    #with open('./logs/moreLdaTopicsReuters.log', 'w') as f:
    #    f.write('\n\n'.join(data))
    #f.close()
    #for t in topics:
    #    t2 = map(lambda x: x.split("*")[1], t[1].split(" + "))
    #    print(t[0], ",".join(t2))

