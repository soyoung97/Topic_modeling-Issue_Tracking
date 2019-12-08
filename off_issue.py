import os
import json
import pandas
import datetime

from glob import glob
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from sklearn.cluster import DBSCAN
from scipy import spatial, sparse
import math
import numpy as np
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor

def get_articles(data_path, load=False):
    if load:
        return pandas.read_pickle('neuroner_result.pkl')
    total_df = None
    for fname in glob(os.path.join(data_path, '*.json')):
        with open(fname, 'r') as f:
            data = json.load(f)
            temp_df = pandas.DataFrame.from_dict(data)

            if total_df is None:
                total_df = temp_df
            else:
                total_df = total_df.append(temp_df, ignore_index=True)

    return total_df

def get_LDA_model(path):
    model = LdaModel.load(os.path.join(path, 'removedneuronerldamodel-2017.gensim'))
    return model

def topic_numbering(load = False):
    if load:
        return pandas.read_pickle('topic_numbering.pkl')
    df = get_articles('./data', load=True)
    model = get_LDA_model('./saves')
    dictionary = model.id2word
    #print(df['tokenized_body'])
    tc = [dictionary.doc2bow(text) for text in df['neuroner_tokenized']]
    topic_assign = []
    for ins in tc:
        vector = model.get_document_topics(ins)
        idx = np.argmax(list(map(lambda x: x[1], vector)))
        topic_assign.append(vector[idx][0])
    df['topic_num'] = topic_assign
    df.to_pickle('topic_numbering.pkl')
    return df

def convert_timestamp(s):
    return int(datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timestamp())


def get_answer_from_doc(rep_doc):
    extractor = MasterExtractor()
    doc = Document(rep_doc['title'], rep_doc[' description'], rep_doc[' body'], rep_doc[' time'])
    doc = extractor.parse(doc)
    try:
        top_who_answer = doc.get_top_answer('who').get_parts_as_text()
    except:
        top_who_answer = 'unknown'
    try:
        top_what_answer = doc.get_top_answer('what').get_parts_as_text()
    except:
        top_what_answer = 'unknown'
    try:
        top_when_answer = doc.get_top_answer('when').get_parts_as_text()
    except:
        top_when_answer = 'unknown'
    try:
        top_where_answer = doc.get_top_answer('where').get_parts_as_text()
    except:
        top_where_answer = 'unknown'
    try:
        top_why_answer = doc.get_top_answer('why').get_parts_as_text()
    except:
        top_why_answer = 'unknown'
    try:
        top_how_answer = doc.get_top_answer('how').get_parts_as_text()
    except:
        top_how_answer = 'unknown'

    return (top_who_answer, top_what_answer, top_when_answer, top_where_answer, top_why_answer, top_how_answer)


EPS = 15
MSAMPLE = 3
WVec = 150.0
WTime = 0.0000005

def bow2vec(dictionary, texts, clen):
    tlen = texts.shape[0]
    sp = sparse.dok_matrix((tlen, clen + 1))
    for i in range(tlen):
        bow = dictionary.doc2bow(texts.iloc[i])
        wordSum = sum(map(lambda x: x[1], bow))
        for wordIdx, wordCount in bow:
            sp[i, wordIdx]  = wordCount * WVec / wordSum
    return sp

def do_clustering(model, topic0):
    topic0['timestamp'] = topic0[' time'].map(convert_timestamp)
    tlen = topic0.shape[0]
    timedata = topic0['timestamp'].to_numpy().reshape(tlen, 1)
    dictionary = model.id2word
    clen = max(dictionary.keys())
    tc = bow2vec(dictionary, topic0['neuroner_tokenized'], clen)
    for i in range(tlen):
        tc[i, clen] = topic0['timestamp'].iloc[i] * WTime
    clustering = DBSCAN(eps=EPS, min_samples=MSAMPLE).fit_predict(tc)
    return clustering

def get_rep_doc(model, docs):
    dictionary = model.id2word
    clen = max(dictionary.keys())
    tc = bow2vec(dictionary, docs['neuroner_tokenized'], clen)
    tlen = docs.shape[0]
    for i in range(tlen):
        tc[i, clen] = docs['timestamp'].iloc[i] * WTime
    mi = 0
    mv = None
    #for i in range(tlen):
    #    p = 0
    #    for j in range(tlen):
    #        cd = tc[i, :] - tc[j, :]
    #        p = p + math.sqrt((cd.transpose() * cd)[0,0])
    #    if mv is None or p < mv:
    #        mi = i
    #        mv = p
    return docs.iloc[mi]

def main():
    df = topic_numbering(True)
    model = get_LDA_model('./saves')
    topic0 = df[df.topic_num == 1]
    clustering = do_clustering(model, topic0)
    topic0['event'] = clustering
    #print("clusters: %d " % (max(clustering) + 1))
    for i in range(max(clustering) + 1):
        rep_doc = get_rep_doc(model, topic0[topic0.event == i])
        #print(topic0[topic0.event == i])
        (top_who_answer, top_what_answer, top_when_answer, top_where_answer, top_why_answer, top_how_answer) = get_answer_from_doc(rep_doc)
        print('====================')
        print('event %d' % i)
        print('who: %s' % top_who_answer)
        print('what: %s' % top_what_answer)
        print('when: %s' % top_when_answer)
        print('where: %s' % top_where_answer)
        print('why: %s' % top_why_answer)
        print('how: %s' % top_how_answer)
        print('--------------------')


if __name__ == '__main__':
    main()


