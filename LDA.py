#require running preprocess.py
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import itertools


def makeLDAmodel():
    df = pd.read_pickle('ner_result.pkl')
    #make df with ner count
    tokenized_with_ner_count = []
    for ner_dict, tokens in zip(df['ner'], df['tokenized_body']):
        ner_values = ner_dict.values()
        ner_list = ner_dict['single_word'] + list(itertools.chain.from_iterable(ner_dict['multi_word']))
        for word in ner_list:
            tokens.remove(word)
        tokens += ner_dict['single_word'] * 10
        multi_words = list(map(lambda x: ' '.join(x), ner_dict['multi_word']))
        tokens += multi_words * 10
        tokenized_with_ner_count.append(tokens)
    df['tokenized_with_ner_count'] = tokenized_with_ner_count
    dictionary = Dictionary(df['tokenized_with_ner_count'])
    df['body_vector'] = df['tokenized_with_ner_count'].apply(dictionary.doc2bow)
    corpus = [vector for vector in df['body_vector']]

    temp = dictionary[0]
    id2word = dictionary.id2token
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=2000,
        alpha='auto',
        iterations=400,
        num_topics=20,
        passes=20,
        eval_every=None
    )
    topics = model.show_topics(num_topics=20, num_words=15)
    data = []
    for t in topics:
        data.append(str(t[0]) + "\n" + t[1])
    with open("LDA_last.log", 'w') as f:
        f.write('\n\n'.join(data))
    f.close()
    return model
makeLDAmodel()
