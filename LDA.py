#require running preprocess.py
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel



def makeLDAmodel():
    df = pd.read_pickle('preprocess_result.pkl')
    dictionary = Dictionary(df['lemmatized_body'])
    df['body_vector'] = df['lemmatized_body'].apply(dictionary.doc2bow)
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
    print("LDA done")
    print(topics)
    return model

makeLDAmodel()
