import json
import pandas as pd
from nltk.corpus import stopwords
from gensim.utils import lemmatize
from gensim.corpora import Dictionary
from gensim.models import LdaModel

stopwords = stopwords.words("english")


def preprocess():
    raw_data = [None] * 8
    for i in range(8):
        with open("data/koreaherald_1517_" + str(i) + ".json", 'r') as f:
            raw_data[i] = json.load(f)
        f.close()

    df_data = pd.DataFrame.from_dict(raw_data[0])
    for data in raw_data[1:]:
        df_data.append(data, ignore_index=True)


    # df_data = 'title', ' author', ' time', ' description', ' body', ' section'
    #df_data['preprocessed_body'] = preprocess_body(df_data[' body'], df_data)

    res = []
    for raw_text in df_data[' body']:
        #TODO: lemmatization - remove words with frequency less than certain threshold.
        res.append(lemmatize(raw_text, stopwords=stopwords))
    print("lemmatization done")
    df_data['lemmatized_body'] = res
    #TODO: bigram - phrases?
    dictionary = Dictionary(res)

    df_data['body_vector'] = df_data['lemmatized_body'].apply(dictionary.doc2bow)

    corpus = [vector for vector in df_data['body_vector']]




    id2word = dictionary.id2token
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=2000,
        alpha='auto',
        eta='auto',
        iterations=400,
        num_topics=20,
        passes=20,
        eval_every=None
    )
    topics = model.show_topics(num_topics=20, num_words=15)
    import pdb; pdb.set_trace()



preprocess()
