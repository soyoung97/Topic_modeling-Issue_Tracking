# require running preprocess.py
import itertools
import pandas as pd

from gensim.models import LdaModel
from gensim.corpora import Dictionary

NUM_TOPICS = 20
NUM_WORDS = 15
ALPHA = 10


def make_LDA_model(alpha):
    df = pd.read_pickle('ner_result.pkl')
    # make df with ner count
    tokenized_with_ner_count = []
    for ner_dict, tokens in zip(df['ner'], df['tokenized_body']):
        ner_list = ner_dict['single_word'] + \
            list(itertools.chain.from_iterable(ner_dict['multi_word']))
        for word in ner_list:
            tokens.remove(word)
        tokens += ner_dict['single_word'] * alpha
        multi_words = [' '.join(x) for x in ner_dict['multi_word']]
        tokens += multi_words * alpha
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


def write_topics(model):
    topics = model.show_topics(num_topics=NUM_TOPICS, num_words=NUM_WORDS)
    data = []
    for t in topics:
        data.append(str(t[0]) + "\n" + t[1])

    with open("LDA_last.log", 'w') as f:
        f.write('\n\n'.join(data))
    f.close()


if __name__ == '__main__':
    model = make_LDA_model(ALPHA)

    model.save('./saves/model.gensim')

    write_topics(model)
