# require running preprocess.py
import itertools
import pandas as pd

from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords

NUM_TOPICS = 10
NUM_WORDS = 15
NER_WEIGHT = 10
stopwords = stopwords.words('english')

def make_LDA_model(ner_weight, year):
    df = pd.read_pickle('ner_result.pkl')
    df = df[df[' time'] > str(year)]
    df = df[df[' time'] < str(year+1)]
    # make df with ner count
    tokenized_with_ner_count = []
    for ner_dict, tokens in zip(df['ner'], df['tokenized_body']):
        ner_list = ner_dict['single_word'] + \
            list(itertools.chain.from_iterable(ner_dict['multi_word']))
        for word in ner_list:
            tokens.remove(word)
        tokens += ner_dict['single_word'] * ner_weight
        multi_words = [' '.join(x) for x in ner_dict['multi_word']]
        tokens += multi_words * ner_weight
        # filter out stopwords
        tokens = list(filter(lambda x: x not in stopwords, tokens))
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


def write_topics(model, year):
    topics = model.show_topics(num_topics=NUM_TOPICS, num_words=NUM_WORDS)
    data = []
    for t in topics:
        data.append(str(t[0]) + '\n' + t[1])

    data = list(map(lambda x: str(x.encode('utf-8')), data))
    with open('./logs/LdaTopics'+str(year)+'.log', 'w') as f:
        f.write('\n\n'.join(data))
    f.close()
    print("LDA model of year" + str(year) + "Topic log has been saved.")
    print(topics)


if __name__ == '__main__':
    for year in [2015, 2016, 2017]:
        model = make_LDA_model(NER_WEIGHT, year)
        model.save('./saves/ldamodel-' + str(year) + '.gensim')
        print('LDA model has just saved as ./saves/ldamodel-' + str(year) + '.gensim')
        write_topics(model, year)
