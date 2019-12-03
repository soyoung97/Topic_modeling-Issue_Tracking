import os
import json
import pandas

from glob import glob
from tqdm import tqdm
from collections import Counter
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor


def get_articles(data_path, load=False):
    if load:
        return pandas.read_pickle('ner_result.pkl')
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
    model = LdaModel.load(os.path.join(path, 'model.gensim'))
    return model


def classify_docs(df, model):
    clsfyd = [list() for _ in range(model.num_topics)]

    dictionary = model.id2word
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        body = dictionary.doc2bow(row['tokenized_body'])
        vector = model[body]
        cat = max(vector, key=lambda x: x[1])[0]
        clsfyd[cat].append(row)

    return clsfyd


def get_parts(doc, query):
    try:
        return doc.get_top_answer(query).get_parts_as_text()
    except IndexError:
        return None


def get_5w1h(row, extractor):
    # doc = Document(row['title'], row[' description'],
    #                row[' body'], row[' time'])
    doc = Document.from_text(row[' body'], row[' time'])
    doc = extractor.parse(doc)

    return {
        'when': get_parts(doc, 'when'),
        'where': get_parts(doc, 'where'),
        'what': get_parts(doc, 'what'),
        'who': get_parts(doc, 'who'),
        'how': get_parts(doc, 'how')
    }


def get_issue_stats(rows, extractor):
    when_counter = Counter()
    where_counter = Counter()
    what_counter = Counter()
    who_counter = Counter()
    how_counter = Counter()

    for row in tqdm(rows):
        res = get_5w1h(row, extractor)

        when_counter.update({res['when']: 1})
        where_counter.update({res['where']: 1})
        what_counter.update({res['what']: 1})
        who_counter.update({res['who']: 1})
        how_counter.update({res['how']: 1})

    return {
        'when': when_counter,
        'where': where_counter,
        'what': what_counter,
        'who': who_counter,
        'how': how_counter
    }


def main():
    df = get_articles('./data', load=True)
    model = get_LDA_model('./saves')
    clsfyd = classify_docs(df, model)

    extractor = MasterExtractor()
    for cat, item in enumerate(clsfyd):
        res = get_issue_stats(item, extractor)

        print('===Category %d===' % cat)
        for key in res.keys():
            print(key, res[key].most_common(3))

    # dictionary = model.id2word
    # other_texts = [
    #     ['Mexico', 'Korean', 'European'],
    #     ['US Air Force', 'New Zealand', 'French']]
    # other_corpus = [dictionary.doc2bow(text) for text in other_texts]

    # for unseen_doc in other_corpus:
    #    vector = model[unseen_doc]
    #    print(vector)


if __name__ == '__main__':
    main()
