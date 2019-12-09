import os
import json
import pandas
import datetime
import argparse

from glob import glob
from tqdm import tqdm
from collections import Counter
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--offset', type=int, default=0)


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


def get_LDA_model(path, year):
    model = LdaModel.load(os.path.join(
        path, 'removedneuronerldamodel-%d.gensim' % year))
    return model


def split_by_year(df):
    df['timestamp'] = pandas.to_datetime(df[' time'])
    df = df.sort_values(by=['timestamp'])
    year = df['timestamp'].dt.to_period('Y')
    return df.groupby(year)


def split_by_quarter(df):
    year = df['timestamp'].dt.to_period('Q')
    return df.groupby(year)


def split_by_month(df):
    month = df['timestamp'].dt.to_period('M')
    return df.groupby(month)


def classify_docs(df, model):
    # quarter_docs = split_by_quarter(df)
    quarter_docs = split_by_month(df)
    clsfyd = [[pandas.DataFrame() for _ in range(model.num_topics)]
              for _ in quarter_docs]

    dictionary = model.id2word
    for q_idx, (_, quarter) in tqdm(enumerate(quarter_docs), total=len(quarter_docs)):
        for _, row in quarter.iterrows():
            body = dictionary.doc2bow(row['tokenized_body'])
            vector = model[body]
            cat = max(vector, key=lambda x: x[1])[0]
            clsfyd[q_idx][cat] = clsfyd[q_idx][cat].append(row)
    return clsfyd


def get_parts(doc, query):
    try:
        return doc.get_top_answer(query).get_parts_as_text()
    except:
        return 'unknown'


def get_5w1h(row, extractor):
    doc = Document(row[1]['title'], row[1][' description'],
                   row[1][' body'], row[1][' time'])
    # doc = Document.from_text(row[' body'], row[' time'])
    try:
        doc = extractor.parse(doc)

        return {
            'when': get_parts(doc, 'when'),
            'where': get_parts(doc, 'where'),
            'what': get_parts(doc, 'what'),
            'who': get_parts(doc, 'who'),
            'why': get_parts(doc, 'why'),
            'how': get_parts(doc, 'how')
        }
    except RuntimeError:
        return None


def get_issue_stats(rows, extractor):
    when_counter = Counter()
    where_counter = Counter()
    what_counter = Counter()
    who_counter = Counter()
    why_counter = Counter()
    how_counter = Counter()

    for row in tqdm(rows.iterrows(), total=rows.shape[0]):
        res = get_5w1h(row, extractor)

        if not res:
            continue

        when_counter.update({res['when']: 1})
        where_counter.update({res['where']: 1})
        what_counter.update({res['what']: 1})
        who_counter.update({res['who']: 1})
        why_counter.update({res['why']: 1})
        how_counter.update({res['how']: 1})

    return {
        'when': when_counter,
        'where': where_counter,
        'what': what_counter,
        'who': who_counter,
        'why': why_counter,
        'how': how_counter
    }


def get_score(counter, word):
    if word == 'unknown':
        return 0
    elif word in counter:
        return counter[word]
    else:
        return 0


def get_top_headline(counter, rows, extractor):
    max_row = None
    max_score = 0
    for row in tqdm(rows.iterrows(), total=rows.shape[0]):
        res = get_5w1h(row, extractor)

        if not res:
            continue
        score = 0
        for query in res:
            score += get_score(counter[query], res[query])

        if max_score < score:
            max_score = score
            max_row = row

    return max_row[1]['title'], get_5w1h(max_row, extractor)


TO_FIND = {
    0: [2],  # 2015
    1: [3],  # 2016
    2: []  # 2017
}


def __main(offset):
    df = get_articles('./data', load=True)
    clsfyd = list()
    for i, yeardf in split_by_year(df):
        if int(i.year) == 2018:
            continue
        model = get_LDA_model('./saves', int(i.year))
        clsfyd += classify_docs(yeardf, model)
    print(len(clsfyd))
    resf = open('res_on_issue_month.txt', 'w')

    extractor = MasterExtractor()
    for idx, quarter in enumerate(clsfyd):
        resf.write('===Quarter %d===\n' % idx)
        for cat, docs in enumerate(quarter):
            if cat in TO_FIND[idx // 12]:
                res = get_issue_stats(docs, extractor)

                resf.write('===Category %d===\n' % cat)
                for key in res:
                    resf.write('%s: %s\n' %
                               (key, str(res[key].most_common(3))))
                top_headline, top_res = get_top_headline(res, docs, extractor)
                resf.write('Headline: %s\n' % top_headline)
                for key in top_res:
                    resf.write('%s: %s\n' % (key, top_res[key]))
                resf.flush()
    resf.close()


if __name__ == '__main__':
    __main(parser.parse_args().offset)
