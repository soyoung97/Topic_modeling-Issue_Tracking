import os
import json
import pandas

from glob import glob
from gensim.models import LdaModel
from gensim.corpora import Dictionary

def get_articles(data_path, load=False):
    if load:
        return padnas.read_pickle('ner_result.pkl')
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

def main():
    df = get_articles('./data', load=True)
    model = get_LDA_model('./saves')
    dictionary = model.id2word
    other_texts = [
        ['Mexico', 'Korean', 'European'],
        ['US Air Force', 'New Zealand', 'French']]
    other_corpus = [dictionary.doc2bow(text) for text in other_texts]

    for unseen_doc in other_corpus:
        vector = model[unseen_doc]
        print(vector)



if __name__ == '__main__':
    main()
