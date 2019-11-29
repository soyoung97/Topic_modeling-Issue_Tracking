import os
import json
import pickle
import pandas as pd

from glob import glob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stopwords = stopwords.words("english")


def preprocess(data_path):
    df_data = None
    for fname in glob(os.path.join(data_path, '*.json')):
        with open(fname, 'r') as f:
            data = json.load(f)
            temp_df = pd.DataFrame.from_dict(data)

            if df_data is None:
                df_data = temp_df
            else:
                df_data = df_data.append(temp_df, ignore_index=True)

    # df_data = 'title', ' author', ' time', ' description', ' body', ' section'
    #df_data['preprocessed_body'] = preprocess_body(df_data[' body'], df_data)

    res = []
    for raw_text in df_data[' body']:
        #TODO: lemmatization - remove words with frequency less than certain threshold.
        res.append(word_tokenize(raw_text))
    print('tokenization done')
    df_data['tokenized_body'] = res
    # save the output
    df_data.to_pickle('preprocess_result.pkl')
    return df_data


if __name__ == '__main__':
    preprocess('./data')
