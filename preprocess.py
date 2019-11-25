import json
import pandas as pd
from nltk.corpus import stopwords
import pickle
from nltk.tokenize import word_tokenize

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
        res.append(word_tokenize(raw_text))
    print("tokenization done")
    df_data['tokenized_body'] = res
    # save the output
    df_data.to_pickle("preprocess_result.pkl")
    return df_data



preprocess()
