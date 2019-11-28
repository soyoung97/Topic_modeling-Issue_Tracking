from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import ne_chunk
import pandas as pd
from nltk.tree import Tree

def extract_ner(ner_result):
    res = []
    assert type(ner_result) == Tree
    for entity in ner_result:
        if type(entity) == Tree:
            if len(entity) > 1:
                res.append(entity)
    return res
df = pd.read_pickle('preprocess_result.pkl')
nered_result = []
for preprocessed_text in df['tokenized_body']:
    ner_result = ne_chunk(pos_tag(preprocessed_text), binary=False)
    ner_set = extract_ner(ner_result)
    nered_result.append(ner_set)
df['ner_category'] = nered_result
df.to_pickle("ner_collection.pkl")
