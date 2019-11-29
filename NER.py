from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import ne_chunk
import pandas as pd
from nltk.tree import Tree

# NEED TO RUN PREPROCESS.PY before RUNNING NER.py!
'''
EXAMPLE_TEXT = 'Michelle Obama takes one last walk through the White House with the family dogs as workers continue to ready her new DC home.'
tokens = word_tokenize(EXAMPLE_TEXT)    # tokenizing
# ['Michelle', 'Obama', 'takes', 'one', 'last', 'walk', 'through', 'the',
# 'White', 'House', 'with', 'the', 'family', 'dogs', 'as', 'workers',
# 'continue', 'to', 'ready', 'her', 'new', 'DC', 'home', '.']
pos = pos_tag(tokens)                   # pos tagging
#  # [('Michelle', 'NNP'), ('Obama', 'NNP'), ('takes', 'VBZ'), ('one', 'CD'),
#  ('last', 'JJ'), ('walk', 'NN'), ('through', 'IN'), ('the', 'DT'), ('White',
#  'NNP'), ('House', 'NNP'), ('with', 'IN'), ('the', 'DT'), ('family', 'NN'),
#  ('dogs', 'NNS'), ('as', 'IN'), ('workers', 'NNS'), ('continue', 'VBP'),
#  ('to', 'TO'), ('ready', 'VB'), ('her', 'PRP$'), ('new', 'JJ'), ('DC', 'NNP'),
#  ('home', 'NN'), ('.', '.')]
namedEnt = ne_chunk(pos, binary=True)   # NER
print(namedEnt)
'''
def extract_ner(ner_result):
    res = {'single_word': [], 'multi_word': []}
    # {'single word': ['Bank', ..], 'multi word': [['South', 'Korea'], ...[]]}
    # ner_result type should be tree.
    assert type(ner_result) == Tree
    for entity in ner_result:
        if type(entity) == Tree:
            if len(entity) > 1:
                multi_word = []
                for word_tuple in entity:
                    multi_word.append(word_tuple[0])
                res['multi_word'].append(multi_word)
            else:
                res['single_word'].append(entity[0][0])
    return res
df = pd.read_pickle('preprocess_result.pkl')
nered_result = []
for preprocessed_text in df['tokenized_body']:
    ner_result = ne_chunk(pos_tag(preprocessed_text), binary=True)
    ner_set = extract_ner(ner_result)
    nered_result.append(ner_set)
df['ner'] = nered_result
df.to_pickle('ner_result.pkl')
