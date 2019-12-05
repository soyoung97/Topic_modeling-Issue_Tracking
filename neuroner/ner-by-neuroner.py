from nltk.tag import pos_tag
from nltk import ne_chunk
import pandas as pd
from nltk.tree import Tree
from neuroner import neuromodel

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
def ner_by_nltk():
    df = pd.read_pickle('../pickle/preprocess_result.pkl')
    nered_result = []
    for preprocessed_text in df['tokenized_body']:
        ner_result = ne_chunk(pos_tag(preprocessed_text), binary=True)
        ner_set = extract_ner(ner_result)
        nered_result.append(ner_set)
    df['ner'] = nered_result
    df.to_pickle('../pickle/ner_result.pkl')

def tokenize(word_list):
    res = [None] * len(word_list)
    for i, body in enumerate(word_list):
        res[i] = word_tokenize(body)
    return res

def extract_by_neuroner(file_path):
    nn = neuromodel.NeuroNER(train_model=False, use_pretrained_model=True)
    df = pd.read_pickle(file_path)
    neuroner_infos = []
    txt_without_ners = []
    for preprocessed_text in df[' body']:
        try:
            ner_list = nn.predict(preprocessed_text)
        except FileNotFoundError:
            res = extract_ner(ne_chunk(pos_tag(preprocessed_text), binary=True))
            word_list = res['single_word']
            multi_word = res['multi_word']
            for m_w in multi_word:
                w = " ".join(m_w)
                word_list.append(w)
            neuroner_infos.append(word_list)
            tt = preprocessed_text
            for ner in word_list:
                tt = tt.replace(ner, '')
            txt_without_ners.append(tt)
            continue
        neuroner_result = [None] * len(ner_list)
        txt = ''
        oldend = 0
        for i, ner_info in enumerate(ner_list):
            start = ner_info['start']
            txt += preprocessed_text[oldend:start]
            oldend = ner_info['end']
            neuroner_result[i] = ner_info['text']
        txt += preprocessed_text[oldend:]
        neuroner_infos.append(neuroner_result)
        txt_without_ners.append(txt)
        # make txt with NER text removed, and return list of ner tokens.
    df['neuroner_list'] = neuroner_infos
    df['neuroner_body'] = txt_without_ners
    df['neuroner_tokenized'] = tokenize(txt_without_ners)
    df.to_pickle('../pickle/neuroner_result.pkl')

if __name__ == '__main__':
    file_path = '../pickle/preprocess_result.pkl'
    extract_by_neuroner(file_path)
