import pandas as pd

df = pd.read_pickle("ner_collection.pkl")
print(df)
print(df["ner_category"][14])

from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor

extractor = MasterExtractor()
doc = Document.from_text(df[" body"][14], df[" time"][14])
# or: doc = Document(title, lead, text, date_publish)
doc = extractor.parse(doc)

top_when_answer = doc.get_top_answer('when').get_parts_as_text()
print(top_when_answer)
top_where_answer = doc.get_top_answer('where').get_parts_as_text()
print(top_where_answer)
