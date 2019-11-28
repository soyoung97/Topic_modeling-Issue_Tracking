import pandas as pd

df = pd.read_pickle("ner_collection.pkl")
print(df["ner_category"])
