from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
twenty = fetch_20newsgroups()

tfidf = TfidfVectorizer().fit_transform(twenty.data)
from sklearn.metrics.pairwise import linear_kernel
cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
related_docs_indices = cosine_similarities.argsort()[:-5:-1]
for i in related_docs_indices:
    print(twenty.data[i])
    print("================================")
