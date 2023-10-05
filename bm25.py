# We also compare the results to lexical search (keyword search). Here, we use
# the BM25 algorithm which is implemented in the rank_bm25 package.

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np
import os
import torch


# We lower case our text and remove stop-words from indexing
def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc


def load_bm25(passages, bm25_file="bm25.pt", data_folder="data/"):
    path = data_folder + bm25_file
    if os.path.exists(path):
        print("Load model BM25 from ", path)
        return torch.load(path)
    else:
        print("Build model bm25")
        tokenized_corpus = []
        # each passage = [title, content]
        for passage in tqdm(passages):
            tokenized_corpus.append(bm25_tokenizer(passage[1]))

        bm25 = BM25Okapi(tokenized_corpus)
        torch.save(bm25, path)
        print("Save model BM25 at ", path)
        return bm25


def lexical_search(bm25, query, passages, top_k=10):
    ##### BM25 search (lexical search) #####
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -top_k)[-top_k:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

    results = []
    for hit in bm25_hits[0:top_k]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']]))
        results.append(passages[hit['corpus_id']])
    return results, bm25_hits