import os.path
import numpy as np


def init_embeddings_map(fname):
    with open(os.path.join("data", "glove.6B", fname)) as glove:
        return {l[0]: np.asarray(l[1:], dtype="float32") for l in
                [line.split() for line in glove]}


def get_embed_and_pad_func(i_seq_len, u_seq_len, pad_value, embedding_map):
    def embed(row):
        sentence = row["userReviews"].split()[:u_seq_len]
        reviews = list(map(lambda word: embedding_map.get(word)
            if word in embedding_map else pad_value, sentence))
        row["userReviews"] = reviews +\
                [pad_value] * (u_seq_len - len(reviews))
        sentence = row["movieReviews"].split()[:i_seq_len]
        reviews = list(map(lambda word: embedding_map.get(word)
            if word in embedding_map else pad_value, sentence))
        row["movieReviews"] = reviews +\
                [pad_value] * (i_seq_len - len(reviews))
        return row
    return embed
