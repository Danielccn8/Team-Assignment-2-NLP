from gensim.models import FastText
import numpy as np

file_path = "../src/glove.6B.200d.txt"

def load_glove_model(glove_file_path):
    print("Loading GloVe Model...")
    glove_model = {}
    with open(glove_file_path, 'r', encoding="utf8") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_model[word] = embedding
    print(f"Done. {len(glove_model)} words loaded!")
    return glove_model

def choose_embedding_model(dataset, vector_size, window, sg, epochs, batch_words, embed_model):
    if embed_model == "FastText":
        return FastText(dataset, # Both premise and Hypothesis 
                           vector_size=vector_size,
                           window=window,
                           sg=sg,
                           epochs=epochs,
                           batch_words=batch_words
                           )
    elif embed_model == "GloVe":
        if file_path is None:
            raise ValueError("A file_path is required to load a GloVe model.")
        return load_glove_model(file_path)
    else:
        raise ValueError(f"Unknown embedding model: {embed_model}. Choose 'FastText' or 'GloVe'.")
