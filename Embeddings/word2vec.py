import gensim


# Takes a class label and Word2Vec model and returns the word embedding for that label
# If the label contains multiple words, their embeddings are averaged (Xie, 2021)
def word2vec(model, label, double_first=False):
    # Get the vocab so we can check if any words are not in the model
    vocab = model.index_to_key

    # Split the label to get individual words
    words = label.split("_")
    num_words = 0

    # If the first word should be doubled, do so
    # This may be to emphasise the true class label in the case of synonyms
    if double_first:
        words.append(words[0])

    # Loop over each word in the label
    vector = []
    for part in words:
        # If it's in the vocab, add it to the vector
        if part in vocab:
            num_words += 1
            v = model[part]
            # Either set the vector or add the two together
            if vector == []:
                vector = v
            else:
                vector = [(vector[i] + v[i]) for i in range(len(v))]
        # If it's not, we shouldn't really continue, so raise an error
        else:
            print("{} is not in the Word2Vec model vocab.".format(part))

    # Return the word embedding
    return [v / num_words for v in vector]
