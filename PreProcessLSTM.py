import numpy as np
import os
import nltk
import itertools
import io


def pre_process(data_dir):
    x = []
    y = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            args = line.split('\",\"')
            label = int(args[0].lstrip('\"')) - 1
            title = args[1].lower()
            body = args[2].rstrip('\n').rstrip('\"').lower()
            text = nltk.word_tokenize(title + ' ' + body)
            x.append(text)
            y.append(label)

    return x, y


def main():
    train_directory = '../project/amazon_review_full_csv/train.csv'
    test_directory = '../project/amazon_review_full_csv/test.csv'

    if not os.path.isdir('preprocessed_data'):
        os.mkdir('preprocessed_data')

    x_train, y_train = pre_process(train_directory)
    x_test, y_test = pre_process(test_directory)

    # word_to_id and id_to_word. associate an id to every unique token in the training data
    all_tokens = itertools.chain.from_iterable(x_train)
    word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

    all_tokens = itertools.chain.from_iterable(x_train)
    id_to_word = [token for idx, token in enumerate(set(all_tokens))]
    id_to_word = np.asarray(id_to_word)

    x_train_token_ids = [[word_to_id[token] for token in x] for x in x_train]
    count = np.zeros(id_to_word.shape)
    for x in x_train_token_ids:
        for token in x:
            count[token] += 1
    indices = np.argsort(-count)
    id_to_word = id_to_word[indices]

    word_to_id = {token: idx for idx, token in enumerate(id_to_word)}
    x_train_token_ids = [[word_to_id.get(token, -1) + 1 for token in x] for x in x_train]
    x_test_token_ids = [[word_to_id.get(token, -1) + 1 for token in x] for x in x_test]

    # save dictionary
    np.save('preprocessed_data/amazon_dictionary.npy', np.asarray(id_to_word))

    # save training data to single text file
    with io.open('preprocessed_data/amazon_train.txt', 'w', encoding='utf-8') as f:
        for tokens in x_train_token_ids:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")

    # save test data to single text file
    with io.open('preprocessed_data/amazon_test.txt', 'w', encoding='utf-8') as f:
        for tokens in x_test_token_ids:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")


if __name__ == '__main__':
    main()