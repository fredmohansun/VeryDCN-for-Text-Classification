import numpy as np
import os
import nltk
import itertools
import io


def pre_process_amazon(data_dir):
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


def pre_process_dbpedia(data_dir):
    x = []
    y = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            label = int(line.split(',')[0])-1
            text = line.split(',')
            text = ','.join(text[1: len(text)])
            args = text.split('\",\"')
            title = args[0].lstrip('\"').lower()
            body = args[1].rstrip('\n').rstrip('\"').lower()
            text = nltk.word_tokenize(title + ' ' + body)
            x.append(text)
            y.append(label)
    return x, y


def pre_process_yelp(data_dir):
    x = []
    y = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            args = line.split('\",\"')
            label = int(args[0].lstrip('\"')) - 1
            review = args[1].rstrip('\n').rstrip('\"').lower()
            x.append(review)
            y.append(label)
            print(review)
            print(label)
    return x, y


def main():

    dataset = 'yelp'
    base_directory = '../project/'
    train_directories = {'dbpedia': base_directory + 'dbpedia_csv/train.csv', 'yelp': base_directory + 'yelp_review_full_csv/train.csv'}
    test_directories = {'dbpedia': base_directory + 'dbpedia_csv/test.csv', 'yelp': base_directory + 'yelp_review_full_csv/test.csv'}

    if not os.path.isdir(base_directory + 'preprocessed_data_lstm'):
        os.mkdir(base_directory + 'preprocessed_data_lstm')

    x_train, y_train = [], []
    x_test, y_test = [], []
    if dataset == 'dbpedia':
        x_train, y_train = pre_process_dbpedia(train_directories[dataset])
        x_test, y_test = pre_process_dbpedia(test_directories[dataset])
    elif dataset == 'amazon':
        x_train, y_train = pre_process_amazon(train_directories[dataset])
        x_test, y_test = pre_process_amazon(test_directories[dataset])
    elif dataset == 'yelp':
        x_train, y_train = pre_process_yelp(train_directories[dataset])
        x_test, y_test = pre_process_yelp(test_directories[dataset])
    else:
        print("dataset " + str(dataset) + " is not supported")
        exit(-1)

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
    np.save(base_directory + 'preprocessed_data_lstm/' + dataset + '_dictionary.npy', np.asarray(id_to_word))

    # save training data to single text file
    with io.open(base_directory + 'preprocessed_data_lstm/' + dataset + '_train.txt', 'w', encoding='utf-8') as f:
        for tokens in x_train_token_ids:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")

    # save test data to single text file
    with io.open(base_directory + 'preprocessed_data_lstm/' + dataset + '_test.txt', 'w', encoding='utf-8') as f:
        for tokens in x_test_token_ids:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")


if __name__ == '__main__':
    main()