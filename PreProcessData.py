import numpy as np
import os
from TextDataset import TextDataset


def pre_process_amazon(data_dir):
    x = []
    y = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            args = line.split('\",\"')
            label = int(args[0].lstrip('\"')) - 1
            title = args[1].lower()
            body = args[2].rstrip('\n').rstrip('\"').lower()

            x.append(title + ' ' + body)
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
            x.append(title + ' ' + body)
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
    return x, y


def write_data_to_file(output_dir, data):
    with open(output_dir, 'w', encoding='utf-8') as f:
        for tokens in data:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")


def write_labels_to_file(output_dir, data):
    with open(output_dir, 'w', encoding='utf-8') as f:
        for label in data:
            f.write("%i " % label)
            f.write("\n")


def main():

    dataset = 'yelp'
    base_directory = '../project/'
    train_directories = {'dbpedia': base_directory + 'dbpedia_csv/train.csv', 'yelp': base_directory + 'yelp_review_full_csv/train.csv'}
    test_directories = {'dbpedia': base_directory + 'dbpedia_csv/test.csv', 'yelp': base_directory + 'yelp_review_full_csv/test.csv'}

    if not os.path.isdir(base_directory + 'preprocessed_data'):
        os.mkdir(base_directory + 'preprocessed_data')

    id_2_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                   'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                   'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6',
                   '7', '8', '9', '-', ',', ';', '.', '!', '?', ':', '’', '\"',
                   '/', '|', '_', '#', '$', '%', 'ˆ', '&', '*', '˜', '‘', '+',
                   '=', '<', '>', '(', ')', '[', ']', '{', '}', ' ']

    char_2_id = {}
    for i in range(len(id_2_char)):
        char_2_id[id_2_char[i]] = i
    unknown_id = len(id_2_char)

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

    x_train_token_ids = [[char_2_id.get(token, unknown_id) for token in x] for x in x_train]
    x_test_token_ids = [[char_2_id.get(token, unknown_id) for token in x] for x in x_test]

    np.save(base_directory + 'preprocessed_data/' + dataset + '_dictionary.npy', np.asarray(id_2_char))
    write_data_to_file(base_directory + 'preprocessed_data/' + dataset + '_train.txt', x_train_token_ids)
    write_labels_to_file(base_directory + 'preprocessed_data/' + dataset + '_train_labels.txt', y_train)
    write_data_to_file(base_directory + 'preprocessed_data/' + dataset + '_test.txt', x_test_token_ids)
    write_labels_to_file(base_directory + 'preprocessed_data/' + dataset + '_test_labels.txt', y_test)

if __name__ == '__main__':
    main()