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
    train_directory = 'amazon_review_full_csv/train.csv'
    test_directory = 'amazon_review_full_csv/test.csv'

    if not os.path.isdir('preprocessed_lstm_data'):
        os.mkdir('preprocessed_lstm_data')

    x_train, y_train = pre_process(train_directory)
    x_test, y_test = pre_process(test_directory)


    glove_filename = '/projects/training/bauh/NLP/glove.840B.300d.txt'
    with io.open(glove_filename,'r',encoding='utf-8') as f:
        lines = f.readlines()

    glove_dictionary = []
    glove_embeddings = []
    count = 0
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        glove_dictionary.append(line[0])
        embedding = np.asarray(line[1:],dtype=np.float)
        glove_embeddings.append(embedding)
        count+=1
        if(count>=100000):
            break

    glove_dictionary = np.asarray(glove_dictionary)
    glove_embeddings = np.asarray(glove_embeddings)
    # added a vector of zeros for the unknown tokens
    glove_embeddings = np.concatenate((np.zeros((1,300)),glove_embeddings))

    # word_to_id and id_to_word. associate an id to every unique token in the training data
    word_to_id = {token: idx for idx, token in enumerate(glove_dictionary)}

    x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
    x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]
    

    np.save('preprocessed_lstm_data/glove_dictionary.npy',glove_dictionary)
    np.save('preprocessed_lstm_data/glove_embeddings.npy',glove_embeddings)

    with io.open('preprocessed_data/imdb_train_glove.txt','w',encoding='utf-8') as f:
        for tokens in x_train_token_ids:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")

    with io.open('preprocessed_lstm_data/imdb_test_glove.txt','w',encoding='utf-8') as f:
        for tokens in x_test_token_ids:
            for token in tokens:
                f.write("%i " % token)
            f.write("\n")


if __name__ == '__main__':
    main()
\