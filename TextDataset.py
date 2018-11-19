import torch
from torch.utils import data


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, labels_dir, num_tokens=1014):
        self.idx_to_image = {}
        self.idx_to_class = {}
        self.class_to_idx = {}
        self.label_name_to_id = {}
        padding_id = 68

        self.data = []
        self.labels = []

        with open(data_dir, 'r') as input_file:
            for line in input_file:
                sample = [padding_id] * num_tokens
                tokens = line.rstrip('\n').split()
                for i in range(0, min(num_tokens, len(tokens))):
                    sample[i] = int(tokens[i])
                self.data += [sample]

        with open(labels_dir, 'r') as input_file:
            for line in input_file:
                self.labels += [int(line.rstrip('\n'))]

        self.length = len(self.labels)

    def __getitem__(self, index):
        return torch.LongTensor(self.data[index]), self.labels[index]

    def __len__(self):
        return self.length