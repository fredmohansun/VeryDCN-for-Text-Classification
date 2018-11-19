import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from TextDataset import TextDataset


class BOWmodel(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(BOWmodel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, no_of_hidden_units)
        self.fc_hidden = nn.Linear(no_of_hidden_units, no_of_hidden_units)
        self.bn_hidden = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc_output = nn.Linear(no_of_hidden_units, 5)

    def forward(self, x):
        bow_embedding = []
        for i in range(len(x)):
            lookup_tensor = Variable(torch.LongTensor(x[i]))
            embed = self.embedding(lookup_tensor)
            embed = embed.mean(dim=0)
            bow_embedding.append(embed)
        bow_embedding = torch.stack(bow_embedding)

        h = self.dropout(F.relu(self.bn_hidden(self.fc_hidden(bow_embedding))))
        h = self.fc_output(h)
        return h


def main():
    vocab_size = 69
    num_hidden = 500
    LR = 0.001
    batch_size = 5
    no_of_epochs = 10

    model = BOWmodel(vocab_size, num_hidden)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()

    print("loading")
    train_dataset = TextDataset("preprocessed_data/amazon_train.txt", "preprocessed_data/amazon_train_labels.txt")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    test_dataset = TextDataset("preprocessed_data/amazon_test.txt", "preprocessed_data/amazon_test_labels.txt")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    for epoch in range(no_of_epochs):
        correct_training = 0.0
        total_training = 0.0
        model.train()

        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'step' in state and state['step'] >= 1024:
                    state['step'] = 1000

        for data, target in train_loader:
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            correct_training += (predicted == target).sum().item()
            total_training += target.size(0)

        print(str(epoch) + ": " + str(100 * correct_training / total_training))

if __name__ == '__main__':
    main()