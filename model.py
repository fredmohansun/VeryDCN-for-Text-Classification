import torch
import torch.nn as nn


class KMaxPool(nn.Module):
    def __init__(self, k=0):
        super(KMaxPool, self).__init__()
        self.k = k
    def forward(self, x):
        if(self.k==0):
            self.k = x.size()[2]/2
        return x.topk(k, sorted=False)[0]

# Conv block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BasicBlock, self).__init__()

        # architecture
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.sequential(x)
        return out

# Residual block
class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, shortcut=True):
        super(BasicResBlock, self).__init__()

        if downsample==1:
            self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            first_stride = 1
        elif downsample == 2:
            first_stride = 2
        elif downsample==3:
            self.pool = KMaxPool()
        else:
            first_stride = 1
        self.downsample = downsample

        self.convblock = BasicBlock(in_channels, out_channels, stride=first_stride)

        if shortcut and downsample:
                self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        out = self.convblock(x)
        if downsample:
            out = self.pool(out)
        if shortcut and downsample:
            out = out + self.shortcut(x)
        elif shortcut:
            out = out + x
        else:
            out = out
        return out


# CNN Model
class VDCNN(nn.Module):
    def __init__(self, vocabsize, embedsize, depth, downsample, shortcut, K, num_classes=5):
        super(VDCNN, self).__init__()

        layers = []
        # architecture
        self.embed = nn.Embedding(vocabsize, embedsize, 0)
        layers.append(nn.Conv1d(embedsize, 64, kernel_size=3, padding=1))
        if depth == 9:
            nblock64, nblock128, nblock256, nblock512 = 2, 2, 2, 2
        elif depth == 17:
            nblock64, nblock128, nblock256, nblock512 = 4, 4, 4, 4
        elif depth == 29:
            nblock64, nblock128, nblock256, nblock512 = 10, 10, 4, 4
        elif depth == 49:
            nblock64, nblock128, nblock256, nblock512 = 16, 16, 10, 6

        for i in range(nblock64):
            layers.append(BasicResBlock(64, 64, shortcut=shortcut))

        layers.append(BasicResBlock(64, 128, downsample=downsample, shortcut=shortcut))
        for i in range(nblock128-1):
            layers.append(BasicResBlock(128, 128, shortcut=shortcut))

        layers.append(BasicResBlock(128, 256, downsample=downsample, shortcut=shortcut))
        for i in range(nblock256-1):
            layers.append(BasicResBlock(256, 256, shortcut=shortcut))

        layers.append(BasicResBlock(256, 512, downsample=downsample, shortcut=shortcut))
        for i in range(nblock512-1):
            layers.append(BasicResBlock(512, 512, shortcut=shortcut))

        self.layers = nn.Sequential(*layers)
        self.kmax_pooling = KMaxPool(k=K)
        self.fc = nn.Sequential( # fully connected layers
            nn.Linear(512 * K, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )


    def forward(self, x):
        out = self.embed(x)
        out = torch.transpose(out,1,2)
        out = self.layers(out)
        out = self.kmax_pooling(out)
        #print out.shape
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
