'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['LeNet', 'mnist_lenet']
model_urls = {
    'lenet': 'https://github.com/rhhc/zxd_releases/releases/download/Re/mnist_lenet-eff-99.23-3eb8b274.pth'
}


def mnist_lenet(pretrained=False, **kwargs):
    model = LeNet()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['lenet'], map_location='cpu'))
    return model


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        width = 1.0
        self.conv1 = nn.Conv2d(1, int(width*6), 5)
        self.conv2 = nn.Conv2d(int(width*6), int(width*16), 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.fc1 = nn.Conv2d(int(width*16), int(width*120), 1)
        self.fc2 = nn.Conv2d(int(width*120), int(width*84), 1)
        # self.fc3 = nn.Conv2d(84, 10, 1)
        self.fc3 = nn.Linear(int(width*84*5*5), 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = out.reshape(out.size(0), -1)
        out = self.fc3(out)
        return out
