import torch.nn as nn

class ClassifierVIT(nn.Module):
    def __init__(self, num_features=768, num_classes=6):
        super(ClassifierVIT, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.fc(x)
