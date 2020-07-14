from torch.utils.data import Dataset
import pandas as pd


class TextCSVDataset(Dataset):
    def __init__(self, path, text_transforms=None, label_transforms=None):
        super(TextCSVDataset, self).__init__()
        self.data = pd.read_csv(path)
        self.text_tfs = text_transforms
        self.label_tfs = label_transforms

    def __getitem__(self, index):
        text = self.data.iloc[index, 0]
        if self.text_tfs:
            text = self.text_tfs(text)
        label = self.data.iloc[index, 1]
        if self.label_tfs:
            label = self.label_tfs(label)
        return text, label

    def __len__(self):
        return self.data.shape[0]
