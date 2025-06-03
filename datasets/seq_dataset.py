import torch


category_to_index = {category: idx for idx, category in enumerate(["Immunogenic", "Non-Immunogenic", "Weakly Immunogenic"])}

class PeptideSeqDataset(torch.utils.data.Dataset):
    def __init__(self, df):

        self.x = df['Sequence']
        self.y = df['Known Outcome'].map(category_to_index)
        self.name = "PeptideSeqDataset"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx]}
