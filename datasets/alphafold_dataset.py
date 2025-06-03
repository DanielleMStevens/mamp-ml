import torch


category_to_index = {category: idx for idx, category in enumerate(["Immunogenic", "Non-Immunogenic", "Weakly Immunogenic"])}

class AlphaFoldDataset(torch.utils.data.Dataset):
    def __init__(self, df, name_to_x):

        df['Epitope-Seq'] = df['Epitope'] + "-" + df['Sequence']
        self.x = [name_to_x[epitope_seq] for epitope_seq in df['Epitope-Seq']]
        self.y = list(df['Known Outcome'].map(category_to_index))
        self.seqs = list(df['Sequence'])
        self.name = "AlphaFoldDataset"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx], 'seqs': self.seqs[idx]}
    
