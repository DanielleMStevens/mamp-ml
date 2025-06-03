import torch


category_to_index = {category: idx for idx, category in enumerate(["Immunogenic", "Non-Immunogenic", "Weakly Immunogenic"])}

class PeptideSeqWithReceptorDataset(torch.utils.data.Dataset):
    def __init__(self, df):

        self.peptide_x = df['Sequence']
        self.receptor_x = df['Receptor Sequence']
        self.y = df['Known Outcome'].map(category_to_index)
        self.name = "PeptideSeqWithReceptorDataset"

    def __len__(self):
        return len(self.peptide_x)

    def __getitem__(self, idx):
        return { 'peptide_x': self.peptide_x[idx], 'receptor_x': self.receptor_x[idx], 'y': self.y[idx]}
