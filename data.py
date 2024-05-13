import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __getitem__(self, index) -> str:
        text = self.dataset[index]['text'] + '\0'
        return torch.frombuffer(text.encode(), dtype=torch.uint8)

    def __len__(self) -> int:
        return len(self.dataset)


class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, index):
        return self.transform(self.dataset[index])

    def __len__(self) -> int:
        return len(self.dataset)


def pad_collate_fn(batch):
    """
    batch is a list of tuple of torch arrays
    """
    return list(torch.nn.utils.rnn.pad_sequence(xs, True, 0).to(torch.long) for xs in zip(*batch))
