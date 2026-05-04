import torch
from torch.utils.data import Dataset
import random

class SimpleDataset(Dataset):
    def __init__(self, short_data, news2idx):
        self.users = list(short_data.keys())
        self.short_data = short_data
        self.max_news = max(news2idx.values())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.short_data[user]

        # ensure valid sequence
        if len(seq) < 2:
            return self.__getitem__((idx + 1) % len(self.users))

        input_seq = seq[:-1]
        target = seq[-1]

        # extract fields
        news_seq = [x[0] for x in input_seq]
        cat_seq  = [x[2] for x in input_seq]
        delta_t  = [x[3] for x in input_seq]
        delta_t = [x[3] / 3600.0 for x in input_seq]   # seconds → hours

        pos_item = target[0]

        user_clicked = set(x[0] for x in seq)

        neg_item = random.randint(1, self.max_news)
        while neg_item in user_clicked:
            neg_item = random.randint(1, self.max_news)

        return (
            torch.tensor(news_seq, dtype=torch.long),
            torch.tensor(cat_seq, dtype=torch.long),
            torch.tensor(delta_t, dtype=torch.float32),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long)
        )
    

def collate_fn(batch):
    news_seq, cat_seq, delta_t, pos, neg = zip(*batch)

    max_len = max(len(x) for x in news_seq)

    def pad(seq, value=0):
        return torch.nn.functional.pad(seq, (0, max_len - len(seq)), value=value)

    news_seq = torch.stack([pad(x) for x in news_seq])
    cat_seq  = torch.stack([pad(x) for x in cat_seq])
    delta_t  = torch.stack([pad(x) for x in delta_t])

    pos = torch.stack(pos)
    neg = torch.stack(neg)

    return news_seq, cat_seq, delta_t, pos, neg