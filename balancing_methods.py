from transformers import Trainer
from torch.utils.data import Sampler, DataLoader
import random
from collections import defaultdict

class LabelBalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.label_to_indices = defaultdict(list)
        
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        
        self.num_labels = len(set(labels))
        assert self.batch_size >= self.num_labels, (
            f"Batch size ({self.batch_size}) must be at least the number of unique labels ({self.num_labels})"
        )

    def __iter__(self):
        label_indices = {label: indices[:] for label, indices in self.label_to_indices.items()}
        for indices in label_indices.values():
            random.shuffle(indices)
        
        batches = []
        while all(label_indices.values()):
            batch = []
            for label in self.label_to_indices:
                if label_indices[label]:
                    batch.append(label_indices[label].pop())
            remaining = self.batch_size - len(batch)
            all_remaining = sum(label_indices.values(), [])
            random.shuffle(all_remaining)
            batch += all_remaining[:remaining]
            batches.append(batch)
        
        random.shuffle(batches)
        return iter(sum(batches, []))

    def __len__(self):
        return len(self.labels)


class BalancedTrainer(Trainer):
    def get_train_dataloader(self):
        sampler = LabelBalancedBatchSampler(
            labels=self.train_dataset['labels'],
            batch_size=self.args.per_device_train_batch_size
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
        )
