import itertools
import torch
from torch.utils.data import IterableDataset, get_worker_info


class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length, skip_batches=0):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.skip_batches = skip_batches

    def __iter__(self):
        # skip trained batches when resume training
        worker_info = get_worker_info()
        if worker_info is None:
            iter_data = iter(self.data)
            worker_skip_batches = self.skip_batches
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)
            worker_skip_batches = self.skip_batches // num_workers + (1 if worker_id < self.skip_batches % num_workers else 0)

        batch = []
        batches_yielded = 0

        for example in iter_data:
            if len(batch) < self.batch_size:
                batch.append(example)

            if len(batch) == self.batch_size:
                # only tokenize non-skipping batch
                if batches_yielded >= worker_skip_batches:
                    tokenized_batch = [
                        self.tokenizer(
                            ex["text"],
                            max_length=self.max_length,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt",
                        ) for ex in batch
                    ]
                    yield self._format_batch(tokenized_batch)
                
                batch = []
                batches_yielded += 1

        # yield remaining data, if any
        if batch and batches_yielded >= worker_skip_batches:
            tokenized_batch = [
                self.tokenizer(
                    ex["text"],
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ) for ex in batch
            ]
            yield self._format_batch(tokenized_batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask}