import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

from DictList import DictList

def get_folds(dataset, k):
    _, n_samples = np.unique([cls for _, cls in dataset], return_counts=True)
    fold_sizes = n_samples // k
    fold_remainders = np.mod(n_samples, k)
    class_offsets = np.cumsum(n_samples) - n_samples[0]
    folds = []
    for i in range(k):
        fold = []
        for fsize, offset, rem in zip(fold_sizes, class_offsets, fold_remainders):
            if i < rem:
                fsize = fsize + 1
            fold.extend(
                np.arange(i*fsize + offset, (i+1)*fsize + offset)
            )
        folds.append(np.array(fold))
            
    return folds

class LOOCV:
    def __init__(self, dataset, train_dataset=None, k=10, **loader_kwargs):
        self.k = k

        # Separate train and eval datsets for different transforms
        self.eval_dataset = dataset
        if train_dataset is None:
            self.train_dataset = dataset
        else:
            self.train_dataset = train_dataset
        self.folds = get_folds(dataset, k)

        self.loader_kwargs = loader_kwargs

    def getLoaders(self, fold_n, **kwargs):
        train_indices = np.hstack(
            self.folds[:fold_n] + self.folds[fold_n + 1:]
        )
        test_indices = self.folds[fold_n]
        
        train_loader = DataLoader(
            self.train_dataset,
            sampler=SubsetRandomSampler(train_indices),
            **kwargs
        )
        val_loader = DataLoader(
            self.eval_dataset,
            sampler=SubsetRandomSampler(test_indices),
            **kwargs
        )
        return train_loader, val_loader

    def apply(self, func, **kwargs):
        data = DictList()
        print(f"Starting LOOCV with {self.k} folds...")
        for i in range(self.k):
            train_loader, val_loader = self.getLoaders(
                i, **self.loader_kwargs
            )
            data.merge(
                func(train_loader, val_loader, **kwargs),
                mode = "append"
            )
            print(f"[fold {i+1}/{self.k} finished]")
        return data
            
