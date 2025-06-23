import torch
from torch.utils.data import IterableDataset
from collections import defaultdict
from torch.utils.data import DataLoader
from config import DATASET_NAME, DS_DIR_BASE, MAX_CAPTION_LEN, MODELS_DIR_BASE, SIGLIP_HF_NAME
from datasets import load_dataset


def custom_collate(batch):
    captions = [item['caption'] for item in batch]
    labels = [item['label'] for item in batch]
    ae_latents = [item["ae_latent"] for item in batch]
    ae_latent_shapes = [item["ae_latent_shape"] for item in batch]
    dinov2_pca_features = [item["dinov2_pca_features"] for item in batch]

    return {
        'caption': captions,
        'label': labels,
        'ae_latent': ae_latents,
        'ae_latent_shape': ae_latent_shapes,
        'dinov2_pca_features': dinov2_pca_features
    }


class ShapeBatchingDataset(IterableDataset):
    def __init__(self, hf_dataset, batch_size, device, num_workers, shuffle=True, seed=42, buffer_multiplier=20):
        self.dataset = hf_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_multiplier = buffer_multiplier

        self.device = device
        self.num_workers = num_workers

    def __iter__(self):
        while True:
            self.dataloader = DataLoader(self.dataset, self.batch_size * 2, prefetch_factor=10, num_workers=self.num_workers, collate_fn=custom_collate, shuffle=self.shuffle)
            
            shape_batches = defaultdict(lambda: {'caption': [], 'ae_latent': [], 'dinov2_pca_features': [], 'label': [], 'label_txt': []})
            for batch in self.dataloader:
                caption = batch['caption']
                ae_latent = batch['ae_latent']
                ae_latent_shape = batch['ae_latent_shape']
                dinov2_pca_features = batch['dinov2_pca_features']
                label = batch['label']
                label_txt = self.dataset.features["label"].int2str(label)

                for i in range(len(caption)):
                    shape_key = tuple(ae_latent_shape[i])
                    shape_batches[shape_key]['caption'].append(caption[i])
                    shape_batches[shape_key]['ae_latent'].append(ae_latent[i])
                    shape_batches[shape_key]['dinov2_pca_features'].append(dinov2_pca_features[i])
                    shape_batches[shape_key]['label'].append(label[i])
                    shape_batches[shape_key]['label_txt'].append(label_txt[i])

                    # If enough samples are accumulated for this shape, yield a batch
                    if len(shape_batches[shape_key]['caption']) == self.batch_size:
                        batch = self.prepare_batch(shape_batches[shape_key], shape_key)
                        yield batch
                        shape_batches[shape_key]['caption'] = []
                        shape_batches[shape_key]['ae_latent'] = []
                        shape_batches[shape_key]['dinov2_pca_features'] = []
                        shape_batches[shape_key]['label'] = []
                        shape_batches[shape_key]['label_txt'] = []
    
    def prepare_batch(self, samples, latent_shape):
        # Convert lists of samples into tensors
        ae_latent = torch.tensor(samples["ae_latent"])
        dinov2_pca_features = torch.tensor(samples["dinov2_pca_features"])
        label = torch.tensor(samples["label"])
        label_txt = samples["label_txt"]
        
        batch = {
            'caption': samples["caption"],
            'ae_latent': ae_latent,
            'ae_latent_shape': latent_shape,
            'dinov2_pca_features': dinov2_pca_features,
            'label': label,
            'label_txt': label_txt
        }
        return batch

def get_dataset(bs, seed, device, num_workers=16, split="train"):
    ds = load_dataset(DATASET_NAME, cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}", num_proc=num_workers, split=split)

    ds = ShapeBatchingDataset(ds, bs, None, None, device, num_workers, shuffle=True, seed=seed)
    return ds