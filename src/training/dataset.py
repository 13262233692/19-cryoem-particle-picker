import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any
import random
from scipy import ndimage
from src.utils.logging import get_logger
from src.io.mrc_parser import MRCStreamParser
from src.preprocessing.pipeline import PreprocessingPipeline

logger = get_logger("training.dataset")

class CryoEMDataset(Dataset):
    def __init__(self,
                 image_paths: List[str],
                 label_paths: Optional[List[str]] = None,
                 preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
                 augment: bool = True,
                 patch_size: int = 512,
                 max_patches_per_image: int = 10,
                 noise_std: float = 0.01,
                 use_patching: bool = False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.augment = augment
        self.patch_size = patch_size
        self.max_patches_per_image = max_patches_per_image
        self.noise_std = noise_std
        self.use_patching = use_patching
        self.preprocessing = preprocessing_pipeline
        self._cache: Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        self._patch_indices = self._generate_patch_indices()
        logger.info(f"CryoEMDataset initialized: {len(image_paths)} images, "
                    f"{len(self._patch_indices)} patches, augment={augment}")

    def _generate_patch_indices(self) -> List[Tuple[int, int]]:
        indices = []
        for img_idx in range(len(self.image_paths)):
            for patch_idx in range(self.max_patches_per_image):
                indices.append((img_idx, patch_idx))
        return indices

    def _load_data(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if idx in self._cache:
            return self._cache[idx]
        img_path = self.image_paths[idx]
        with MRCStreamParser(img_path, zero_copy=True) as parser:
            image = parser.get_image(0)
        if self.preprocessing is not None:
            image = self.preprocessing.process(image, use_patching=self.use_patching).image
        label = None
        if self.label_paths and idx < len(self.label_paths):
            label_path = self.label_paths[idx]
            if os.path.exists(label_path):
                if label_path.endswith('.npy'):
                    label = np.load(label_path)
                elif label_path.endswith('.mrc'):
                    with MRCStreamParser(label_path, zero_copy=True) as parser:
                        label = parser.get_image(0)
                elif label_path.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    try:
                        import cv2
                        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    except ImportError:
                        from PIL import Image
                        label = np.array(Image.open(label_path).convert('L'))
                else:
                    try:
                        label = np.load(label_path, allow_pickle=True)
                    except Exception:
                        try:
                            import cv2
                            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                        except Exception:
                            label = None
                if label.ndim > 2:
                    label = label.squeeze()
        self._cache[idx] = (image, label)
        return self._cache[idx]

    def _random_crop(self, image: np.ndarray, label: Optional[np.ndarray],
                     patch_idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        h, w = image.shape
        ps = self.patch_size
        if h <= ps and w <= ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            image_pad = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
            if label is not None:
                label_pad = np.pad(label, ((0, pad_h), (0, pad_w)), mode='reflect')
                return image_pad, label_pad
            return image_pad, None
        if label is not None and np.any(label > 0):
            foreground_coords = np.argwhere(label > 0.5)
            if len(foreground_coords) > 0:
                if random.random() < 0.7:
                    center = foreground_coords[random.randint(0, len(foreground_coords) - 1)]
                    y = max(0, min(center[0] - ps // 2, h - ps))
                    x = max(0, min(center[1] - ps // 2, w - ps))
                else:
                    y = random.randint(0, max(0, h - ps))
                    x = random.randint(0, max(0, w - ps))
            else:
                y = random.randint(0, max(0, h - ps))
                x = random.randint(0, max(0, w - ps))
        else:
            y = random.randint(0, max(0, h - ps))
            x = random.randint(0, max(0, w - ps))
        image_patch = image[y:y+ps, x:x+ps]
        label_patch = label[y:y+ps, x:x+ps] if label is not None else None
        return image_patch, label_patch

    def _augment(self, image: np.ndarray, label: Optional[np.ndarray]
                 ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if random.random() < 0.5:
            image = np.fliplr(image)
            if label is not None:
                label = np.fliplr(label)
        if random.random() < 0.5:
            image = np.flipud(image)
            if label is not None:
                label = np.flipud(label)
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k=k)
            if label is not None:
                label = np.rot90(label, k=k)
        if random.random() < 0.3:
            noise = np.random.normal(0, self.noise_std, image.shape).astype(np.float32)
            image = image + noise
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            image = ndimage.rotate(image, angle, reshape=False, order=1, mode='reflect')
            if label is not None:
                label = ndimage.rotate(label, angle, reshape=False, order=0, mode='reflect')
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.2)
            image = np.power(np.clip(image, 0, 1), gamma)
        return image, label

    def __len__(self) -> int:
        return len(self._patch_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_idx, patch_idx = self._patch_indices[idx]
        image, label = self._load_data(img_idx)
        if self.patch_size is not None:
            image, label = self._random_crop(image, label, patch_idx)
        if self.augment:
            image, label = self._augment(image, label)
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        if label is not None:
            label = (label > 0.5).astype(np.int64)
            label_tensor = torch.from_numpy(label).long()
        else:
            label_tensor = torch.zeros(image.shape, dtype=torch.long)
        return {
            "image": image_tensor,
            "label": label_tensor,
            "mask": label_tensor,
            "index": torch.tensor(img_idx, dtype=torch.long)
        }

class MRCDataModule:
    def __init__(self,
                 data_dir: str,
                 image_dir: str = "images",
                 label_dir: str = "labels",
                 val_split: float = 0.2,
                 test_split: float = 0.1,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 patch_size: int = 512,
                 preprocessing_config: Optional[Dict[str, Any]] = None,
                 augment: bool = True):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, image_dir)
        self.label_dir = os.path.join(data_dir, label_dir)
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.augment = augment
        self.preprocessing = PreprocessingPipeline(
            **(preprocessing_config or {})
        )
        self._load_file_paths()
        self._split_data()
        logger.info(f"MRCDataModule initialized: train={len(self.train_paths)}, "
                    f"val={len(self.val_paths)}, test={len(self.test_paths)}")

    def _load_file_paths(self) -> None:
        image_extensions = ['*.mrc', '*.mrcs', '*.tif', '*.tiff']
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(sorted(glob.glob(os.path.join(self.image_dir, ext))))
        self.label_paths = []
        for img_path in self.image_paths:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            for ext in ['.npy', '.mrc', '.npz']:
                label_path = os.path.join(self.label_dir, basename + ext)
                if os.path.exists(label_path):
                    self.label_paths.append(label_path)
                    break
            else:
                self.label_paths.append(None)
        logger.info(f"Found {len(self.image_paths)} images, {sum(1 for l in self.label_paths if l)} labels")

    def _split_data(self) -> None:
        indices = list(range(len(self.image_paths)))
        random.shuffle(indices)
        n = len(indices)
        n_test = int(n * self.test_split)
        n_val = int(n * self.val_split)
        self.test_indices = indices[:n_test]
        self.val_indices = indices[n_test:n_test + n_val]
        self.train_indices = indices[n_test + n_val:]
        self.train_paths = [self.image_paths[i] for i in self.train_indices]
        self.train_labels = [self.label_paths[i] for i in self.train_indices]
        self.val_paths = [self.image_paths[i] for i in self.val_indices]
        self.val_labels = [self.label_paths[i] for i in self.val_indices]
        self.test_paths = [self.image_paths[i] for i in self.test_indices]
        self.test_labels = [self.label_paths[i] for i in self.test_indices]

    def train_dataloader(self) -> DataLoader:
        dataset = CryoEMDataset(
            self.train_paths, self.train_labels,
            preprocessing_pipeline=self.preprocessing,
            augment=self.augment,
            patch_size=self.patch_size
        )
        return DataLoader(dataset, batch_size=self.batch_size,
                         shuffle=True, num_workers=self.num_workers,
                         pin_memory=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        dataset = CryoEMDataset(
            self.val_paths, self.val_labels,
            preprocessing_pipeline=self.preprocessing,
            augment=False,
            patch_size=self.patch_size,
            max_patches_per_image=3
        )
        return DataLoader(dataset, batch_size=self.batch_size,
                         shuffle=False, num_workers=self.num_workers,
                         pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        dataset = CryoEMDataset(
            self.test_paths, self.test_labels,
            preprocessing_pipeline=self.preprocessing,
            augment=False,
            patch_size=self.patch_size,
            max_patches_per_image=1
        )
        return DataLoader(dataset, batch_size=1,
                         shuffle=False, num_workers=0,
                         pin_memory=True)
