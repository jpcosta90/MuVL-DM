import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from cavl_doc.utils.image_processing import build_transform, dynamic_preprocess

class DocumentPairDataset(Dataset):
    def __init__(self, csv_path, base_dir, input_size=448, max_num=12, device="cuda"):
        """
        csv_path: path para train_pairs.csv
        base_dir: caminho base onde estão as imagens (ex: data/)
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.input_size = input_size
        self.max_num = max_num
        self.device = device
        self.transform = build_transform(input_size)

    def __len__(self):
        return len(self.df)

    def _load_tensor(self, rel_path):
        img_path = os.path.join(self.base_dir, rel_path)
        image = Image.open(img_path).convert("RGB")
        blocks = dynamic_preprocess(image, image_size=self.input_size, use_thumbnail=True, max_num=self.max_num)
        tensor = [self.transform(b) for b in blocks]
        return torch.stack(tensor).to(torch.bfloat16).to(self.device)  # [N, 3, H, W]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_a = self._load_tensor(row["file_a_path"])
        img_b = self._load_tensor(row["file_b_path"])
        label = float(row["is_equal"])
        return {
            "image_a": img_a,
            "image_b": img_b,
            "label": label
        }
