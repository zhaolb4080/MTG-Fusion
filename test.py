# test.py

"""
Test script for MTG-Fusion: loads a trained model checkpoint, performs inference on test image pairs.

Usage:
    python test.py/
        --checkpoint pretrained_weights/text_fusion.pth
        --test_image_list test/image_pairs.txt
        --test_text_desc test/text_descriptions.json
        --output_dir outputs/
        --batch_size
"""

import os
import argparse
import json

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms as T

from models.fusion_net import MTGFusionNet
from utils.text_utils import get_text_embeddings, get_image_to_text_embeddings

from config import (
    IMAGE_SIZE,
    FEATURE_DIM,
    TEXT_EMBED_DIM,
    HIDDEN_DIM,
    DEVICE
)


class FusionTestDataset(Dataset):
    """
    Dataset for testing MTG-Fusion. Each item returns:
      - imgA_tensor, imgB_tensor: torch.FloatTensor (1, H, W) in [0,1]
      - textA, textB, textAB, textAS, textBS: five Python strings for text guidance
      - identifier: a unique string (e.g., file basename) for saving outputs
    """

    def __init__(self, image_list_file: str, text_desc_file: str, image_size: tuple):
        """
        Args:
          - image_list_file: path to a TXT file where each line contains two image paths separated by a whitespace:
                path/to/imageA.png path/to/imageB.png
          - text_desc_file: path to a JSON file mapping "identifier" → {
                "textA": ...,
                "textB": ...,
                "textAB": ...,
                "textAS": ...,
                "textBS": ...
            }
          - image_size: (H, W) for resizing input images
        """
        # Read image pairs
        with open(image_list_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        self.pairs = []
        for line in lines:
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid line in {image_list_file}: '{line}'")
            imgA_path, imgB_path = parts
            # Use basename (without extension) of A as identifier
            identifier = os.path.splitext(os.path.basename(imgA_path))[0]
            self.pairs.append((imgA_path, imgB_path, identifier))

        # Load text descriptions JSON
        with open(text_desc_file, 'r') as f:
            self.text_desc = json.load(f)

        self.image_size = image_size  # (H, W)
        # Define a transform: resize to (H, W), convert to tensor [0,1]
        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        imgA_path, imgB_path, identifier = self.pairs[idx]
        # Load images with PIL, apply transform → (1, H, W)
        imgA = Image.open(imgA_path).convert("RGB")
        imgB = Image.open(imgB_path).convert("RGB")
        imgA_tensor = self.transform(imgA)  # (1, H, W), values in [0,1]
        imgB_tensor = self.transform(imgB)

        # Fetch text descriptions by identifier
        if identifier not in self.text_desc:
            raise KeyError(f"Identifier '{identifier}' not found in {self.text_desc}")
        desc = self.text_desc[identifier]
        # Expecting keys: "textA", "textB", "textAB", "textAS", "textBS"
        textA = desc["textA"]
        textB = desc["textB"]
        textAB = desc["textAB"]
        textAS = desc["textAS"]
        textBS = desc["textBS"]

        return imgA_tensor, imgB_tensor, textA, textB, textAB, textAS, textBS, identifier


def collate_fn(batch):
    """
    Customized collate function for batching test items.
    Returns:
      - imgA_batch: (B,1,H,W)
      - imgB_batch: (B,1,H,W)
      - txtA_list, txtB_list, txtAB_list, txtAS_list, txtBS_list: lists of strings
      - id_list: list of identifiers
    """
    imgA_list, imgB_list = [], []
    txtA_list, txtB_list, txtAB_list = [], [], []
    txtAS_list, txtBS_list = [], []
    id_list = []

    for (imgA, imgB, tA, tB, tAB, tAS, tBS, identifier) in batch:
        imgA_list.append(imgA)
        imgB_list.append(imgB)
        txtA_list.append(tA)
        txtB_list.append(tB)
        txtAB_list.append(tAB)
        txtAS_list.append(tAS)
        txtBS_list.append(tBS)
        id_list.append(identifier)

    imgA_batch = torch.stack(imgA_list, dim=0)
    imgB_batch = torch.stack(imgB_list, dim=0)
    return imgA_batch, imgB_batch, txtA_list, txtB_list, txtAB_list, txtAS_list, txtBS_list, id_list


def save_tensor_as_image(tensor: torch.Tensor, save_path: str):
    """
    Save a single-channel tensor (1, H, W) or (H, W) in [0,1] as a PNG image.
    """
    # Move to CPU and clamp
    img = tensor.squeeze(0).clamp(0.0, 1.0)  # (H, W)
    to_pil = T.ToPILImage()
    pil_img = to_pil(img)
    pil_img.save(save_path)


def main(args):
    # 1) Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 2) Prepare test dataset & loader
    test_dataset = FusionTestDataset(
        image_list_file=args.test_image_list,
        text_desc_file=args.test_text_desc,
        image_size=IMAGE_SIZE
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 3) Instantiate model and load checkpoint
    model = MTGFusionNet(
        image_size=IMAGE_SIZE,
        feature_dim=FEATURE_DIM,
        text_embed_dim=TEXT_EMBED_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            imgA_batch, imgB_batch, txtA_list, txtB_list, txtAB_list, txtAS_list, txtBS_list, id_list = batch
            B = imgA_batch.size(0)
            imgA_batch = imgA_batch.to(DEVICE)
            imgB_batch = imgB_batch.to(DEVICE)

            # 5) Obtain text embeddings via BLIP stubs
            tA_embed = get_text_embeddings(txtA_list, max_tokens=32, embed_dim=TEXT_EMBED_DIM, device=DEVICE)
            tB_embed = get_text_embeddings(txtB_list, max_tokens=32, embed_dim=TEXT_EMBED_DIM, device=DEVICE)
            tAB_embed = get_text_embeddings(txtAB_list, max_tokens=32, embed_dim=TEXT_EMBED_DIM, device=DEVICE)
            tAS_embed = get_text_embeddings(txtAS_list, max_tokens=32, embed_dim=TEXT_EMBED_DIM, device=DEVICE)
            tBS_embed = get_text_embeddings(txtBS_list, max_tokens=32, embed_dim=TEXT_EMBED_DIM, device=DEVICE)

            # 6) Forward pass → fused images
            fused_batch, _, _ = model(
                imgA_batch, imgB_batch,
                tA_embed=tA_embed,
                tB_embed=tB_embed,
                tAB_embed=tAB_embed,
                tAS_embed=tAS_embed,
                tBS_embed=tBS_embed,
                tGT_embed=None  # Not used in inference
            )
            # fused_batch: (B, 1, H, W), values in [0,1]

            # 7) For each image in the batch: save fused output & compute metrics
            for i in range(B):
                identifier = id_list[i]
                fused_img = fused_batch[i].cpu()  # (1, H, W)
                origA = imgA_batch[i].cpu()       # (1, H, W)
                origB = imgB_batch[i].cpu()       # (1, H, W)

                # Save fused image
                save_path = os.path.join(args.output_dir, f"{identifier}_fused.png")
                save_tensor_as_image(fused_img, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MTG-Fusion on a set of image pairs")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained model checkpoint (e.g., checkpoints/text_fusion.pth)"
    )
    parser.add_argument(
        "--test_image_list", type=str, required=True,
        help="Path to a TXT file listing test image pairs (one pair per line: pathA pathB)"
    )
    parser.add_argument(
        "--test_text_desc", type=str, required=True,
        help="Path to a JSON file mapping identifiers to text descriptions"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save fused output images"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for testing"
    )
    args = parser.parse_args()

    main(args)

