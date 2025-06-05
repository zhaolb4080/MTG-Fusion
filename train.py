import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from models.fusion_net import MTGFusionNet
from models.losses import MTG_FusionLoss
from utils.text_utils import get_text_embeddings, get_image_to_text_embeddings
from config import (TRAIN_IMAGE_DIR, TRAIN_TEXT_DESC_FILE,
                    VAL_IMAGE_DIR, VAL_TEXT_DESC_FILE,
                    IMAGE_SIZE, FEATURE_DIM, TEXT_EMBED_DIM,
                    HIDDEN_DIM, EPOCHS_STAGE1, EPOCHS_STAGE2,
                    BATCH_SIZE, LEARNING_RATE, ALPHA, BETA, GAMMA, ETA,
                    CHECKPOINT_DIR, DEVICE)


# -------------------------------------------------------------------
# Each __getitem__ returns:
#   imgA_tensor, imgB_tensor: (1,H,W)
#   textA, textB, textAB, textAS, textBS, textGT: six Python strings
# -------------------------------------------------------------------
class FusionDataset(Dataset):
    def __init__(self, image_pairs: list, text_descriptions: dict):
        """
        image_pairs: list of tuples [(pathA, pathB), ...]
        text_descriptions: dict mapping index → (textA, textB, textAB, textAS, textBS, textGT)
        """
        self.pairs = image_pairs
        self.text_desc = text_descriptions

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pathA, pathB = self.pairs[idx]
        # Stub: generate random images of size IMAGE_SIZE
        H, W = IMAGE_SIZE
        imgA = torch.rand(1, H, W)
        imgB = torch.rand(1, H, W)

        # Retrieve text strings for this index
        textA, textB, textAB, textAS, textBS, textGT = self.text_desc[idx]
        return imgA, imgB, textA, textB, textAB, textAS, textBS, textGT


def collate_fn(batch: list):
    """
    Converts a list of dataset __getitem__ returns into batched tensors & lists.
    """
    imgA_list, imgB_list = [], []
    txtA_list, txtB_list, txtAB_list = [], [], []
    txtAS_list, txtBS_list, txtGT_list = [], [], []

    for (imgA, imgB, tA, tB, tAB, tAS, tBS, tGT) in batch:
        imgA_list.append(imgA)
        imgB_list.append(imgB)
        txtA_list.append(tA)
        txtB_list.append(tB)
        txtAB_list.append(tAB)
        txtAS_list.append(tAS)
        txtBS_list.append(tBS)
        txtGT_list.append(tGT)

    imgA_batch = torch.stack(imgA_list, dim=0)
    imgB_batch = torch.stack(imgB_list, dim=0)
    return imgA_batch, imgB_batch, txtA_list, txtB_list, txtAB_list, txtAS_list, txtBS_list, txtGT_list


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def train_mtg_fusion():
    # 1) Prepare training and validation datasets (stub)
    #    Replace with actual image file lists and text-description loading.
    train_image_pairs = [...]  # e.g., list of (imgA_path, imgB_path)
    train_text_desc = {
        idx: ("Description A", "Description B", "Description AB",
              "Description AS", "Description BS", "Description GT")
        for idx in range(len(train_image_pairs))
    }
    train_dataset = FusionDataset(train_image_pairs, train_text_desc)

    # If you have a validation set, set it up similarly:
    val_dataset = None
    # val_image_pairs = [...]
    # val_text_desc = { ... }
    # val_dataset = FusionDataset(val_image_pairs, val_text_desc)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                collate_fn=collate_fn)

    # 2) Instantiate model
    model = MTGFusionNet(image_size=IMAGE_SIZE,
                        feature_dim=FEATURE_DIM,
                        text_embed_dim=TEXT_EMBED_DIM,
                        hidden_dim=HIDDEN_DIM).to(DEVICE)

    # 3) Stage 1: Pre-training (no text guidance)
    criterion_stage1 = MTG_FusionLoss(alpha=ALPHA,
                                      beta=BETA,
                                      gamma=GAMMA,
                                      use_text=False).to(DEVICE)
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           betas=(0.9, 0.999))

    print("=== Stage 1: Pre-training (Purely Visual) ===")
    model.train()
    for epoch in range(EPOCHS_STAGE1):
        running_loss = 0.0
        for batch in train_loader:
            imgA, imgB, _, _, _, _, _, _ = batch
            imgA = imgA.to(DEVICE)
            imgB = imgB.to(DEVICE)

            optimizer.zero_grad()

            # Forward with dummy zero-text embeddings
            B = imgA.size(0)
            dummy_text = torch.zeros(B, 1, TEXT_EMBED_DIM, device=DEVICE)
            fused_img, _, _ = model(
                imgA, imgB,
                tA_embed=dummy_text,
                tB_embed=dummy_text,
                tAB_embed=dummy_text,
                tAS_embed=dummy_text,
                tBS_embed=dummy_text,
                tGT_embed=None
            )

            loss = criterion_stage1(fused_img, imgA, imgB)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * B

        epoch_loss = running_loss / len(train_dataset)
        print(f"[Stage1] Epoch {epoch + 1}/{EPOCHS_STAGE1}  Loss: {epoch_loss:.4f}")

        # LR decay every 20 epochs
        if (epoch + 1) % 20 == 0:
            for g in optimizer.param_groups:
                g["lr"] *= 0.5

    # Save Stage 1 checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "stage1.pth"))

    # 4) Stage 2: Fine-tuning with text guidance
    print("\n=== Stage 2: Fine-tuning (Text Guidance) ===")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "stage1.pth")))
    criterion_stage2 = MTG_FusionLoss(alpha=ALPHA,
                                      beta=BETA,
                                      gamma=GAMMA,
                                      eta=ETA,
                                      use_text=True).to(DEVICE)

    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE * 0.5,
                           betas=(0.9, 0.999))
    model.train()

    for epoch in range(EPOCHS_STAGE2):
        running_loss = 0.0
        for batch in train_loader:
            imgA, imgB, txtA_list, txtB_list, txtAB_list, txtAS_list, txtBS_list, txtGT_list = batch
            imgA = imgA.to(DEVICE)
            imgB = imgB.to(DEVICE)

            # 1) Obtain word embeddings for text strings via BLIP (stubbed)
            B = imgA.size(0)
            tA_embed = get_text_embeddings(txtA_list, max_tokens=32, embed_dim=TEXT_EMBED_DIM, device=DEVICE)
            tB_embed = get_text_embeddings(txtB_list, max_tokens=32, embed_dim=TEXT_EMBED_DIM, device=DEVICE)
            tAB_embed = get_text_embeddings(txtAB_list, max_tokens=32, embed_dim=TEXT_EMBED_DIM, device=DEVICE)
            tAS_embed = get_text_embeddings(txtAS_list, max_tokens=32, embed_dim=TEXT_EMBED_DIM, device=DEVICE)
            tBS_embed = get_text_embeddings(txtBS_list, max_tokens=32, embed_dim=TEXT_EMBED_DIM, device=DEVICE)
            tGT_embed = get_text_embeddings(txtGT_list, max_tokens=32, embed_dim=TEXT_EMBED_DIM, device=DEVICE)

            optimizer.zero_grad()

            # 2) Forward pass with text guidance
            fused_img, _, _ = model(
                imgA, imgB,
                tA_embed=tA_embed,
                tB_embed=tB_embed,
                tAB_embed=tAB_embed,
                tAS_embed=tAS_embed,
                tBS_embed=tBS_embed,
                tGT_embed=tGT_embed
            )

            # 3) Compute fused-image text embedding tF via BLIP (stubbed)
            tF_embed_3D = get_image_to_text_embeddings(fused_img,
                                                        max_tokens=32,
                                                        embed_dim=TEXT_EMBED_DIM,
                                                        device=DEVICE)
            # Average across token dimension → (B, D)
            tF_embed = tF_embed_3D.mean(dim=1)
            tA_avg = tA_embed.mean(dim=1)
            tB_avg = tB_embed.mean(dim=1)
            tGT_avg = tGT_embed.mean(dim=1)

            # 4) Compute Stage 2 loss
            loss = criterion_stage2(fused_img, imgA, imgB,
                                    tF_embed=tF_embed,
                                    tA_embed=tA_avg,
                                    tB_embed=tB_avg,
                                    tGT_embed=tGT_avg)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * B

        epoch_loss = running_loss / len(train_dataset)
        print(f"[Stage2] Epoch {epoch + 1}/{EPOCHS_STAGE2}  Loss: {epoch_loss:.4f}")

        # LR decay every 20 epochs
        if (epoch + 1) % 20 == 0:
            for g in optimizer.param_groups:
                g["lr"] *= 0.5

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "stage2_final.pth"))
    print("Training complete. Final model saved at:", os.path.join(CHECKPOINT_DIR, "stage2_final.pth"))

    return model


if __name__ == "__main__":
    train_mtg_fusion()
