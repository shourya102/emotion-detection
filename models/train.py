import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights
from torchvision.transforms import transforms
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(__name__)
IMAGE_FOLDER_TRAIN = os.path.join(CURRENT_DIR, "images_split\\train")
IMAGE_FOLDER_TEST = os.path.join(CURRENT_DIR, "images_split\\test")
IMAGEE_FOLDER_VAL = os.path.join(CURRENT_DIR, "images_split\\val")

train_annotations_path = os.path.join(CURRENT_DIR, "annotations\\annot_arrs_train.csv")
valid_annotations_path = os.path.join(CURRENT_DIR, "annotations\\annot_arrs_val.csv")
test_annotations_path = os.path.join(CURRENT_DIR, "annotations\\annot_arrs_test.csv")

BATCH_SIZE = 32
NUM_EPOCH = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiModalEmotic(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.context_net.fc = nn.Identity()
        self.body_net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.body_net.fc = nn.Identity()
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.continuous_head = nn.Linear(512, 3)
        self.categorical_head = nn.Linear(512, 26)

    def forward(self, context, body):
        context_feats = self.context_net(context)
        body_feats = self.body_net(body)
        combined = torch.cat([context_feats, body_feats], dim=1)
        fused = self.fusion(combined)
        cont_output = self.continuous_head(fused)
        cat_output = self.categorical_head(fused)
        return cont_output, cat_output


class EmoticMultiModalDataset(Dataset):
    def __init__(self, csv_path, context_dir, body_dir, context_transform=None, body_transform=None):
        self.df = pd.read_csv(csv_path)
        self.context_dir = context_dir
        self.body_dir = body_dir
        self.context_transform = context_transform
        self.body_transform = body_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        context_arr = np.load(os.path.join(self.context_dir, row['Arr_name']))
        body_arr = np.load(os.path.join(self.body_dir, row['Crop_name']))
        context = torch.from_numpy(context_arr).float().permute(2, 0, 1)
        body = torch.from_numpy(body_arr).float().permute(2, 0, 1)
        if self.body_transform:
            body = self.body_transform(body)
        if self.context_transform:
            context = self.context_transform(context)
        return (context, body), {
            'continuous': torch.tensor(row[5:8].values.astype(np.float32)),
            'categorical': torch.tensor(row[8:34].values.astype(np.int8))
        }


class DiscreteLoss(nn.Module):
    def __init__(self, weight_type='mean', device=torch.device('cpu')):
        super(DiscreteLoss, self).__init__()
        self.weight_type = weight_type
        self.device = device
        if self.weight_type == 'mean':
            self.weights = torch.ones((1, 26)) / 26.0
            self.weights = self.weights.to(self.device)
        elif self.weight_type == 'static':
            self.weights = torch.FloatTensor([0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                                              0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                                              0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520,
                                              0.1537]).unsqueeze(0)
            self.weights = self.weights.to(self.device)

    def forward(self, pred, target):
        if self.weight_type == 'dynamic':
            self.weights = self.prepare_dynamic_weights(target)
            self.weights = self.weights.to(self.device)
        loss = (((pred - target) ** 2) * self.weights)
        return loss.sum()

    def prepare_dynamic_weights(self, target):
        target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
        weights = torch.zeros((1, 26))
        weights[target_stats != 0] = 1.0 / torch.log(target_stats[target_stats != 0].data + 1.2)
        weights[target_stats == 0] = 0.0001
        return weights


model = MultiModalEmotic().to(DEVICE)

context_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

body_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(
    EmoticMultiModalDataset(
        csv_path=train_annotations_path,
        context_dir=IMAGE_FOLDER_TRAIN + "\\context",
        body_dir=IMAGE_FOLDER_TRAIN + "\\body",
        context_transform=context_transform,
        body_transform=body_transform
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    EmoticMultiModalDataset(
        csv_path=valid_annotations_path,
        context_dir=IMAGEE_FOLDER_VAL + "\\context",
        body_dir=IMAGEE_FOLDER_VAL + "\\body",
        context_transform=context_transform,
        body_transform=body_transform
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

for param in model.context_net.parameters():
    param.requires_grad = False
for param in model.body_net.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam([
    {'params': model.context_net.parameters(), 'lr': 1e-6},
    {'params': model.body_net.parameters(), 'lr': 1e-5},
    {'params': model.fusion.parameters(), 'lr': 1e-3},
    {'params': model.continuous_head.parameters(), 'lr': 1e-4},
    {'params': model.categorical_head.parameters(), 'lr': 1e-3}
])


def categorical_accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    true_classes = torch.argmax(labels, dim=1)
    correct = (predicted == true_classes).float()
    return correct.mean()


def continuous_accuracy(pred, target):
    return torch.mean(1.0 - torch.abs(pred - target) / 10.0)


def train():
    continuous_loss = nn.MSELoss()
    categorical_loss = DiscreteLoss(device=DEVICE)
    best_val_loss = 100
    for epoch in range(NUM_EPOCH):
        model.train()
        total_train_loss = 0
        total_cat_acc = 0
        total_cont_acc = 0
        num_batches = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCH} [Training]", leave=False)
        for (contexts, bodies), labels in train_bar:
            contexts = contexts.to(DEVICE)
            bodies = bodies.to(DEVICE)
            cont_labels = labels['continuous'].to(DEVICE)
            cat_labels = labels['categorical'].float().to(DEVICE)
            optimizer.zero_grad()
            cont_pred, cat_pred = model(contexts, bodies)
            cat_loss_batch = categorical_loss(cat_pred, cat_labels)
            cont_loss_batch = continuous_loss(cont_pred, cont_labels)
            loss = ((categorical_loss(cat_pred, cat_labels)) + (continuous_loss(cont_pred, cont_labels))) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}", cont_loss=f"{cont_loss_batch.item():.4f}",
                                  cat_loss=f"{cat_loss_batch.item():.4f}")
            total_cat_acc += categorical_accuracy(cat_pred, cat_labels).item()
            total_cont_acc += continuous_accuracy(cont_pred, cont_labels).item()
            num_batches += 1
        avg_train_loss = total_train_loss / len(train_loader)
        avg_cat_acc = total_cat_acc / num_batches
        avg_cont_acc = total_cont_acc / num_batches

        model.eval()
        total_val_loss = 0
        valid_bar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCH} [Validation]", leave=False)
        with torch.no_grad():
            for (contexts, bodies), labels in valid_bar:
                contexts = contexts.to(DEVICE)
                bodies = bodies.to(DEVICE)
                cont_labels = labels['continuous'].to(DEVICE)
                cat_labels = labels['categorical'].to(DEVICE)
                cont_pred, cat_pred = model(contexts, bodies)
                val_loss = ((categorical_loss(cat_pred, cat_labels)) + (continuous_loss(cont_pred, cont_labels))) / 2
                total_val_loss += val_loss.item()
                valid_bar.set_postfix(val_loss=f"{val_loss.item():.4f}")
        avg_val_loss = total_val_loss / len(valid_loader)
        print(
            f"\nEpoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Categorical Accuracy: {avg_cat_acc:.4f}, Continuous Accuracy: {avg_cont_acc:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_emotic_model.pth")
            print("Best model saved!")
