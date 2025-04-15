import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import EmoticMultiModalDataset, test_annotations_path, IMAGE_FOLDER_TEST, context_transform, \
    body_transform, MultiModalEmotic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiModalEmotic().to(DEVICE)
model.load_state_dict(torch.load("best_emotic_model.pth", map_location=DEVICE))
model.eval()


def test_scikit_ap(cat_preds, cat_labels, ind2cat):
    ap = np.zeros(26, dtype=np.float32)
    for i in range(26):
        ap[i] = average_precision_score(cat_labels[:, i], cat_preds[:, i])
        print(f'Category {ind2cat[i]:<16} {ap[i]:.5f}')
    print(f'Mean AP {ap.mean():.5f}')
    plt.figure(figsize=(10, 5))
    plt.bar(range(26), ap, tick_label=[ind2cat[i] for i in range(26)])
    plt.xticks(rotation=90)
    plt.ylabel('Average Precision')
    plt.title('Average Precision per Emotion Category')
    plt.savefig("average_precision.png")
    return ap


def test_vad(cont_preds, cont_labels, ind2vad):
    vad = np.zeros(3, dtype=np.float32)
    for i in range(3):
        vad[i] = np.mean(np.abs(cont_preds[:, i] - cont_labels[:, i]))
        print(f'Continuous {ind2vad[i]:<10} {vad[i]:.5f}')
    print(f'Mean VAD Error {vad.mean():.5f}')
    plt.figure(figsize=(6, 4))
    plt.bar(range(3), vad, tick_label=[ind2vad[i] for i in range(3)], color=['r', 'g', 'b'])
    plt.ylabel('Mean Absolute Error')
    plt.title('VAD Errors')
    plt.savefig("vad_errors.png")
    return vad


def get_thresholds(cat_preds, cat_labels):
    thresholds = np.zeros(26, dtype=np.float32)
    plt.figure(figsize=(10, 8))
    for i in range(26):
        p, r, t = precision_recall_curve(cat_labels[:, i], cat_preds[:, i])
        plt.plot(r, p, label=f'Class {i}')
        for k in range(len(p)):
            if p[k] == r[k]:
                thresholds[i] = t[k]
                break
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.savefig("precision_recall_curves.png")
    np.save("thresholds.npy", thresholds)
    return thresholds


def evaluate(model, dataloader):
    cat_preds, cat_labels = [], []
    cont_preds, cont_labels = [], []
    with torch.no_grad():
        for (contexts, bodies), labels in tqdm(dataloader, desc="Evaluating"):
            contexts, bodies = contexts.to(DEVICE), bodies.to(DEVICE)
            cont_gt, cat_gt = labels['continuous'].to(DEVICE), labels['categorical'].to(DEVICE)
            cont_out, cat_out = model(contexts, bodies)

            cat_preds.append(cat_out.cpu().numpy())
            cat_labels.append(cat_gt.cpu().numpy())
            cont_preds.append(cont_out.cpu().numpy())
            cont_labels.append(cont_gt.cpu().numpy())

    return np.vstack(cat_preds), np.vstack(cat_labels), np.vstack(cont_preds), np.vstack(cont_labels)


ind2cat = {i: cat for i, cat in enumerate([
    'Peace', 'Affection', 'Esteem', 'Anticipation', 'Engagement',
    'Confidence', 'Happiness', 'Pleasure', 'Excitement', 'Surprise',
    'Sympathy', 'Doubt/Confusion', 'Disconnection', 'Fatigue',
    'Embarrassment', 'Yearning', 'Disapproval', 'Aversion', 'Annoyance',
    'Anger', 'Sensitivity', 'Sadness', 'Disquietment', 'Fear', 'Pain',
    'Suffering'])}
ind2vad = {0: 'Valence', 1: 'Arousal', 2: 'Dominance'}

test_loader = DataLoader(
    EmoticMultiModalDataset(
        csv_path=test_annotations_path,
        context_dir=IMAGE_FOLDER_TEST + "\\context",
        body_dir=IMAGE_FOLDER_TEST + "\\body",
        context_transform=context_transform,
        body_transform=body_transform
    ),
    batch_size=32,
    shuffle=False
)

cat_preds, cat_labels, cont_preds, cont_labels = evaluate(model, test_loader)

test_scikit_ap(cat_preds, cat_labels, ind2cat)
test_vad(cont_preds, cont_labels, ind2vad)
get_thresholds(cat_preds, cat_labels)
