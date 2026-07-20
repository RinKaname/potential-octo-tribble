import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- CONSTANTS & MAPPINGS ---
BEAT_TYPES = ["N", "S", "V", "U"]
BEAT_TO_IDX = {"N": 1, "S": 2, "V": 3, "U": 4} # 0 is background
IDX_TO_BEAT = {1: "N", 2: "S", 3: "V", 4: "U"}

RHYTHM_FAMILIES = [
    "sinus_rhythm",
    "atrial_fibrillation_flutter",
    "patterned_atrial_ectopy",
    "patterned_ventricular_ectopy",
    "sinus_node_dysfunction",
    "atrioventricular_block",
    "supraventricular_tachyarrhythmia",
    "ventricular_tachyarrhythmia",
    "wandering_multifocal_atrial_rhythm"
]
RHYTHM_TO_IDX = {r: i for i, r in enumerate(RHYTHM_FAMILIES)}
IDX_TO_RHYTHM = {i: r for i, r in enumerate(RHYTHM_FAMILIES)}

SEQ_LEN = 750

# --- DATASET ---
class ECGDataset(Dataset):
    def __init__(self, df, data_dir, is_train=True):
        self.df = df
        self.data_dir = data_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig_path = os.path.join(self.data_dir, row['signal'])

        # IN COMPETITION, `sig_path` will point to the valid path since `data_dir` is provided.
        # But for testing locally without actual files, let's mock it if it doesn't exist
        if not os.path.exists(sig_path):
            signal = np.zeros(SEQ_LEN, dtype=np.float32)
        else:
            signal = np.load(sig_path).astype(np.float32)

        # expand dims for channel
        signal = np.expand_dims(signal, axis=0)

        if not self.is_train:
            return torch.tensor(signal), row['id']

        rhythm_idx = RHYTHM_TO_IDX[row['rhythm_family']]
        beats = json.loads(row['beats'])

        beat_seq = np.zeros(SEQ_LEN, dtype=np.int64)
        for sample_idx, beat_type in beats:
            b_idx = BEAT_TO_IDX[beat_type]
            start = max(0, sample_idx - 2)
            end = min(SEQ_LEN, sample_idx + 3)
            beat_seq[start:end] = b_idx

        return torch.tensor(signal), torch.tensor(beat_seq), torch.tensor(rhythm_idx)

# --- MODEL ARCHITECTURE ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        res = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(res)
        x = self.relu(x)
        return x

class ECGModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.layer1 = ConvBlock(32, 64, dilation=1)
        self.layer2 = ConvBlock(64, 64, dilation=2)
        self.layer3 = ConvBlock(64, 128, dilation=4)
        self.layer4 = ConvBlock(128, 128, dilation=8)
        self.layer5 = ConvBlock(128, 128, dilation=16)

        self.event_head = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 5, kernel_size=1)
        )

        self.rhythm_pool = nn.AdaptiveAvgPool1d(1)
        self.rhythm_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 9)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = self.layer5(x)

        event_out = self.event_head(features)

        pooled = self.rhythm_pool(features).squeeze(-1)
        rhythm_out = self.rhythm_head(pooled)

        return event_out, rhythm_out

# --- TRAINING AND INFERENCE UTILS ---

def get_class_weights(df):
    rhythm_counts = df['rhythm_family'].value_counts()
    rhythm_weights = []
    for r in RHYTHM_FAMILIES:
        count = rhythm_counts.get(r, 0)
        weight = 1.0 / (count + 10)
        rhythm_weights.append(weight)

    rhythm_weights = torch.tensor(rhythm_weights, dtype=torch.float32)
    rhythm_weights = rhythm_weights / rhythm_weights.sum() * len(RHYTHM_FAMILIES)

    event_weights = torch.tensor([0.1, 1.0, 1.5, 5.0, 10.0], dtype=torch.float32)
    return event_weights, rhythm_weights

def extract_peaks(event_probs, threshold=0.3):
    pred_classes = np.argmax(event_probs, axis=0)
    max_probs = np.max(event_probs, axis=0)

    beats = []
    i = 0
    while i < SEQ_LEN:
        if pred_classes[i] > 0 and max_probs[i] > threshold:
            start = i
            while i < SEQ_LEN and pred_classes[i] > 0 and max_probs[i] > threshold:
                i += 1
            end = i

            region_non_bg_sum = np.sum(event_probs[1:, start:end], axis=0)
            local_peak = start + np.argmax(region_non_bg_sum)
            peak_class = pred_classes[local_peak]

            if peak_class > 0:
                beats.append([int(local_peak), IDX_TO_BEAT[peak_class]])
        else:
            i += 1

    return beats


# --- TRAINING LOOP ---
def train_model(model, train_loader, epochs=5, lr=1e-3, device='cpu', event_weights=None, rhythm_weights=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

    event_criterion = nn.CrossEntropyLoss(weight=event_weights.to(device))
    rhythm_criterion = nn.CrossEntropyLoss(weight=rhythm_weights.to(device))

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        total_event_loss = 0
        total_rhythm_loss = 0

        for batch_idx, (signals, event_targets, rhythm_targets) in enumerate(train_loader):
            signals = signals.to(device)
            event_targets = event_targets.to(device)
            rhythm_targets = rhythm_targets.to(device)

            optimizer.zero_grad()

            event_out, rhythm_out = model(signals)

            loss_event = event_criterion(event_out, event_targets)
            loss_rhythm = rhythm_criterion(rhythm_out, rhythm_targets)

            loss = loss_event + loss_rhythm

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_event_loss += loss_event.item()
            total_rhythm_loss += loss_rhythm.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Event: {total_event_loss/len(train_loader):.4f} | Rhythm: {total_rhythm_loss/len(train_loader):.4f}")

    return model


# --- INFERENCE ---
def predict(model, test_loader, device='cpu'):
    model.eval()
    predictions = []

    with torch.no_grad():
        for signals, ids in test_loader:
            signals = signals.to(device)
            event_out, rhythm_out = model(signals)

            event_probs = F.softmax(event_out, dim=1).cpu().numpy()
            rhythm_probs = F.softmax(rhythm_out, dim=1).cpu().numpy()

            for i in range(len(ids)):
                rhythm_idx = np.argmax(rhythm_probs[i])
                rhythm = IDX_TO_RHYTHM[rhythm_idx]

                beats = extract_peaks(event_probs[i], threshold=0.35)

                predictions.append({
                    'id': ids[i],
                    'rhythm_family': rhythm,
                    'beats': json.dumps(beats)
                })

    return pd.DataFrame(predictions)

# --- MAIN ---
def main():
    if len(sys.argv) != 3:
        print("Usage: python solution.py <data_dir> <submission_path>")
        sys.exit(1)

    data_dir = sys.argv[1]
    submission_path = sys.argv[2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    if not os.path.exists(train_path):
        train_path = 'train.csv'
    if not os.path.exists(test_path):
        test_path = 'test.csv'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    event_weights, rhythm_weights = get_class_weights(train_df)

    train_dataset = ECGDataset(train_df, data_dir=data_dir, is_train=True)
    test_dataset = ECGDataset(test_df, data_dir=data_dir, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = ECGModel()

    print("Training model...")
    # Train for 15 epochs for better accuracy. CPU will be fine for this small dataset and model
    model = train_model(model, train_loader, epochs=15, lr=2e-3, device=device,
                       event_weights=event_weights, rhythm_weights=rhythm_weights)

    print("Generating predictions...")
    sub_df = predict(model, test_loader, device=device)

    sub_df = sub_df[['id', 'rhythm_family', 'beats']]
    sub_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    main()
