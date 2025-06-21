import os
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from tqdm import tqdm
from torchvision import models, transforms
from transformers import AutoModel, AutoTokenizer

def load_data_list(txt_file_path):
    data = []
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            image_file, question_answer, match_label = parts
            label = 1 if match_label.lower() == "match" else 0
            data.append((image_file, question_answer, label))
    return data

class VQADataset(Dataset):
    def __init__(self, samples, image_root, image_processor, tokenizer, max_text_length=32):
        self.samples = samples
        self.image_root = image_root
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_file, text, label = self.samples[idx]
        img_path = os.path.join(self.image_root, image_file)
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.image_processor(image)

        text_inputs = self.tokenizer(
            text=text,
            truncation=True,
            padding='max_length',
            max_length=self.max_text_length,
            return_tensors='pt'
        )

        return {
            'pixel_values': pixel_values,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ResNetRobertaClassifier(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=2):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity() 

        self.roberta = AutoModel.from_pretrained("roberta-base")
        for p in self.resnet.parameters():
            p.requires_grad = False
        for p in self.roberta.parameters():
            p.requires_grad = False

        self.image_proj = nn.Linear(2048, hidden_dim)
        self.text_proj = nn.Linear(self.roberta.config.hidden_size, hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        img_feat = self.resnet(pixel_values)                       
        txt_feat = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]  

        img_proj = self.image_proj(img_feat)                       
        txt_proj = self.text_proj(txt_feat)                         
        fused = torch.cat([img_proj, txt_proj], dim=1)             

        logits = self.classifier(fused)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = nn.CrossEntropyLoss()(logits, labels)
        return output

def mean_reciprocal_rank(pred_logits, labels, group_size=4):
    logits = pred_logits.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    ranks = []
    for i in range(0, len(logits), group_size):
        scores = logits[i:i+group_size, 1]
        group_labels = labels[i:i+group_size]
        sorted_indices = np.argsort(scores)[::-1]
        for rank, idx in enumerate(sorted_indices, start=1):
            if group_labels[idx] == 1:
                ranks.append(1.0 / rank)
                break
    return np.mean(ranks)

def main():

    train_txt = r"/workspaces/cmp9137-advanced-machine-learning/CMP9137 Advanced Machine Learning/ITM_Classifier-baselines/visual7w-text/v7w.TrainImages.itm.txt"
    test_txt = r"/workspaces/cmp9137-advanced-machine-learning/CMP9137 Advanced Machine Learning/ITM_Classifier-baselines/visual7w-text/v7w.TestImages.itm.txt"
    image_root = r"/workspaces/cmp9137-advanced-machine-learning/CMP9137 Advanced Machine Learning/ITM_Classifier-baselines/visual7w-images"
    save_model_path = "resnet_roberta_vqa.pth"

    train_samples = load_data_list(train_txt)
    test_samples = load_data_list(test_txt)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = VQADataset(train_samples, image_root, transform, tokenizer)
    test_dataset = VQADataset(test_samples, image_root, transform, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    df = pd.DataFrame(train_samples, columns=["Image", "QA", "Label"])
    sns.countplot(data=df, x="Label")
    plt.xticks([0, 1], ['No Match', 'Match'])
    plt.title("Class Distribution in Training Set")
    plt.show()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetRobertaClassifier().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("Training started...")
    num_epochs = 30
    train_acc, train_prec, train_rec = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_preds, epoch_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            output = model(**inputs, labels=labels)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = torch.argmax(torch.softmax(output["logits"], dim=1), dim=1)
            epoch_preds.extend(preds.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(epoch_labels, epoch_preds)
        prec = precision_score(epoch_labels, epoch_preds)
        rec = recall_score(epoch_labels, epoch_preds)
        train_acc.append(acc)
        train_prec.append(prec)
        train_rec.append(rec)

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

    torch.save(model.state_dict(), save_model_path)

    plt.plot(range(1, num_epochs+1), train_acc, label="Accuracy")
    plt.plot(range(1, num_epochs+1), train_prec, label="Precision")
    plt.plot(range(1, num_epochs+1), train_rec, label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid()
    plt.show()

    model.eval()
    all_logits, all_preds, all_probs, all_labels = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            output = model(**inputs, labels=labels)
            logits = output["logits"]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_logits.append(logits)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.tensor(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    mrr = mean_reciprocal_rank(logits_tensor, labels_tensor)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nTest Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, MRR: {mrr:.4f}")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Match", "Match"], yticklabels=["No Match", "Match"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid()
    plt.show()

    pr, re, _ = precision_recall_curve(all_labels, all_probs)
    plt.plot(re, pr)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()