import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VisualBertModel
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import torch.nn.functional as F

'''
    3.
    A main python file for thesis.
    This is where VisualBERT is trained using the train dataset.
    Early stopping is applied to ensure that the model that will be saved is at its best state. 
'''
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# DatasetLoader for JSONL files
class DatasetLoader:
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path

    def load_dataset(self):
        """Load and parse the JSONL dataset."""
        dataset = []
        with open(self.jsonl_path, "r") as file:
            for idx, line in enumerate(tqdm(file, desc="Loading Dataset")):
                entry = json.loads(line.strip())

                # Debugging for visual and clip embedding dimensions
                try:
                    visual_shape = torch.tensor(entry["visual_embeddings"]).shape
                    clip_shape = torch.tensor(entry["clip_image_embeddings"]).shape
                    # print(f"[DEBUG] Entry {idx}: visual shape {visual_shape}, clip shape {clip_shape}")
                except KeyError as e:
                    print(f"[ERROR] Missing key in Entry {idx}: {e}")
                except Exception as e:
                    print(f"[ERROR] Unexpected issue in Entry {idx}: {e}")

                dataset.append(entry)
        return dataset



# Preprocessor for text and visual embeddings
class Preprocessor:
    def __init__(self, max_len=512):
        self.max_len = max_len

    def prepare_text(self, entry):
        """Handle text_chunks or use input_ids directly."""
        if "text_chunks" in entry:  # If sliding window chunks are present
            input_ids = [torch.tensor(chunk["input_ids"], dtype=torch.long) for chunk in entry["text_chunks"]]
            attention_masks = [torch.tensor(chunk["attention_mask"], dtype=torch.long) for chunk in entry["text_chunks"]]
            return input_ids, attention_masks
        elif "input_ids" in entry:  # Single sequence
            return torch.tensor(entry["input_ids"], dtype=torch.long).unsqueeze(0), \
                   torch.tensor(entry["attention_mask"], dtype=torch.long).unsqueeze(0)
        raise ValueError(f"Unexpected entry format: {entry}")

    def prepare_visual_embeddings(self, entry):
        """Prepare visual embeddings and handle edge cases."""
        # Safely load visual embeddings
        visual_embeddings = torch.tensor(entry.get("visual_embeddings", []), dtype=torch.float)
        clip_embeddings = torch.tensor(entry.get("clip_image_embeddings", []), dtype=torch.float)

        # Ensure clip_embeddings has the correct dimensions
        if clip_embeddings.dim() == 1:
            clip_embeddings = clip_embeddings.unsqueeze(0)  # Add batch dimension if missing

        # Handle empty visual_embeddings
        if visual_embeddings.size(0) == 0:
            # Assume a default feature dimension of 2048 for empty visual embeddings
            visual_embeddings = torch.zeros((1, 2048), dtype=torch.float)

        # Handle empty clip_embeddings
        if clip_embeddings.size(0) == 0:
            # Assume a default feature dimension of 512 for empty clip embeddings
            clip_embeddings = torch.zeros((1, 512), dtype=torch.float)

        # Determine the maximum number of regions
        max_regions = max(visual_embeddings.size(0), clip_embeddings.size(0))

        # Pad visual_embeddings to match the maximum number of regions
        padded_visual_embeddings = F.pad(
            visual_embeddings,
            (0, 0, 0, max_regions - visual_embeddings.size(0)),
            mode="constant",
            value=0,
        )

        # Pad clip_embeddings to match the maximum number of regions
        padded_clip_embeddings = F.pad(
            clip_embeddings,
            (0, 0, 0, max_regions - clip_embeddings.size(0)),
            mode="constant",
            value=0,
        )

        # Concatenate along the feature dimension
        fused_embeddings = torch.cat((padded_visual_embeddings, padded_clip_embeddings), dim=-1)

        # print(f"[DEBUG] visual_embeddings shape: {visual_embeddings.shape}")
        # print(f"[DEBUG] clip_embeddings shape: {clip_embeddings.shape}")
        # print(f"[DEBUG] fused_embeddings shape: {fused_embeddings.shape}")
        return fused_embeddings

# Custom Dataset Class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocessor):
        self.dataset = dataset
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        input_ids, attention_mask = self.preprocessor.prepare_text(entry)
        fused_embeddings = self.preprocessor.prepare_visual_embeddings(entry)
        label = torch.tensor(entry["label"], dtype=torch.float)
        return input_ids, attention_mask, fused_embeddings, label


def custom_collate_fn(batch):
    input_ids, attention_masks, fused_embeddings, labels = zip(*batch)

    max_chunks = max(len(x) for x in input_ids)
    max_seq_len = max(chunk.size(0) for chunks in input_ids for chunk in chunks)
    max_regions = max(x.size(0) for x in fused_embeddings)
    max_fused_dim = max(x.size(1) for x in fused_embeddings)

    # Pad input_ids and attention_masks
    padded_input_ids, padded_attention_masks = [], []
    for chunks, masks in zip(input_ids, attention_masks):
        padded_chunks = [F.pad(chunk, (0, max_seq_len - chunk.size(0))) for chunk in chunks]
        padded_masks = [F.pad(mask, (0, max_seq_len - mask.size(0))) for mask in masks]
        padded_chunks += [torch.zeros((max_seq_len,), dtype=torch.long)] * (max_chunks - len(chunks))
        padded_masks += [torch.zeros((max_seq_len,), dtype=torch.long)] * (max_chunks - len(masks))
        padded_input_ids.append(torch.stack(padded_chunks))
        padded_attention_masks.append(torch.stack(padded_masks))

    # Pad fused_embeddings
    padded_fused_embeddings = [
        F.pad(x, (0, max_fused_dim - x.size(1), 0, max_regions - x.size(0))) for x in fused_embeddings
    ]

    labels = torch.stack(labels)
    return (
        torch.stack(padded_input_ids), 
        torch.stack(padded_attention_masks), 
        torch.stack(padded_fused_embeddings), 
        labels,
    )


# FusionModule to align embeddings
class FusionModule(torch.nn.Module):
    def __init__(self, input_dim=2560, output_dim=2048):  # Match VisualBERT's visual_projection
        super(FusionModule, self).__init__()
        self.projection_layer = torch.nn.Linear(input_dim, output_dim)

    def forward(self, fused_embeddings):
        # print(f"[DEBUG] FusionModule - Input fused_embeddings shape: {fused_embeddings.shape}")
        projected_embeddings = self.projection_layer(fused_embeddings)
        # print(f"[DEBUG] FusionModule - Output projected shape: {projected_embeddings.shape}")
        return projected_embeddings



# VisualBERT Model with Fusion
class VisualBERTWithFusion(torch.nn.Module):
    def __init__(self, visualbert_model_name="uclanlp/visualbert-vqa-coco-pre", fused_dim=2560):
        super(VisualBERTWithFusion, self).__init__()
        self.visualbert = VisualBertModel.from_pretrained(visualbert_model_name)
        self.fusion_module = FusionModule(input_dim=fused_dim, output_dim=2048)  # Match VisualBERT's input size

    def forward(self, input_ids, attention_mask, fused_embeddings):
        # print(f"[DEBUG] Input IDs shape: {input_ids.shape}")
        # print(f"[DEBUG] Attention Mask shape: {attention_mask.shape}")
        # print(f"[DEBUG] Fused Embeddings shape: {fused_embeddings.shape}")

        # Project fused_embeddings
        visual_embeds = self.fusion_module(fused_embeddings)
        # print(f"[DEBUG] Projected Visual Embeds shape: {visual_embeds.shape}")

        # Reshape visual_embeds to [batch_size, num_regions, hidden_dim]
        batch_size, num_regions, embed_dim = visual_embeds.shape
        visual_embeds = visual_embeds.view(batch_size, num_regions, -1)  # Restore batch size and regions
        # print(f"[DEBUG] Reshaped Visual Embeds shape: {visual_embeds.shape}")

        # Create visual attention mask
        visual_attention_mask = torch.ones((batch_size, num_regions), dtype=torch.float).to(visual_embeds.device)
        # print(f"[DEBUG] Visual Attention Mask shape: {visual_attention_mask.shape}")

        # Pass through VisualBERT
        outputs = self.visualbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
        )
        # print(f"[DEBUG] VisualBERT Output shape: {outputs.last_hidden_state.shape}")
        return outputs.last_hidden_state

# Classification Head
class ClassificationHead(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(ClassificationHead, self).__init__()
        self.fc = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, cls_embedding):
        return self.sigmoid(self.fc(cls_embedding))


# Training and Validation Loop
def train_and_validate(model, classification_head, train_loader, val_loader, criterion, optimizer, device, epochs=20, patience=5):
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        classification_head.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            input_ids, attention_mask, fused_embeddings, labels = batch

            # Flatten input_ids and attention_mask
            batch_size, max_chunks, seq_len = input_ids.size()
            input_ids = input_ids.view(-1, seq_len).to(device)  # [batch_size * max_chunks, seq_len]
            attention_mask = attention_mask.view(-1, seq_len).to(device)  # [batch_size * max_chunks, seq_len]
            fused_embeddings = fused_embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, fused_embeddings)
            cls_embedding = outputs[:, 0, :]  # Use [CLS] token embedding
            predictions = classification_head(cls_embedding).squeeze()
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        model.eval()
        classification_head.eval()
        total_val_loss = 0
        all_predictions, all_labels = [], []
        # Validation loop
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                input_ids, attention_mask, fused_embeddings, labels = batch

                # Flatten input_ids and attention_mask for VisualBERT
                batch_size, max_chunks, seq_len = input_ids.size()
                input_ids = input_ids.view(-1, seq_len).to(device)  # [batch_size * max_chunks, seq_len]
                attention_mask = attention_mask.view(-1, seq_len).to(device)  # [batch_size * max_chunks, seq_len]
                fused_embeddings = fused_embeddings.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask, fused_embeddings)
                cls_embedding = outputs[:, 0, :]  # Use [CLS] token embedding
                predictions = classification_head(cls_embedding).squeeze()
                
                # Compute validation loss
                loss = criterion(predictions, labels)
                total_val_loss += loss.item()

                # Collect predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())


        avg_val_loss = total_val_loss / len(val_loader)
        binary_predictions = (torch.tensor(all_predictions) >= 0.5).int().numpy()
        conf_matrix = confusion_matrix(all_labels, binary_predictions)
        accuracy = accuracy_score(all_labels, binary_predictions)
        precision = precision_score(all_labels, binary_predictions)
        recall = recall_score(all_labels, binary_predictions)
        f1 = f1_score(all_labels, binary_predictions)

        print(f"Epoch {epoch+1} Metrics:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        print("  Confusion Matrix:")
        print(conf_matrix)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'classification_head_state_dict': classification_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "VisualBERT_Model.pth")
            print(f"  Model saved to: VisualBERT_Model")
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss for {early_stop_counter} epoch(s).")
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break


# Main Script
if __name__ == "__main__":
    train_dataset_path = r"K:\Thesis\MAIN\train_with_clip_embeddings.jsonl"
    val_dataset_path = r"K:\Thesis\MAIN\dev_with_clip_embeddings.jsonl"

    dataset_loader = DatasetLoader(train_dataset_path)
    train_dataset = dataset_loader.load_dataset()

    dataset_loader = DatasetLoader(val_dataset_path)
    val_dataset = dataset_loader.load_dataset()

    model = VisualBERTWithFusion().to(device)
    classification_head = ClassificationHead(hidden_dim=768).to(device)

    preprocessor = Preprocessor()
    train_dataset = CustomDataset(train_dataset, preprocessor)
    val_dataset = CustomDataset(val_dataset, preprocessor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.79]).to(device))
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classification_head.parameters()), lr=1e-5)

    train_and_validate(model, classification_head, train_loader, val_loader, criterion, optimizer, device)
