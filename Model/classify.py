import torch
import os
import json
from tqdm import tqdm
from transformers import VisualBertModel, CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image

"""
    4.
    Output from both model are combined through weighted sum.
    From this program, it was determined that a threshold of 0.6 is the best for this hybridized with ensembled output model.
    
    This is where the trained VisualBERT is loaded and used. It is now then use to evaluate and classify the dev dataset then further classify it using CLIP as post classification.
    
"""
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DatasetLoader
class DatasetLoader:
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path

    def load_dataset(self):
        dataset = []
        with open(self.jsonl_path, "r") as file:
            for idx, line in enumerate(tqdm(file, desc="Loading Dataset")):
                entry = json.loads(line.strip())
                dataset.append(entry)
        return dataset

# Preprocessor
class Preprocessor:
    def prepare_text(self, entry):
        if "text_chunks" in entry:
            input_ids = [torch.tensor(chunk["input_ids"], dtype=torch.long) for chunk in entry["text_chunks"]]
            attention_masks = [torch.tensor(chunk["attention_mask"], dtype=torch.long) for chunk in entry["text_chunks"]]
            return input_ids, attention_masks
        elif "input_ids" in entry:
            return torch.tensor(entry["input_ids"], dtype=torch.long).unsqueeze(0), \
                   torch.tensor(entry["attention_mask"], dtype=torch.long).unsqueeze(0)
        raise ValueError(f"Unexpected entry format: {entry}")

    def prepare_visual_embeddings(self, entry):
        visual_embeddings = torch.tensor(entry.get("visual_embeddings", []), dtype=torch.float)
        clip_embeddings = torch.tensor(entry.get("clip_image_embeddings", []), dtype=torch.float)
        if clip_embeddings.dim() == 1:
            clip_embeddings = clip_embeddings.unsqueeze(0)
        if visual_embeddings.size(0) == 0:
            visual_embeddings = torch.zeros((1, 2048), dtype=torch.float)
        if clip_embeddings.size(0) == 0:
            clip_embeddings = torch.zeros((1, 512), dtype=torch.float)

        max_regions = max(visual_embeddings.size(0), clip_embeddings.size(0))
        padded_visual_embeddings = torch.nn.functional.pad(
            visual_embeddings, (0, 0, 0, max_regions - visual_embeddings.size(0)), mode="constant", value=0)
        padded_clip_embeddings = torch.nn.functional.pad(
            clip_embeddings, (0, 0, 0, max_regions - clip_embeddings.size(0)), mode="constant", value=0)
        return torch.cat((padded_visual_embeddings, padded_clip_embeddings), dim=-1)

# CustomDataset
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

    padded_input_ids, padded_attention_masks = [], []
    for chunks, masks in zip(input_ids, attention_masks):
        padded_chunks = [torch.nn.functional.pad(chunk, (0, max_seq_len - chunk.size(0))) for chunk in chunks]
        padded_masks = [torch.nn.functional.pad(mask, (0, max_seq_len - mask.size(0))) for mask in masks]
        padded_chunks += [torch.zeros((max_seq_len,), dtype=torch.long)] * (max_chunks - len(chunks))
        padded_masks += [torch.zeros((max_seq_len,), dtype=torch.long)] * (max_chunks - len(masks))
        padded_input_ids.append(torch.stack(padded_chunks))
        padded_attention_masks.append(torch.stack(padded_masks))

    padded_fused_embeddings = [
        torch.nn.functional.pad(x, (0, max_fused_dim - x.size(1), 0, max_regions - x.size(0))) for x in fused_embeddings
    ]

    labels = torch.stack(labels)
    return (
        torch.stack(padded_input_ids),
        torch.stack(padded_attention_masks),
        torch.stack(padded_fused_embeddings),
        labels,
    )

# Models
class FusionModule(torch.nn.Module):
    def __init__(self, input_dim=2560, output_dim=2048):
        super(FusionModule, self).__init__()
        self.projection_layer = torch.nn.Linear(input_dim, output_dim)

    def forward(self, fused_embeddings):
        return self.projection_layer(fused_embeddings)

class VisualBERTWithFusion(torch.nn.Module):
    def __init__(self, visualbert_model_name="uclanlp/visualbert-vqa-coco-pre", fused_dim=2560):
        super(VisualBERTWithFusion, self).__init__()
        self.visualbert = VisualBertModel.from_pretrained(visualbert_model_name)
        self.fusion_module = FusionModule(input_dim=fused_dim, output_dim=2048)

    def forward(self, input_ids, attention_mask, fused_embeddings):
        visual_embeds = self.fusion_module(fused_embeddings)
        batch_size, num_regions, embed_dim = visual_embeds.shape
        visual_attention_mask = torch.ones((batch_size, num_regions), dtype=torch.float).to(visual_embeds.device)
        outputs = self.visualbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
        )
        return outputs.last_hidden_state

class ClassificationHead(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(ClassificationHead, self).__init__()
        self.fc = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, cls_embedding):
        return self.sigmoid(self.fc(cls_embedding))

# CLIP Post-Classifier
# Define the base folder for images
IMAGE_FOLDER_PATH = r"K:\Thesis\Facebook Hateful Meme Dataset\data"

def clip_post_classifier(image_path, clip_model, clip_processor, prompts):
    """
    Generate similarity scores using CLIP between image and prompts.
    """
    # Prepend the base folder path
    full_image_path = os.path.join(IMAGE_FOLDER_PATH, image_path)

    # Ensure the path exists
    if not os.path.exists(full_image_path):
        raise FileNotFoundError(f"Image file not found: {full_image_path}")

    # Load the image
    image = Image.open(full_image_path).convert("RGB")
    inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        similarities = torch.nn.functional.cosine_similarity(outputs.image_embeds, outputs.text_embeds, dim=-1)
    return similarities.cpu().numpy()




# Evaluation Function
def evaluate_combined_model(
    model, classification_head, clip_model, clip_processor, val_loader, 
    threshold, weight_vbert, weight_clip
):
    print("Evaluating combined model...")
    all_predictions = []
    all_labels = []


    # Define prompts for CLIP
    prompts = [
        "a nonhateful image that is good",
        "a hateful image that contains racism, sexism, nationality, religion, or disability",
    ]

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing Batches")):
            input_ids, attention_mask, fused_embeddings, labels = batch
            
            # Flatten input_ids and attention_mask for VisualBERT
            batch_size, max_chunks, seq_len = input_ids.size()
            input_ids = input_ids.view(-1, seq_len).to(device)  # Flatten to [batch_size * max_chunks, seq_len]
            attention_mask = attention_mask.view(-1, seq_len).to(device)  # Flatten to match input_ids
            fused_embeddings = fused_embeddings.to(device)
            labels = labels.to(device)

            # VisualBERT predictions
            vbert_outputs = model(input_ids, attention_mask, fused_embeddings)
            cls_embedding = vbert_outputs[:, 0, :]
            vbert_logits = classification_head(cls_embedding).squeeze()

            # CLIP predictions
            clip_scores = []
            for i in range(len(labels)):
                image_path = val_dataset[batch_idx * val_loader.batch_size + i]["img"]
                clip_similarities = clip_post_classifier(image_path, clip_model, clip_processor, prompts)
                clip_scores.append(max(clip_similarities))  # Use max similarity score for binary decision

            clip_scores_tensor = torch.tensor(clip_scores).to(device)

            # Combine predictions
            combined_logits = (weight_vbert * vbert_logits) + (weight_clip * clip_scores_tensor)
            combined_predictions = (combined_logits >= threshold).int()

            all_predictions.extend(combined_predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print(f"Combined Model Metrics with threshold {threshold}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)



# Main
if __name__ == "__main__":
    # Paths and initialization
    val_dataset_path = r"K:\Thesis\MAIN\dev_with_clip_embeddings.jsonl"
    checkpoint_path = "best_hybrid_model.pth"

    print("Loading dataset...")
    dataset_loader = DatasetLoader(val_dataset_path)
    val_dataset = dataset_loader.load_dataset()

    preprocessor = Preprocessor()
    custom_dataset = CustomDataset(val_dataset, preprocessor)
    val_loader = DataLoader(custom_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

    # Initialize VisualBERT Hybrid model
    model = VisualBERTWithFusion().to(device)
    classification_head = ClassificationHead(hidden_dim=768).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    classification_head.load_state_dict(checkpoint["classification_head_state_dict"])
    model.eval()
    classification_head.eval()

    # Initialize CLIP model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Prompts for CLIP
    prompts = ["This image contains hate speech.", "This image does not contain hate speech."]

    # Evaluate
    # Run evaluation experiments with different thresholds
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        evaluate_combined_model(
            model=model,
            classification_head=classification_head,
            clip_model=clip_model,
            clip_processor=clip_processor,
            val_loader=val_loader,
            threshold=threshold,
            
            # This is where you can change the value from each of the models.
            weight_vbert=0.9, # VisualBERT
            weight_clip=0.1   # CLIP
        )

