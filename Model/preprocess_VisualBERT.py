import os
import json
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from transformers import BertTokenizer
from torchvision.ops import roi_align
from tqdm import tqdm


'''
    1.
    This code is the preprocessing step for VisualBERT model.
    It preprocess the image using Faster R-CNN.
    It preprocess the text using BERT.
    If the the text exceeds the 512 token limit of visualBERT, a sliding window mechanism is included.
        This turns the text into chunks in which can still be later preprocess. 
        
    Output: dev_preprocessed.jsonl & train_preprocessed.jsonl
    
'''

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
input_jsonl_path = r"K:\Thesis\Facebook Hateful Meme Dataset\data\dev.jsonl"  # Path to the existing JSONL file
image_folder_path = r"K:\Thesis\Facebook Hateful Meme Dataset\data"  # Path to the folder containing images
output_jsonl_path = r"K:\Thesis\MAIN\dev_preprocessed.jsonl"  # Path to save the updated JSONL file

# Initialize Models
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
faster_rcnn = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define Feature Extractor Class
class RegionFeatureExtractor:
    def __init__(self, model, device, top_k=5, feature_dim=2048):
        self.model = model
        self.device = device
        self.top_k = top_k
        self.feature_dim = feature_dim
        self.projection_layer = torch.nn.Linear(256, feature_dim).to(device)

    def extract(self, image_path):
        """Extracts region features from the image."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        boxes = outputs[0]["boxes"][:self.top_k]
        scores = outputs[0]["scores"][:self.top_k]

        backbone = self.model.backbone
        feature_maps = backbone(image_tensor)["0"]

        pooled_features = roi_align(
            input=feature_maps,
            boxes=[boxes],
            output_size=(7, 7),
            spatial_scale=1.0 / 16,
        )

        flattened_features = pooled_features.mean(dim=[-1, -2])
        projected_features = self.projection_layer(flattened_features)

        # Detach tensors before converting to numpy
        return projected_features.detach().cpu().numpy(), scores.detach().cpu().numpy()

# Initialize Feature Extractor
region_extractor = RegionFeatureExtractor(faster_rcnn, device)

# Sliding Window Tokenization
def tokenize_with_sliding_window(text, tokenizer, max_len=512, overlap=128):
    """Tokenizes text into overlapping chunks using a sliding window."""
    encoded_text = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
    input_ids = encoded_text["input_ids"].squeeze(0)
    attention_mask = encoded_text["attention_mask"].squeeze(0)

    chunks = []
    num_tokens = len(input_ids)

    # Sliding window logic
    start = 0
    while start < num_tokens:
        end = min(start + max_len, num_tokens)
        chunk_input_ids = input_ids[start:end]
        chunk_attention_mask = attention_mask[start:end]

        # Pad the chunk if it's shorter than max_len
        padding_length = max_len - len(chunk_input_ids)
        if padding_length > 0:
            chunk_input_ids = torch.cat([chunk_input_ids, torch.zeros(padding_length, dtype=torch.long)])
            chunk_attention_mask = torch.cat([chunk_attention_mask, torch.zeros(padding_length, dtype=torch.long)])

        chunks.append({"input_ids": chunk_input_ids, "attention_mask": chunk_attention_mask})

        # Move the start forward with overlap
        start += max_len - overlap

    return chunks

# Process Dataset
def process_dataset(input_path, image_folder, output_path, max_len=512, overlap=128):
    """Processes the dataset and saves the updated JSONL with a progress bar."""
    # Count total lines for progress bar initialization
    total_lines = sum(1 for _ in open(input_path, "r"))

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        # Initialize tqdm progress bar
        for line in tqdm(infile, total=total_lines, desc="Processing Dataset"):
            entry = json.loads(line)

            # Extract visual features
            image_path = os.path.join(image_folder, entry["img"].replace("/", os.sep))
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            try:
                visual_embeddings, _ = region_extractor.extract(image_path)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

            # Tokenize text with sliding window
            text = entry["text"]
            text_chunks = tokenize_with_sliding_window(text, tokenizer, max_len=max_len, overlap=overlap)

            # Add visual embeddings and tokenized chunks to the entry
            entry["visual_embeddings"] = visual_embeddings.tolist()
            entry["text_chunks"] = [
                {
                    "input_ids": chunk["input_ids"].tolist(),
                    "attention_mask": chunk["attention_mask"].tolist(),
                }
                for chunk in text_chunks
            ]

            # Write updated entry to the new JSONL
            outfile.write(json.dumps(entry) + "\n")

# Run the processing function
process_dataset(input_jsonl_path, image_folder_path, output_jsonl_path)

print("Processing complete. Updated dataset saved to:", output_jsonl_path)
