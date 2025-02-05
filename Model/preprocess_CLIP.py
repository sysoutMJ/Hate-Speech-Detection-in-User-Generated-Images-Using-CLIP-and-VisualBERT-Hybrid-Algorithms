import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

'''
    2.
    The preprocessing program for CLIP.
    
    Output: train_with_clip_embeddings.jsonl & dev_with_clip_embeddings.jsonl
'''
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
input_jsonl_path = r"K:\Thesis\MAIN\dev_preprocessed.jsonl"  # Input JSONL
output_jsonl_path = r"K:\Thesis\MAIN\dev_with_clip_embeddings.jsonl"  # Output JSONL
image_folder_path = r"K:\Thesis\Facebook Hateful Meme Dataset\data"  # Image folder path

# Initialize CLIP Model and Processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Class for handling text processing with sliding window
class TextProcessorWithSlidingWindow:
    def __init__(self, clip_processor, max_len=77, overlap=27):
        self.tokenizer = clip_processor.tokenizer
        self.max_len = max_len
        self.overlap = overlap

    def tokenize_with_sliding_window(self, text):
        """Split text into chunks using a sliding window."""
        tokenized = self.tokenizer(text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
        chunks = []

        start = 0
        while start < len(tokenized):
            end = min(start + self.max_len, len(tokenized))
            chunk = tokenized[start:end]

            # Pad if necessary
            if len(chunk) < self.max_len:
                padding = torch.zeros(self.max_len - len(chunk), dtype=torch.long)
                chunk = torch.cat([chunk, padding])

            chunks.append(chunk)
            start += self.max_len - self.overlap  # Slide the window forward

        return chunks

    def get_text_embedding(self, text):
        """Process text with sliding window and aggregate embeddings."""
        chunks = self.tokenize_with_sliding_window(text)
        chunk_embeddings = []

        for chunk in chunks:
            inputs = {"input_ids": chunk.unsqueeze(0).to(device)}
            with torch.no_grad():
                outputs = clip_model.get_text_features(**inputs)
                chunk_embeddings.append(outputs)

        # Aggregate chunk embeddings (e.g., average pooling)
        aggregated_embedding = torch.mean(torch.stack(chunk_embeddings, dim=0), dim=0)
        return aggregated_embedding


# Class for handling image processing
class ImageProcessor:
    def __init__(self, clip_processor):
        self.processor = clip_processor

    def get_image_embedding(self, image_path):
        """Process image and extract embeddings."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)

        return outputs.squeeze(0)


# Class for processing and updating the dataset
class DatasetProcessor:
    def __init__(self, image_processor, text_processor):
        self.image_processor = image_processor
        self.text_processor = text_processor

    def process_entry(self, entry, image_folder):
        """Process a single dataset entry."""
        # Extract image path and text
        image_path = os.path.join(image_folder, entry["img"].replace("/", os.sep))
        text = entry["text"]

        # Process image
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None

        try:
            image_embedding = self.image_processor.get_image_embedding(image_path).cpu().numpy()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

        # Process text
        try:
            text_embedding = self.text_processor.get_text_embedding(text).cpu().numpy()
        except Exception as e:
            print(f"Error processing text: {e}")
            return None

        # Append embeddings
        entry["clip_image_embeddings"] = image_embedding.tolist()
        entry["clip_text_embeddings"] = text_embedding.tolist()

        return entry

    def process_dataset(self, input_path, image_folder, output_path):
        """Process the entire dataset."""
        total_lines = sum(1 for _ in open(input_path, "r"))

        with open(input_path, "r") as infile, open(output_path, "w") as outfile:
            with tqdm(total=total_lines, desc="Processing Dataset") as pbar:
                for line in infile:
                    entry = json.loads(line)

                    updated_entry = self.process_entry(entry, image_folder)
                    if updated_entry:
                        outfile.write(json.dumps(updated_entry) + "\n")

                    pbar.update(1)


# Initialize processors
image_processor = ImageProcessor(clip_processor)
text_processor = TextProcessorWithSlidingWindow(clip_processor)
dataset_processor = DatasetProcessor(image_processor, text_processor)

# Run the dataset processing
dataset_processor.process_dataset(input_jsonl_path, image_folder_path, output_jsonl_path)

print("Processing complete. Updated dataset saved to:", output_jsonl_path)
