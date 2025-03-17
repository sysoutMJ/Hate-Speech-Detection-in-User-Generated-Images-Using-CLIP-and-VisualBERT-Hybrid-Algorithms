import torch
import os
from PIL import Image
import easyocr
from transformers import BertTokenizer, CLIPProcessor, CLIPModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torchvision.ops import roi_align
import cv2
import logging

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
faster_rcnn = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Region Feature Extractor for VisualBERT
class RegionFeatureExtractor:
    def __init__(self, model, device, top_k=5, feature_dim=2048):
        self.model = model
        self.device = device
        self.top_k = top_k
        self.feature_dim = feature_dim
        self.projection_layer = torch.nn.Linear(256, feature_dim).to(device)

    def extract(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        boxes = outputs[0]["boxes"][:self.top_k]
        scores = outputs[0]["scores"][:self.top_k]
        
        backbone = self.model.backbone
        feature_maps = backbone(image_tensor)["0"]
        pooled_features = roi_align(input=feature_maps, boxes=[boxes], output_size=(7, 7), spatial_scale=1.0 / 16)
        flattened_features = pooled_features.mean(dim=[-1, -2])
        projected_features = self.projection_layer(flattened_features)
        
        return projected_features.detach().cpu().numpy()

# Sliding Window Tokenization for BERT
class TextProcessor:
    def __init__(self, tokenizer, max_len=512, overlap=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap = overlap

    def tokenize_with_sliding_window(self, text):
        encoded_text = self.tokenizer(text, return_tensors="pt", truncation=False, padding=False)
        input_ids = encoded_text["input_ids"].squeeze(0)
        chunks = []
        start = 0
        while start < len(input_ids):
            end = min(start + self.max_len, len(input_ids))
            chunk = input_ids[start:end]
            if len(chunk) < self.max_len:
                padding = torch.zeros(self.max_len - len(chunk), dtype=torch.long)
                chunk = torch.cat([chunk, padding])
            chunks.append(chunk)
            start += self.max_len - self.overlap
        return chunks

# CLIP Feature Extractor
class CLIPFeatureExtractor:
    def __init__(self, clip_model, clip_processor, device):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device

    def extract_text_embedding(self, text):
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_text_features(**inputs)
        return outputs.squeeze(0).cpu().numpy()
    
    def extract_image_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
        return outputs.squeeze(0).cpu().numpy()

class InferencePreprocessor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.reader = easyocr.Reader(['en'])
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Load Faster R-CNN and move to correct device
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.faster_rcnn = fasterrcnn_resnet50_fpn(weights=weights).to(self.device).eval()

        # Correct projection: 2560 → 2048 (matches training)
        self.fusion_module = torch.nn.Linear(2560, 2048).to(self.device).eval()

    def preprocess(self, image_path, text=None):
        """Extract features from image and text to match training pipeline."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Extract text using OCR if no text is provided
        if not text:
            text = self.extract_text(image_path)

        # Tokenize text using BERT
        encoded_text = self.tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
        ).to(self.device)

        # Extract image features using Faster R-CNN
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.faster_rcnn(image_tensor)

        # Use all available region proposals and ensure 2048-dim features
        feature_maps = self.faster_rcnn.backbone(image_tensor)["0"]
        pooled_features = roi_align(
            input=feature_maps, boxes=[outputs[0]["boxes"][:5]], output_size=(7, 7), spatial_scale=1.0 / 16
        )
        visual_embeddings = pooled_features.mean(dim=[-1, -2]).to(self.device)

        # Ensure visual embeddings are always (2048,)
        if visual_embeddings.dim() == 2:
            visual_embeddings = visual_embeddings.mean(dim=0)  # Convert to (2048,)
        
        if visual_embeddings.shape[0] != 2048:
            padding_needed = 2048 - visual_embeddings.shape[0]
            visual_embeddings = torch.nn.functional.pad(visual_embeddings, (0, padding_needed))

        # Extract CLIP text and image embeddings
        clip_inputs = self.clip_processor(text=[text], images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            clip_outputs = self.clip_model(**clip_inputs)
        
        clip_text_embedding = clip_outputs.text_embeds.squeeze().to(self.device)
        # Remove extra image embedding (we only uses one)
        # clip_image_embedding = clip_outputs.image_embeds.squeeze().to(self.device)

        # Ensure embeddings match expected sizes
        print("Final Visual Embeddings Shape:", visual_embeddings.shape)  # (2048,)
        print("CLIP Text Embedding Shape:", clip_text_embedding.shape)  # (512,)

        # Concatenate embeddings correctly: (2048 + 512 = 2560)
        fused_embeddings = torch.cat([visual_embeddings, clip_text_embedding], dim=-1)

        # Project to 2048 to match training
        fused_embeddings = self.fusion_module(fused_embeddings).unsqueeze(0)  # (1, 2048)

        return encoded_text["input_ids"], encoded_text["attention_mask"], fused_embeddings


    # Threshold of OCR is can be modified.
    def extract_text(self, image_path, confidence_threshold=0.1):
        try:
            # Preprocess image for better OCR accuracy
            # img = cv2.imread(image_path)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # enhanced = cv2.equalizeHist(gray)  # ✅ Enhancing contrast
            # cv2.imwrite("processed_image.png", enhanced)
            
                    # Load image and convert to grayscale
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding for better OCR contrast
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Extract text using EasyOCR
            result = self.reader.readtext(image_path)

            # Lower threshold to keep more text
            # extracted_texts = [res[1] for res in result if res[2] > confidence_threshold]
            text = " ".join([res[1] for res in result if res[2] > confidence_threshold])

            # Debug: Print detected OCR results cleanly
            print(f"\n📝 OCR Raw Output for {image_path}:")
            extracted_texts = []
            for res in result:
                bbox, detected_text, confidence = res[0], res[1], res[2]
                print(f"   - Text: \"{detected_text}\" | Confidence: {confidence:.2f} | BBox: {bbox}")

                if confidence > confidence_threshold:
                    extracted_texts.append(detected_text)
            
            # Join extracted text and clean minor OCR mistakes
            text = " ".join(extracted_texts)
            text = text.replace("\n", " ").strip()  # Remove unnecessary line breaks
            
            # Adjust text length for VisualBERT
            max_length = 128  
            if len(text) > max_length:
                text = text[:max_length]

            # Ensure fallback text if extraction fails
            final_text = text if text.strip() else "No text found"
            print(f"🔍 Extracted Text (After Filtering & Processing): \"{final_text}\"")

            return final_text
        except Exception as e:
            logging.error(f"Error extracting text from image {image_path}: {e}")
            return "No text found"



