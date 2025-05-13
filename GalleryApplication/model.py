import torch
import torch.nn as nn
from transformers import VisualBertModel, BertTokenizer, CLIPModel, CLIPProcessor
from PIL import Image


"""
    ℹ️ [MAIN FILE]
    
    This is model.py
    
    This contains the main architecture of the hybrid model.
    The late fusion in this python is to be experimented later on for late fusion. Not yet added.
    This focuses on the hybrid model itself.
    The purpose of late fusion is to boost the performance of the hybrid model to get the good metrics results.
    
    A multimodal hate speech detector combining VisualBERT and CLIP with late fusion.

    Pipeline:
    1. Text → VisualBERT tokenizer
    2. Image → CLIP encoder
    3. VisualBERT fuses text + CLIP features (early fusion)

"""


class TrueHybridHateDetector(nn.Module):
    def __init__(self, config):
        super(TrueHybridHateDetector, self).__init__()
        self.config = config
        self.device = config.device

        # Initialize CLIP (image encoder)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # Initialize VisualBERT (text + fusion)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.visualbert = VisualBertModel.from_pretrained(
            "uclanlp/visualbert-vqa-coco-pre"
        ).to(self.device)

        self.visualbert.embeddings.visual_projection = nn.Linear(512, 768).to(
            self.device
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        ).to(self.device)

    def encode_image(self, image_path):
        """Extract CLIP visual features from an image.

        Args:
            image_path: Path to input image
        Returns:
            torch.Tensor: CLIP features [1, 512]
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        return self.clip_model.get_image_features(**inputs).unsqueeze(
            1
        )  # Shape: [1, 1, 512]

    def forward(self, texts, image_paths):
        """Process batch of texts and image paths."""
        # Step 1: Batch-process images with CLIP
        clip_features = []
        for path in image_paths:
            features = self.encode_image(path)
            clip_features.append(features)
        clip_features = torch.cat(clip_features, dim=0)  # [batch_size, 1, 512]

        # Step 2: Batch-process texts
        text_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        # Step 3: Multimodal fusion
        outputs = self.visualbert(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            visual_embeds=clip_features,
            visual_attention_mask=torch.ones(clip_features.shape[:2]).to(self.device),
        )

        # Step 4: Classify
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled).squeeze()  # [batch_size]

    def predict_batch(self, batch):
        """Batch prediction for DataLoader outputs."""
        with torch.no_grad():
            logits = self.forward(batch["text"], batch["image_paths"])
            probabilities = torch.sigmoid(logits)
        return {
            "probabilities": probabilities.cpu().numpy(),
            "predictions": (probabilities >= self.config.desired_threshold)
            .int()
            .cpu()
            .numpy(),
        }


# if __name__ == "__main__":

#     class Config:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         desired_threshold = 0.5

#     # Initialize the model
#     model = TrueHybridHateDetector(Config())

#     # Test sample (replace with your actual image/text)
#     test_text = "my body my choice abortion"
#     test_image_path = r"K:\thesisDataset\facebook_and_momenta\01382.png"  # Must match your JSONL 'id'.png format

#     # Run prediction
#     with torch.no_grad():
#         result = model.predict(test_text, test_image_path)

#     # Print detailed results
#     print("\nTest Prediction Results:")
#     print(f"Text: {test_text}")
#     print(f"Image: {test_image_path}")
#     print("-" * 50)
#     print(f"Logits: {result['logits']}")
#     print(f"Final score: {result['probability']:.4f}")
#     print(f"Prediction: {'HATEFUL' if result['prediction'] else 'NOT HATEFUL'}")
