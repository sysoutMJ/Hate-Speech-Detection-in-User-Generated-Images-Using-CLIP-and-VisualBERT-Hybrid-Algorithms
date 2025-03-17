import os
import torch
from transformers import VisualBertModel
import torch.nn.functional as F
from inference_preprocessing import InferencePreprocessor  # Import the preprocessing class

# Model Classes
class FusionModule(torch.nn.Module):
    def __init__(self, input_dim=2560, output_dim=2048):
        super(FusionModule, self).__init__()
        self.projection_layer = torch.nn.Linear(input_dim, output_dim)

    def forward(self, fused_embeddings):
        return self.projection_layer(fused_embeddings)

class VisualBERTWithFusion(torch.nn.Module):
    def __init__(self, model_name="uclanlp/visualbert-vqa-coco-pre", fused_dim=2560):
        super(VisualBERTWithFusion, self).__init__()
        self.visualbert = VisualBertModel.from_pretrained(model_name)
        self.fusion_module = FusionModule(input_dim=fused_dim, output_dim=2048)

    def forward(self, input_ids, attention_mask, fused_embeddings):
        visual_attention_mask = torch.ones((input_ids.shape[0], 1), dtype=torch.float).to(fused_embeddings.device)
        outputs = self.visualbert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            visual_embeds=fused_embeddings, 
            visual_attention_mask=visual_attention_mask
        )
        return outputs.last_hidden_state

class ClassificationHead(torch.nn.Module):
    def __init__(self, hidden_dim=768):
        super(ClassificationHead, self).__init__()
        self.fc = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, cls_embedding):
        return self.sigmoid(self.fc(cls_embedding))

class HateSpeechDetector:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.preprocessor = InferencePreprocessor(device)  # Ensure preprocessor is on same device

        # Load model on correct device
        self.model = VisualBERTWithFusion().to(self.device)
        self.classification_head = ClassificationHead().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.classification_head.load_state_dict(checkpoint["classification_head_state_dict"])

        self.model.eval()
        self.classification_head.eval()

    def predict(self, image_path, text=None, threshold=0.6):
        """Predict whether the image contains hate speech."""
        
        # Extract text first
        if text is None:
            text = self.preprocessor.extract_text_from_image(image_path)
        
        # Preprocess the inputs
        input_ids, attention_mask, fused_embeddings = self.preprocessor.preprocess(image_path, text)

        # Move tensors to the correct device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        fused_embeddings = fused_embeddings.to(self.device).unsqueeze(1)  # Ensure correct shape

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, fused_embeddings)
            cls_embedding = outputs[:, 0, :]
            prediction = self.classification_head(cls_embedding).squeeze().item()

        is_hate_speech = prediction > threshold
        print(f"📊 Prediction for {image_path}: {prediction} (Threshold: {threshold}) → {'Hate Speech' if is_hate_speech else 'Not Hate Speech'}")

        return prediction

