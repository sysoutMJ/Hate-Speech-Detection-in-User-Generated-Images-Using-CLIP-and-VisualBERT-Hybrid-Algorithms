from PIL import Image
import torch
import torch.nn.functional as F
from transformers import (
    CLIPModel,
    CLIPProcessor,
)
from model import TrueHybridHateDetector
import warnings
from dataset_dataloader import Config
import os
import sys

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Using a slow image processor.*")


"""
    ℹ️ [MAIN FILE]
"""


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = resource_path("models\lr_2e-5_1.1_0506_best_model.pth")
# print(MODEL_PATH, os.path.exists(MODEL_PATH))


def use_CLIP_only(image_path, device=DEVICE):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")

    # Load CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    hatespeech_prompts = [
        "Image promoting hate, racism, or violence",
        "Image with racist, sexist, or anti-immigrant language",
        "Image showing symbols or signs linked to hate groups",
        "Image with slurs or threats targeting people",
        "Image attacking a specific group",
        "Image that encourages hate or exclusion",
        "Image with sarcastic or hidden hateful meaning",
        "Image with protest text using dehumanizing language",
        "Image supporting white supremacy or extreme views",
        "Image encouraging violence or unfair treatment",
        "Image with offensive jokes that insult a group",
        "Image with coded words spreading hate or conspiracy",
    ]

    notHatespeech_prompts = [
        "Image speaking out against racism or injustice",
        "Image with strong words but not hateful",
        "Image with assumption about group but not hateful",
        "Image with re-used slurs used in a non-hateful way",
        "Image with opinions on culture shared calmly",
        "Image about social causes or movements",
        "Image showing harmless or personal content",
        "Image showing nature or normal everyday things",
        "Image with educational or informative text",
        "Image with creative or funny content that isn’t hateful",
        "Image raising awareness about issues respectfully",
        "Image with a positive or uplifting quote",
        "Image with a stereotype or belief shared without bad intent",
    ]

    # Combine prompts for processing
    prompts = hatespeech_prompts + notHatespeech_prompts

    # Preprocess inputs
    inputs = clip_processor(
        text=prompts, images=image, return_tensors="pt", padding=True
    ).to(device)

    # Get CLIP scores
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: [1, num_prompts]
        probs = F.softmax(logits_per_image, dim=1)  # normalized probabilities

        # Create score lists for each category
        hate_scores = []
        not_hate_scores = []

        # Populate score lists correctly
        for i, prompt in enumerate(prompts):
            score = probs[0][i].item()
            if prompt in hatespeech_prompts:
                hate_scores.append((prompt, score))
            else:
                not_hate_scores.append((prompt, score))

        # Find top prompts in each category
        top_hate_prompt, top_hate_score = max(hate_scores, key=lambda x: x[1])
        top_not_hate_prompt, top_not_hate_score = max(
            not_hate_scores, key=lambda x: x[1]
        )

        # Determine overall prediction based on highest score
        if top_hate_score > top_not_hate_score:
            is_hatespeech = True
            confidence = top_hate_score
        else:
            is_hatespeech = False
            confidence = top_not_hate_score

        # Always select the interpretation from the appropriate list
        if is_hatespeech:
            top_prompt = top_hate_prompt
        else:
            top_prompt = top_not_hate_prompt

        return {
            "prediction": int(is_hatespeech),
            "confidence": confidence,
            "top_prompt": top_prompt,
            "top_hate_prompt": top_hate_prompt,
            "top_hate_score": top_hate_score,
            "top_not_hate_prompt": top_not_hate_prompt,
            "top_not_hate_score": top_not_hate_score,
        }


def predict_single_input(model_path, config, image_path, text):
    # Initialize model
    model = TrueHybridHateDetector(config).to(config.device)

    # Load saved weights
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=DEVICE)
    )
    model.eval()

    # Process and predict
    with torch.no_grad():
        logits = model([text], [image_path])
        probability = torch.sigmoid(logits).item()
        prediction = 1 if probability >= config.desired_threshold else 0

    return {
        "probability": probability,
        "prediction": prediction,
    }


class clip_and_hybrid_model:
    def predict_with_adaptive_fusion(self, image_path, text):
        config = Config()
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.desired_threshold = 0.3

        model_path = MODEL_PATH

        # Load CLIP model results
        clip_result = use_CLIP_only(image_path)
        clip_pred = clip_result["prediction"]
        clip_conf = clip_result["confidence"]

        # Load Hybrid model results
        hybrid_result = predict_single_input(
            model_path=model_path,
            config=config,
            image_path=image_path,
            text=text,
        )
        hybrid_pred = hybrid_result["prediction"]
        hybrid_conf = hybrid_result["probability"]

        image_weight = 0.8
        text_weight = 0.2

        # Determine final prediction and confidence
        if hybrid_pred == clip_pred:
            final_pred = clip_pred
            final_confidence = max(clip_conf, hybrid_conf)
        else:
            weighted_score = (clip_conf * image_weight) + (hybrid_conf * text_weight)
            final_confidence = weighted_score
            final_pred = 1 if final_confidence >= 0.3 else 0

        # Select interpretation based on final prediction
        if final_pred == 1:  # Hate speech
            interpretation = clip_result["top_hate_prompt"]
        else:  # Not hate speech
            interpretation = clip_result["top_not_hate_prompt"]

        return final_confidence, interpretation
