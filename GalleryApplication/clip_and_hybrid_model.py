from PIL import Image
import torch
import torch.nn.functional as F
from transformers import (
    BertTokenizer,
    CLIPModel,
    CLIPProcessor,
)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from model import TrueHybridHateDetector
import warnings
from dataset_dataloader import Config

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Using a slow image processor.*")

DEVICE = torch.device("cpu")


class TextProcessor:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            DEVICE
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            "google-bert/bert-base-cased", use_fast=True
        )

        self.prompts = [
            "hate speech about animal",
            "hate speech about society",
            "hate speech about individual",
            "contains about animal but no hate speech",
            "contains about society but no hate speech",
            "contains about individual but no hate speech",
        ]

    def process_text(self, text, max_length=512):
        # Tokenize text for VisualBERT
        inputs = self.tokenizer(
            text, truncation=True, max_length=max_length, return_tensors="pt"
        )

        # If text is truncated, use CLIP to match prompts
        if len(inputs["input_ids"][0]) == max_length:
            print("Text truncated. Falling back to CLIP prompt matching.")
            text_embed = self.clip_model.get_text_features(
                **self.tokenizer(text, return_tensors="pt")
            )
            prompt_embeds = [
                self.clip_model.get_text_features(
                    **self.tokenizer(p, return_tensors="pt")
                )
                for p in self.prompts
            ]

            # Compute cosine similarities
            similarities = [
                cosine_similarity(text_embed.cpu().numpy(), pe.cpu().numpy())[0][0]
                for pe in prompt_embeds
            ]
            best_prompt = self.prompts[np.argmax(similarities)]
            print(
                f"Replaced with prompt: '{best_prompt}' (similarity={max(similarities):.2f})"
            )

            # Retokenize the best prompt
            inputs = self.tokenizer(best_prompt, return_tensors="pt")

        return inputs


def use_CLIP_only(image_path, device=DEVICE):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")

    # Load CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Define prompts
    # hatespeech_prompts = [
    #     "hate speech about animal",
    #     "hate speech about society",
    #     "hate speech about individual",
    #     "hate speech about religion",
    # ]
    # notHatespeech_prompts = [
    #     "contains about animal but no hate speech",
    #     "contains about society but no hate speech",
    #     "contains about individual but no hate speech",
    #     "contains about religion but no hate speech",
    # ]

    hatespeech_prompts = [
        "An image that promotes hate, racism, or violence",
        "A meme with racist, xenophobic, or sexist language",
        "A symbol or sign associated with hate speech",
        "A post spreading hate through slurs or threats",
        "Content that attacks a specific group of people",
        "Text and image encouraging hostility or exclusion",
        "A sarcastic meme with hidden hateful meaning",
        "A protest sign with dehumanizing language",
        "An image supporting white supremacy or extremist views",
        "A post that incites violence or discrimination",
        "An offensive joke with hateful undertones",
        "A coded image spreading conspiracy or hate ideology",
    ]

    notHatespeech_prompts = [
        "A protest image promoting human rights or equality",
        "Satirical or sarcastic post not intended as hate",
        "A meme critiquing racism or injustice",
        "An educational post about hate speech",
        "A message of unity against discrimination",
        "Strong language but not hate speech",
        "Reclaimed or reappropriated slurs used without hate",
        "Cultural commentary with no hostile intent",
        "A social justice slogan or movement post",
        "A peaceful protest image without offensive content",
        "A parody that criticizes hate speech without using it",
    ]

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

        # Pair each prompt with its probability
        scores = list(zip(prompts, probs[0].tolist()))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # Print top scores for debug
        # print("Top CLIP Prompt Scores:")
        # for prompt, score in sorted_scores:
        #     print(f"{score:.4f} -> {prompt}")

        # Get the best prompt
        best_prompt, best_score = sorted_scores[0]
        # Output: True or False
        is_hatespeech = best_prompt in hatespeech_prompts

        # print(is_hatespeech)
        # print(f"\nPrediction: {'HATE SPEECH' if is_hatespeech else 'NOT HATE SPEECH'}")
        # print(f"Confidence Score: {best_score:.4f}")

        return {
            "prediction": int(is_hatespeech),
            "confidence": best_score,
            "top_prompt": best_prompt,
        }


def predict_single_input(model_path, config, image_path, text):
    """
    Predict hate speech classification for a single image-text pair using a saved model.

    Args:
        model_path: Path to the saved .pth model file
        config: Configuration object with device and threshold settings
        image_path: Path to the input image file
        text: Input text string

    Returns:
        Dictionary containing:
        - probability: Prediction probability (0-1)
        - prediction: Binary prediction (0 or 1)
        - class_label: "HATEFUL" or "NOT HATEFUL"
    """
    # Initialize model
    model = TrueHybridHateDetector(config).to(config.device)

    # Load saved weights
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=config.device))
    model.eval()

    # Process and predict
    with torch.no_grad():
        logits = model([text], [image_path])
        probability = torch.sigmoid(logits).item()
        prediction = 1 if probability >= config.desired_threshold else 0

    return {
        "probability": probability,
        "prediction": prediction,
        "class_label": "HATE SPEECH" if prediction else "NOT HATE SPEECH",
        "threshold": config.desired_threshold,
        "raw_score": logits.item(),
    }


class clip_and_hybrid_model:
    def predict_with_adaptive_fusion(self, image_path, text):
        config = Config()
        config.device = torch.device("cpu")
        config.desired_threshold = 0.5

        model_path = r"C:\Users\ACER\Desktop\Thesis\lr_2e-5_1.1_0506_best_model.pth"

        # Load CLIP model results
        result = use_CLIP_only(image_path)
        clip_pred = result["prediction"]
        clip_conf = result["confidence"]

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

        if hybrid_pred == clip_pred:
            final_confidence = max(clip_conf, hybrid_conf)
        else:
            weighted_score = (clip_conf * image_weight) + (hybrid_conf * text_weight)
            final_confidence = weighted_score

        return final_confidence


# class clip_and_hybrid_model:
#     def predict_with_adaptive_fusion(self, image_path, text):
#         config = Config()
#         config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         config.desired_threshold = 0.5

#         model_path = r"K:\0505\lr_2e-5_1.1_0506_best_model.pth"

#         # Load CLIP model results
#         result = use_CLIP_only(image_path)
#         clip_pred = result["prediction"]
#         clip_conf = result["confidence"]

#         # Load Hybrid model results
#         hybrid_result = predict_single_input(
#             model_path=model_path,
#             config=config,
#             image_path=image_path,
#             text=text,
#         )
#         hybrid_pred = hybrid_result["prediction"]
#         hybrid_conf = hybrid_result["probability"]

#         image_weight = 0.8
#         text_weight = 0.2

#         # Weighted decision logic: Use both the prediction and the confidence score
#         if hybrid_pred == clip_pred:
#             # If both models agree, use the hybrid prediction
#             # final_prediction = hybrid_pred
#             final_confidence = max(clip_conf, hybrid_conf)
#         else:
#             # If the predictions are different, rely on weighted confidence
#             weighted_score = (clip_conf * image_weight) + (hybrid_conf * text_weight)
#             # Final confidence is based on the weighted score
#             final_confidence = weighted_score

#         # # Decision Logic — Final Confidence only
#         # if hybrid_pred == clip_pred:
#         #     final_confidence = max(clip_conf, hybrid_conf)
#         # else:
#         #     if hybrid_pred and not clip_pred:
#         #         final_confidence = hybrid_conf
#         #     elif clip_pred and not hybrid_pred:
#         #         final_confidence = clip_conf
#         #     else:
#         #         final_confidence = max(1 - clip_conf, 1 - hybrid_conf)

#         return final_confidence

# def predict_with_adaptive_fusion(self, image_path, text):
#     config = Config()
#     config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     config.desired_threshold = 0.5

#     model_path = r"K:\0505\lr_2e-5_1.1_0506_best_model.pth"
#     # Load CLIP model results
#     result = use_CLIP_only(image_path)
#     clip_pred = result["prediction"]
#     clip_conf = result["confidence"]

#     # Load Hybrid model results
#     hybrid_result = predict_single_input(
#         model_path=model_path,
#         config=config,
#         image_path=image_path,
#         text=text,
#     )
#     hybrid_pred = hybrid_result["prediction"]
#     hybrid_conf = hybrid_result["probability"]

#     # Decision Logic
#     if hybrid_pred == clip_pred:
#         final_prediction = hybrid_pred
#         final_confidence = max(clip_conf, hybrid_conf)
#     else:
#         if hybrid_pred and not clip_pred:
#             final_prediction = 1
#             final_confidence = hybrid_conf
#         elif clip_pred and not hybrid_pred:
#             final_prediction = 1
#             final_confidence = clip_conf
#         else:
#             final_prediction = 0
#             final_confidence = max(1 - clip_conf, 1 - hybrid_conf)

#     return final_prediction, final_confidence


# if __name__ == "__main__":
#     image_path = r"C:\Users\genes\Pictures\Andrea Lituania\494839327_1425063448655483_8865251265319096901_n.jpg"
#     text = ""
#     model_path = r"K:\0505\lr_2e-5_1.1_0506_best_model.pth"

#     # Load CLIP model results
#     result = use_CLIP_only(image_path)
#     clip_pred = result["prediction"]
#     clip_conf = result["confidence"]
#     clip_prompt = result["top_prompt"]

#     # Load Hybrid model results
#     from model import TrueHybridHateDetector
#     from dataset_dataloader import Config

#     config = Config()
#     config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     config.desired_threshold = 0.5

#     final_prediction, final_confidence = predict_with_adaptive_fusion(
#         image_path=image_path, text=text
#     )

#     print(
#         f"Final Prediction: {'HATE SPEECH' if final_prediction >= 0.5 else 'NOT HATE SPEECH'}"
#     )
#     print(f"Final Confidence Score: {final_confidence:.4f}")

# hybrid_result = predict_single_input(
#     model_path=model_path,
#     config=config,
#     image_path=image_path,
#     text=text,
# )
# hybrid_pred = hybrid_result["prediction"]
# hybrid_conf = hybrid_result["probability"]

# # Print raw model outputs
# print(f"\nText: {text}")
# print(f"Image: {image_path}")
# print("\n=== CLIP Prediction Results ===")
# print(f"Prediction: {'HATE SPEECH' if clip_pred else 'NOT HATE SPEECH'}")
# print(f"Confidence Score: {clip_conf:.4f}")
# print(f"Top CLIP Prompt: {clip_prompt}")
# print("-" * 60)
# print("\n=== Hybrid Model Prediction Results ===")
# print(f"Prediction: {hybrid_result['class_label']}")
# print(f"Confidence: {hybrid_conf:.4f}")
# print("-" * 60)

# # ------------------------------
# # DECISION LOGIC + INTERPRETATION
# # ------------------------------
# print("\n=== Final Decision Summary ===")

# def explain_disagreement():
#     if hybrid_pred and not clip_pred:
#         print("→ Likely *textual hate speech* missed by CLIP.")
#         return "Hybrid (TEXT) model flagged hate speech."
#     elif clip_pred and not hybrid_pred:
#         if len(text.strip()) < 10:
#             print("→ Possibly *visual hate speech* with no meaningful text.")
#         else:
#             print("→ CLIP sees visual signals missed by Hybrid.")
#         return "CLIP (IMAGE) model flagged hate speech."
#     else:
#         return "Models agree."

# if hybrid_pred == clip_pred:
#     final_prediction = hybrid_pred
#     print("✅ Models agree.")
# else:
#     # Scenario-based trust
#     if hybrid_pred and not clip_pred:
#         final_prediction = 1
#     elif clip_pred and not hybrid_pred:
#         final_prediction = 1
#     else:
#         final_prediction = 0  # default fallback

# explanation = explain_disagreement()

# print(
#     f"\n📌 Final Prediction: {'HATE SPEECH' if final_prediction else 'NOT HATE SPEECH'}"
# )
# print(f"🧠 Reason: {explanation}")
# print("=" * 60)


# if __name__ == "__main__":
#     image_path = r"C:\Users\genes\Pictures\Andrea Lituania\2f5a6bad-2bb4-4039-b1f2-3fa149ee1491.jpeg"
#     text = "black lives matter"
#     model_path = r"K:\0505\lr_2e-5_1.1_0506_best_model.pth"

#     result = use_CLIP_only(image_path)

#     print(f"\nText: {text}")
#     print(f"Image: {image_path}")
#     print("\n=== CLIP Prediction Results ===")
#     # result['prediction'] is true or false
#     print(f"Prediction: {'HATE SPEECH' if result['prediction'] else 'NOT HATE SPEECH'}")
#     print(f"Confidence Score: {result['confidence']:.4f}")
#     print(f"Top CLIP Prompt: {result['top_prompt']}\n")
#     print("-" * 60)

#     from model import TrueHybridHateDetector
#     from dataset_dataloader import Config

#     config = Config()
#     config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     config.desired_threshold = 0.5  # Or use your optimal threshold

#     hybrid_result = predict_single_input(
#         model_path=model_path,
#         config=config,
#         image_path=image_path,
#         text=text,
#     )

#     # Print detailed prediction results
#     print("\n=== Hybrid Model Prediction Results ===")
#     print(f"Prediction: {hybrid_result['class_label']}")
#     print(f"Confidence: {hybrid_result['probability']:.4f}")
#     print(f"Threshold: {config.desired_threshold:.4f}")
