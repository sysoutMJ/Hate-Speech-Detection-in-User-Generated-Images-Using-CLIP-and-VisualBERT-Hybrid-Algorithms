import easyocr
import os
import sys

"""
    ℹ️ [MAIN FILE]
"""


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class OCRExtractor:
    def __init__(self):
        """
        Initialize the OCRExtractor class with the required model directories and settings.
        """
        self.reader = easyocr.Reader(
            ["en"],  # language
            gpu=True,
            detector=True,
            recognizer=True,
            model_storage_directory=resource_path("models"),
            user_network_directory=resource_path("models"),
            detect_network="craft",
            recog_network="english_g2",
            download_enabled=False,
        )

    def extract_text(self, image_path):
        """
        Extract text from an image using EasyOCR.
        """
        results = self.reader.readtext(image_path)
        extracted_texts = []

        for bbox, text, prob in results:
            # print(f"Text: {text} | Confidence: {prob:.2f}")
            # print(f"Bounding Box: {bbox}")
            # print("-" * 40)
            extracted_texts.append(text)

        combined_text = " ".join(extracted_texts)
        # print("\nCombined Text:")
        # print(combined_text)
        return combined_text
