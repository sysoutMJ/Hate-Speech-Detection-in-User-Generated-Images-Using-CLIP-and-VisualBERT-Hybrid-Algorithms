import easyocr


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
            model_storage_directory=r"C:\Users\ACER\Desktop\Thesis\git_Thesis\EASY_OCR\my_model\lang_char",
            user_network_directory=r"C:\Users\ACER\Desktop\Thesis\git_Thesis\EASY_OCR\my_model\lang_char",
            detect_network="craft",  # <- pass network name, not .pth path
            recog_network="english_g2",  # <- same: pass the model name, not the .pth path
            download_enabled=False,
        )

    def extract_text(self, image_path):
        """
        Extract text from an image using EasyOCR.
        """
        results = self.reader.readtext(image_path)
        extracted_texts = []

        for bbox, text, prob in results:
            print(f"Text: {text} | Confidence: {prob:.2f}")
            print(f"Bounding Box: {bbox}")
            print("-" * 40)
            extracted_texts.append(text)

        combined_text = " ".join(extracted_texts)
        # print("\nCombined Text:")
        # print(combined_text)
        return combined_text


# reader = easyocr.Reader(
#     ["en"],  # language
#     gpu=True,
#     detector=True,
#     recognizer=True,
#     model_storage_directory=r"K:\Official_Thesis\EASY_OCR\my_model\lang_char",
#     user_network_directory=r"K:\Official_Thesis\EASY_OCR\my_model\lang_char",
#     detect_network="craft",  # <- pass network name, not .pth path
#     recog_network="english_g2",  # <- same: pass the model name, not the .pth path
#     download_enabled=False,
# )

# # image_path = (
# #     r"C:\Users\genes\Pictures\Tagalog Memes\c40a63639f45c9ae9278261b2b64f4e8.jpg"
# # )
# # results = reader.readtext(image_path)

# # print("OCR Results:")
# # for bbox, text, prob in results:
# #     print(f"Text: {text} | Confidence: {prob:.2f}")
# #     print(f"Bounding Box: {bbox}")
# #     print("-" * 40)


# def extract_text_using_easyocr(image_path):
#     results = reader.readtext(image_path)
#     extracted_texts = []

#     for bbox, text, prob in results:
#         print(f"Text: {text} | Confidence: {prob:.2f}")
#         print(f"Bounding Box: {bbox}")
#         print("-" * 40)
#         extracted_texts.append(text)

#     combined_text = " ".join(extracted_texts)
#     print("\nCombined Text:")
#     print(combined_text)
#     return combined_text


# if __name__ == "__main__":
#     image_path = (
#         r"C:\Users\genes\Pictures\Tagalog Memes\c40a63639f45c9ae9278261b2b64f4e8.jpg"
#     )

#     extract_text_using_easyocr(image_path)
