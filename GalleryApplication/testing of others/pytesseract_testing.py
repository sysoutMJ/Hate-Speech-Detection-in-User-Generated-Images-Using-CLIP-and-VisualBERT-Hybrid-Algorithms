import os
import pytesseract
import cv2
import numpy as np
from PIL import Image

# Path to the Tesseract executable (configure as needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\genes\anaconda3\pkgs\tesseract-5.3.1-hcb5f61f_0\Library\bin\tesseract.exe"


def preprocess_image(image_path):
    """
    Apply advanced preprocessing techniques to improve OCR accuracy

    Args:
        image_path (str): Path to the input image

    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Optional: Deskew the image
    def deskew(image):
        """Deskew the image using moments method"""
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # The above angle is in the range [-90, 0)
        # Convert to absolute angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Rotate the image to deskew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    # Apply deskewing
    deskewed = deskew(thresh)

    # Optional: Noise removal with morphological operations
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(deskewed, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return closing


def extract_text(image_path, custom_config=None):
    """
    Extract text from image using Tesseract with preprocessing

    Args:
        image_path (str): Path to the input image
        custom_config (str, optional): Custom Tesseract configuration

    Returns:
        str: Extracted text
    """
    # Preprocess the image
    preprocessed_img = preprocess_image(image_path)

    # Default Tesseract configuration for creative text
    default_config = r"--oem 3 --psm 11 -c preserve_interword_spaces=1"

    # Use custom config if provided, otherwise use default
    config = custom_config or default_config

    # Extract text
    text = pytesseract.image_to_string(preprocessed_img, config=config)

    return text.strip()


def main():
    # Path to the image
    image_path = (
        r"C:\Users\genes\Pictures\Creative Texts\e48a388001b732d7b61d1c82918447f3.jpg"
    )

    try:
        # Extract text
        extracted_text = extract_text(image_path)

        # Check if text was detected
        if extracted_text:
            print("Extracted Text:")
            print(extracted_text)

            # Optional: Save extracted text to a file
            # output_file = os.path.splitext(image_path)[0] + "_ocr_output.txt"
            # with open(output_file, "w", encoding="utf-8") as f:
            #     f.write(extracted_text)
            # print(f"\nText saved to {output_file}")
        else:
            print("No Text Detected")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
