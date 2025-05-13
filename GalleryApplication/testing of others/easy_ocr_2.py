import easyocr
import cv2
import numpy as np
import torch
from typing import List, Dict, Optional, Union, Tuple
import argparse
import logging
from pathlib import Path
import json
from dataclasses import dataclass, asdict


@dataclass
class TextBlock:
    text: str
    confidence: float
    bounding_box: List[Tuple[float, float]]
    block_type: str = "paragraph"  # paragraph, heading, caption, etc.


class GeminiLikeOCRExtractor:
    def __init__(self):
        """Initialize with Gemini-like extraction capabilities."""
        self.logger = self._setup_logging()
        self.gpu_available = torch.cuda.is_available()
        self.reader = self._initialize_reader()
        self.layout_analyzer = LayoutAnalyzer()

    def _setup_logging(self) -> logging.Logger:
        """Configure advanced logging system."""
        logger = logging.getLogger("GeminiOCR")
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _initialize_reader(self) -> easyocr.Reader:
        """Initialize EasyOCR with correct parameters."""
        try:
            return easyocr.Reader(
                ["en"],
                gpu=self.gpu_available,
                model_storage_directory="model_storage",
                download_enabled=True,
            )
        except Exception as e:
            self.logger.error(f"Reader initialization failed: {e}")
            raise

    def extract_text(self, image_path: Union[str, np.ndarray]) -> Dict:
        """
        Gemini-like text extraction with layout analysis.

        Returns structured output with:
        - Raw text
        - Structured blocks (paragraphs, headings)
        - Confidence scores
        """
        try:
            # Step 1: Preprocess image
            image = self._load_and_preprocess(image_path)

            # Step 2: Perform text detection with proper parameters
            raw_results = self.reader.readtext(
                image,
                paragraph=True,
                min_size=10,
                text_threshold=0.6,
                low_text=0.4,
                link_threshold=0.4,
                canvas_size=1600,
                mag_ratio=1.5,
            )

            # Step 3: Process and structure results
            text_blocks = self._process_raw_results(raw_results)

            # Step 4: Analyze layout and semantic structure
            structured_blocks = self.layout_analyzer.analyze(text_blocks)

            # Step 5: Post-process for better readability
            cleaned_blocks = self._post_process(structured_blocks)

            return {
                "success": True,
                "raw_text": " ".join([b.text for b in cleaned_blocks]),
                "structured_text": [asdict(b) for b in cleaned_blocks],
                "average_confidence": np.mean([b.confidence for b in cleaned_blocks]),
                "analysis": {
                    "paragraph_count": sum(
                        1 for b in cleaned_blocks if b.block_type == "paragraph"
                    ),
                    "heading_count": sum(
                        1 for b in cleaned_blocks if b.block_type == "heading"
                    ),
                    "other_count": sum(
                        1
                        for b in cleaned_blocks
                        if b.block_type not in ["paragraph", "heading"]
                    ),
                },
            }

        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return {"success": False, "error": str(e)}

    def _load_and_preprocess(self, image_path: Union[str, np.ndarray]) -> np.ndarray:
        """Enhanced image loading and preprocessing pipeline."""
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
        else:
            image = image_path

        # Convert to grayscale with contrast enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)

        # Adaptive thresholding with noise reduction
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8
        )

        # Edge-preserving smoothing
        processed = cv2.bilateralFilter(processed, 9, 75, 75)

        return processed

    def _process_raw_results(self, results: List) -> List[TextBlock]:
        """Convert raw OCR results to structured text blocks."""
        blocks = []
        for result in results:
            if len(result) < 2:
                continue

            bbox, text = result[:2]
            confidence = result[2] if len(result) > 2 else 0.85

            blocks.append(
                TextBlock(
                    text=text.strip(),
                    confidence=float(confidence),
                    bounding_box=[(float(p[0]), float(p[1])) for p in bbox],
                )
            )

        return blocks

    def _post_process(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Clean and normalize extracted text blocks."""
        cleaned = []
        for block in blocks:
            # Normalize whitespace
            text = " ".join(block.text.split())

            # Basic capitalization fix for headings
            if block.block_type == "heading":
                text = text.capitalize()

            cleaned.append(
                TextBlock(
                    text=text,
                    confidence=block.confidence,
                    bounding_box=block.bounding_box,
                    block_type=block.block_type,
                )
            )

        return cleaned


class LayoutAnalyzer:
    """Mimics Gemini's layout understanding capabilities."""

    def analyze(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Analyze text layout to determine block types and relationships."""
        if not blocks:
            return []

        # Sort blocks by reading order (top-to-bottom, left-to-right)
        sorted_blocks = sorted(
            blocks,
            key=lambda b: (
                min(p[1] for p in b.bounding_box),  # y-coordinate
                min(p[0] for p in b.bounding_box),  # x-coordinate
            ),
        )

        # Classify blocks based on position and text features
        classified = []
        for i, block in enumerate(sorted_blocks):
            block_type = self._classify_block(block, i, sorted_blocks)
            classified.append(
                TextBlock(
                    text=block.text,
                    confidence=block.confidence,
                    bounding_box=block.bounding_box,
                    block_type=block_type,
                )
            )

        return classified

    def _classify_block(
        self, block: TextBlock, index: int, all_blocks: List[TextBlock]
    ) -> str:
        """Determine the type of text block."""
        text = block.text

        # Heading detection
        if len(text) < 50 and (
            text.isupper() or any(c.isdigit() for c in text) or text.endswith(":")
        ):
            return "heading"

        # Check if this is likely a continuation of previous block
        if index > 0:
            prev_block = all_blocks[index - 1]
            if self._blocks_are_related(prev_block, block):
                return (
                    prev_block.block_type
                    if hasattr(prev_block, "block_type")
                    else "paragraph"
                )

        # Default to paragraph
        return "paragraph"

    def _blocks_are_related(self, block1: TextBlock, block2: TextBlock) -> bool:
        """Determine if two blocks are part of the same logical section."""
        # Calculate vertical and horizontal distances
        b1_bottom = max(p[1] for p in block1.bounding_box)
        b2_top = min(p[1] for p in block2.bounding_box)

        b1_right = max(p[0] for p in block1.bounding_box)
        b2_left = min(p[0] for p in block2.bounding_box)

        # Check if blocks are vertically aligned with reasonable spacing
        vertical_spacing = b2_top - b1_bottom
        horizontal_overlap = min(b1_right, b2_left) - max(
            min(p[0] for p in block1.bounding_box),
            min(p[0] for p in block2.bounding_box),
        )

        return (
            vertical_spacing > 0 and vertical_spacing < 50 and horizontal_overlap > -20
        )


def main():
    """Command line interface with Gemini-like features."""
    parser = argparse.ArgumentParser(
        description="Gemini-like OCR Text Extraction System"
    )
    parser.add_argument("input", help="Image file path or directory")
    parser.add_argument("--output", help="Output file path (JSON format)")
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization of text blocks"
    )

    args = parser.parse_args()

    extractor = GeminiLikeOCRExtractor()

    if Path(args.input).is_dir():
        # Batch processing for directories
        results = {}
        for img_file in Path(args.input).glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                results[img_file.name] = extractor.extract_text(str(img_file))
    else:
        # Single file processing
        results = extractor.extract_text(args.input)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    if not isinstance(results, dict) or "success" not in results:
        print("Error in processing")
        return 1

    if results["success"]:
        print("\nExtraction Results:")
        print(f"Extracted Text: {results.get('raw_text', '')}")
        print(f"Raw Text Length: {len(results.get('raw_text', ''))} characters")
        print(f"Structured Blocks: {len(results.get('structured_text', []))}")
        print(f"Average Confidence: {results.get('average_confidence', 0):.2f}")

        if args.visualize:
            visualize_results(results, args.input)
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")
        return 1

    return 0


def visualize_results(results: Dict, image_path: str):
    """Visualize text blocks on the original image."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        image = cv2.imread(image_path)
        if image is None:
            return

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        for block in results.get("structured_text", []):
            box = np.array(block["bounding_box"])
            poly = Polygon(box, fill=False, edgecolor="red", linewidth=2)
            plt.gca().add_patch(poly)

            # Add text label
            plt.text(
                box[0][0],
                box[0][1],
                f"{block['block_type']}: {block['text'][:30]}...",
                color="yellow",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5),
            )

        plt.title("Text Block Visualization")
        plt.axis("off")
        plt.tight_layout()

        output_path = Path(image_path).stem + "_visualization.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Visualization saved to {output_path}")

    except ImportError:
        print("Visualization requires matplotlib")
    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == "__main__":
    import sys

    sys.exit(main())
