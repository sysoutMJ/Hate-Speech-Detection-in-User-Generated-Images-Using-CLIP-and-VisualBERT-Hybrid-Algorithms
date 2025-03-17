import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QStackedWidget, QLabel, QPushButton, QVBoxLayout, QScrollArea, QWidget, QGridLayout, QHBoxLayout
from PyQt6.uic import loadUi
from PyQt6.QtCore import Qt, QSize, QDateTime
from PyQt6.QtGui import QIcon, QPixmap, QImage, QPainter, QColor
from PIL import Image, ImageFilter
import logging
from classify import HateSpeechDetector
from fpdf import FPDF  # For PDF generation

# Configure logging
logging.basicConfig(level=logging.INFO)

class GalleryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi(r"K:\Jeremy\Cus_merged_jver2_jver3_18032025\GallerApplication\gallery.ui", self)

        # Connect buttons to their respective functions
        self.dashboardButton.clicked.connect(self.show_dashboard)
        self.exploreButton.clicked.connect(self.show_explore)
        self.hatespeechImagesButton.clicked.connect(self.show_hate_speech_images)
        self.scanFolderButton.clicked.connect(self.scan_folder)
        self.selectFolderButton.clicked.connect(self.select_folder)
        self.reportButton.clicked.connect(self.generate_report)
        self.connect_menu_buttons()

        # Connect the menu button to toggle_sidebar
        # self.menuButton = self.findChild(QPushButton, "menuButton")
        # if self.menuButton:
        #     self.menuButton.clicked.connect(self.toggle_sidebar)
        # else:
        #     logging.error("Menu button not found in UI file.")

        # Initialize the main content stack (pages)
        self.mainContentStack = self.findChild(QStackedWidget, "mainContentStack")
        if self.mainContentStack is None:
            raise ValueError("mainContentStack not found in UI file")

        # Initialize folder data
        self.folders = []  # List of scanned folders
        self.folder_data = {}  # Dictionary to store folder metadata (e.g., hate speech percentage)
        self.hate_speech_images = []  # List of hate speech images
        self.non_hate_speech_images = []  # List of non-hate speech images

        # Load the hate speech detector
        self.detector = HateSpeechDetector(
            model_path=r"K:\Jeremy\Jeremy_ver2_100325\Model\best_model.pth",
            device="cpu"  # Use "cuda" if you have a GPU
        )

        # Show the dashboard by default
        self.show_dashboard()

        self.show()
    
    def connect_menu_buttons(self):
        # Page 0
        page0 = self.mainContentStack.widget(0)
        menu_button0 = page0.findChild(QPushButton, "menuButton")
        if menu_button0:
            menu_button0.clicked.connect(self.toggle_sidebar)
            print("Connected menuButton on page 0")
        else:
            print("WARNING: menuButton not found on page 0")

        # Page 1 (Explore Page)
        page1 = self.mainContentStack.widget(1)
        menu_button1 = page1.findChild(QPushButton, "menuButton")
        if not menu_button1:
            # Try finding inside the ScrollArea
            scroll_area = page1.findChild(QScrollArea)
            if scroll_area:
                menu_button1 = scroll_area.findChild(QPushButton, "menuButton")

        if menu_button1:
            menu_button1.clicked.connect(self.toggle_sidebar)
            print("Connected menuButton on page 1")
        else:
            print("WARNING: menuButton not found on page 1")

        # Page 2 (Hate Speech Images Page)
        page2 = self.mainContentStack.widget(2)
        menu_button2 = page2.findChild(QPushButton, "menuButton")
        if not menu_button2:
            # Try finding inside the ScrollArea
            scroll_area = page2.findChild(QScrollArea)
            if scroll_area:
                menu_button2 = scroll_area.findChild(QPushButton, "menuButton")

        if menu_button2:
            menu_button2.clicked.connect(self.toggle_sidebar)
            print("Connected menuButton on page 2")
        else:
            print("WARNING: menuButton not found on page 2")



    def toggle_sidebar(self):
        """Toggle the visibility of the sidebar."""
        if self.leftSidebar.isVisible():
            self.leftSidebar.hide()
        else:
            self.leftSidebar.show()

    def show_dashboard(self):
        """Switch to the Dashboard page."""
        self.mainContentStack.setCurrentIndex(0)
        self.update_title("Dashboard")
        self.dashboardButton.setStyleSheet("background-color: #444;")  # Highlight the Dashboard button
        self.exploreButton.setStyleSheet("background-color: transparent;")
        self.hatespeechImagesButton.setStyleSheet("background-color: transparent;")
        self.update_dashboard()

    def show_explore(self):
        """Switch to the Explore page."""
        self.mainContentStack.setCurrentIndex(1)
        self.update_title("Explore")
        self.dashboardButton.setStyleSheet("background-color: transparent;")
        self.exploreButton.setStyleSheet("background-color: #444;")  # Highlight the Explore button
        self.hatespeechImagesButton.setStyleSheet("background-color: transparent;")
        self.update_explore()

    def show_hate_speech_images(self):
        """Switch to the Hate Speech Images page."""
        if not self.folders:
            QMessageBox.warning(self, "No Folders Scanned", "Select a folder first to detect hate speech images.")
            return
        self.mainContentStack.setCurrentIndex(2)
        self.update_title("Hate Speech Images")
        self.dashboardButton.setStyleSheet("background-color: transparent;")
        self.exploreButton.setStyleSheet("background-color: transparent;")
        self.hatespeechImagesButton.setStyleSheet("background-color: #444;")  # Highlight the Hate Speech Images button
        self.update_hate_speech_images()

    def scan_folder(self):
        """Handle the Scan Folder button click."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.process_folder(folder)
            self.update_dashboard()
            self.update_explore()

    def select_folder(self):
        """Handle the Select Folder button click."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.process_folder(folder)
            self.update_explore()

    def process_folder(self, folder):
        """Process images in the selected folder."""
        if folder not in self.folders:
            self.folders.append(folder)
            self.folder_data[folder] = {"hate_speech_count": 0, "total_images": 0}

        # Process images in the folder
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    try:
                        # Predict hate speech
                        prediction = self.detector.predict(image_path, "")
                        logging.info(f"Prediction for {image_path}: {prediction}")

                        if prediction > 0.5:  # Adjust threshold as needed
                            self.hate_speech_images.append(image_path)
                            self.folder_data[folder]["hate_speech_count"] += 1
                        else:
                            self.non_hate_speech_images.append(image_path)
                        self.folder_data[folder]["total_images"] += 1
                    except Exception as e:
                        logging.error(f"Error processing image {image_path}: {e}")

    def update_dashboard(self):
        """Update the Dashboard page with the top 5 folders."""
        # Clear the existing content
        for i in reversed(range(self.dashboardContentLayout.count())):
            self.dashboardContentLayout.itemAt(i).widget().setParent(None)

        # If no folders are scanned, show the "Click to Scan Folder" button
        if not self.folders:
            scan_button = QPushButton("Click to Scan Folder", self)
            scan_button.setStyleSheet(
                "background-color: #444; color: #ffffff; padding: 10px; border-radius: 5px; border: none;"
            )
            scan_button.clicked.connect(self.scan_folder)
            self.dashboardContentLayout.addWidget(scan_button)
            return

        # Sort folders by hate speech percentage (or most recent if no hate speech)
        sorted_folders = sorted(
            self.folders,
            key=lambda f: (
                self.folder_data[f]["hate_speech_count"] / self.folder_data[f]["total_images"]
                if self.folder_data[f]["total_images"] > 0
                else 0
            ),
            reverse=True,
        )[:5]  # Get top 5 folders

        # Add the top 5 folders
        for folder in sorted_folders:
            folder_name = self.get_folder_name(folder)
            hate_speech_percentage = (
                (self.folder_data[folder]["hate_speech_count"] / self.folder_data[folder]["total_images"]) * 100
                if self.folder_data[folder]["total_images"] > 0
                else 0
            )
            folder_button = QPushButton(f"{folder_name}\n({hate_speech_percentage:.2f}% hate speech)", self)
            folder_button.setStyleSheet(
                "background-color: #444; color: #ffffff; padding: 10px; border-radius: 5px; border: none;"
            )
            self.dashboardContentLayout.addWidget(folder_button)

    def update_explore(self):
        """Update the Explore page with the recently scanned folders and images."""
        # Clear the existing content
        for i in reversed(range(self.exploreScrollAreaLayout.count())):
            self.exploreScrollAreaLayout.itemAt(i).widget().setParent(None)

        # Add the recently scanned folders (3 per row)
        folder_container = QWidget()
        folder_layout = QGridLayout(folder_container)
        folder_layout.setHorizontalSpacing(10)
        folder_layout.setVerticalSpacing(10)
        for i, folder in enumerate(self.folders):
            folder_name = self.get_folder_name(folder)
            hate_speech_percentage = (
                (self.folder_data[folder]["hate_speech_count"] / self.folder_data[folder]["total_images"]) * 100
                if self.folder_data[folder]["total_images"] > 0
                else 0
            )
            folder_button = QPushButton(f"{folder_name}\n({hate_speech_percentage:.2f}% hate speech)", self)
            folder_button.setStyleSheet(
                "background-color: #444; color: #ffffff; padding: 10px; border-radius: 5px; border: none;"
            )
            folder_button.clicked.connect(lambda _, f=folder: self.show_folder_images(f))
            folder_layout.addWidget(folder_button, i // 3, i % 3)
        self.exploreScrollAreaLayout.addWidget(folder_container)

        # Add "All Images" section (4 per row)
        all_images_label = QLabel("All Images", self)
        all_images_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        self.exploreScrollAreaLayout.addWidget(all_images_label)

        image_container = QWidget()
        image_layout = QGridLayout(image_container)
        image_layout.setHorizontalSpacing(10)
        image_layout.setVerticalSpacing(10)
        for i, image_path in enumerate(self.hate_speech_images + self.non_hate_speech_images):
            image_label = QLabel(self)
            pixmap = QPixmap(image_path)
            if image_path in self.hate_speech_images:
                # Apply Gaussian blur to hate speech images
                image = Image.open(image_path)
                blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
                blurred_image.save("temp_blurred.jpg")
                pixmap = QPixmap("temp_blurred.jpg")
            image_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
            image_layout.addWidget(image_label, i // 4, i % 4)
        self.exploreScrollAreaLayout.addWidget(image_container)

    def show_folder_images(self, folder):
        """Show images in the selected folder."""
        # Create a new page for the folder
        folder_page = QWidget()
        folder_layout = QVBoxLayout(folder_page)

        # Add a back button
        back_button = QPushButton("Back to Explore", folder_page)
        back_button.setStyleSheet("background-color: #444; color: #ffffff; padding: 10px; border-radius: 5px; border: none;")
        back_button.clicked.connect(self.show_explore)
        folder_layout.addWidget(back_button)

        # Add images from the folder (4 per row)
        image_container = QWidget()
        image_layout = QGridLayout(image_container)
        image_layout.setHorizontalSpacing(10)
        image_layout.setVerticalSpacing(10)
        for i, (root, _, files) in enumerate(os.walk(folder)):
            for j, file in enumerate(files):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    image_label = QLabel(folder_page)
                    pixmap = QPixmap(image_path)
                    if image_path in self.hate_speech_images:
                        # Apply Gaussian blur to hate speech images
                        image = Image.open(image_path)
                        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
                        blurred_image.save("temp_blurred.jpg")
                        pixmap = QPixmap("temp_blurred.jpg")
                    image_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
                    image_layout.addWidget(image_label, (i * len(files) + j) // 4, ((i * len(files) + j) % 4))
        folder_layout.addWidget(image_container)

        # Add the folder page to the stack
        self.mainContentStack.addWidget(folder_page)
        self.mainContentStack.setCurrentWidget(folder_page)
        self.update_title(f"Explore > {self.get_folder_name(folder)}")

    def update_hate_speech_images(self):
        """Update the Hate Speech Images page with blurred images."""
        # ✅ Clear existing content
        for i in reversed(range(self.hateSpeechImagesScrollAreaLayout.count())):
            self.hateSpeechImagesScrollAreaLayout.itemAt(i).widget().setParent(None)

        # ✅ Create a new container for images
        image_container = QWidget()
        image_layout = QGridLayout(image_container)
        image_layout.setHorizontalSpacing(10)
        image_layout.setVerticalSpacing(10)

        # ✅ Loop through all detected hate speech images and blur them
        for i, image_path in enumerate(self.hate_speech_images):
            image_label = QLabel(self)

            # ✅ Apply Gaussian blur to hide offensive content
            image = Image.open(image_path)
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
            temp_blurred_path = "temp_blurred.jpg"
            blurred_image.save(temp_blurred_path)  # Save temporarily

            # ✅ Display blurred image
            pixmap = QPixmap(temp_blurred_path)
            image_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
            image_layout.addWidget(image_label, i // 4, i % 4)

        # ✅ Add images to the layout
        self.hateSpeechImagesScrollAreaLayout.addWidget(image_container)

    def generate_report(self):
        """Generate a PDF report of hate speech images."""
        if not self.hate_speech_images:
            QMessageBox.warning(self, "No Hate Speech Images", "No hate speech images detected to generate a report.")
            return

        # Create a PDF object
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add the title of the study
        pdf.cell(200, 10, txt="Hate Speech Detection in User-Generated Images Using CLIP and VisualBERT Hybrid Algorithms", ln=True, align='C')

        # Add the date and time
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        pdf.cell(200, 10, txt=f"Report Generated on: {current_time}", ln=True, align='C')

        # Add a section for the percentage of hate speech images in each folder
        pdf.cell(200, 10, txt="Percentage of Hate Speech Images by Folder:", ln=True, align='L')
        for folder in self.folders:
            hate_speech_percentage = (
                (self.folder_data[folder]["hate_speech_count"] / self.folder_data[folder]["total_images"]) * 100
                if self.folder_data[folder]["total_images"] > 0
                else 0
            )
            pdf.cell(200, 10, txt=f"{self.get_folder_name(folder)}: {hate_speech_percentage:.2f}%", ln=True, align='L')

        # Add a section for the hate speech images
        pdf.cell(200, 10, txt="Hate Speech Images:", ln=True, align='L')
        for image_path in self.hate_speech_images:
            # Add the original image
            pdf.cell(200, 10, txt=f"Original Image: {image_path}", ln=True, align='L')
            # pdf.image(image_path, x=10, y=pdf.get_y(), w=100)

            # Add the censored (blurred) image
            blurred_image_path = "temp_blurred.jpg"
            image = Image.open(image_path)
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
            blurred_image.save(blurred_image_path)
            pdf.cell(200, 10, txt=f"Censored Image: {image_path}", ln=True, align='L')
            # pdf.image(blurred_image_path, x=10, y=pdf.get_y(), w=100)

        # Save the PDF
        report_path = "hate_speech_report.pdf"
        pdf.output(report_path)
        QMessageBox.information(self, "Report Generated", f"Report saved as {report_path}")

    def get_folder_name(self, folder_path):
        """Extract the folder name from the full path and limit it to three words."""
        folder_name = folder_path.split("/")[-1]  # Get the last part of the path
        words = folder_name.split()[:3]  # Limit to three words
        return " ".join(words) + ("..." if len(folder_name.split()) > 3 else "")

    def update_title(self, title):
        """Update the title label in the top bar."""
        title_label = self.findChild(QLabel, "dashboardTitleLabel")
        if title_label:
            title_label.setText(title)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GalleryApp()
    sys.exit(app.exec())