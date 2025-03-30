import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QStackedWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QScrollArea,
    QWidget,
    QGridLayout,
    QHBoxLayout,
    QComboBox,
    QDialog,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
from PyQt6.QtCore import Qt, QDateTime
from PyQt6.QtGui import QPixmap, QColor, QPainter, QBrush
from PyQt6.uic import loadUi
from PIL import Image, ImageFilter
import logging
from classify import HateSpeechDetector
from fpdf import FPDF
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

global_var_best_model_path = r"K:\Copy of Official Thesis\best_model.pth"
"May paths ng mga logo here na need i-manually set"
global_var_gallery_ui_path = r"K:\Official_Thesis\GalleryApplication\gallery.ui"


class WelcomeScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hate Speech Detector")
        self.setGeometry(0, 0, 1200, 800)
        self.setStyleSheet("background-color: #F8FAFC;")

        # Welcome message
        welcome_label = QLabel(
            "Hate Speech Detection System\n\n"
            "This application detects hate speech in user-generated images using\n"
            "CLIP and VisualBERT hybrid algorithms.\n\n"
            "To get started:\n"
            "1. Select 'Enter' to open the main application\n"
            "2. Use the sidebar to navigate between modules\n"
            "3. Upload images or folders to scan for hate speech\n"
            "4. View results in the dashboard and reports",
            self,
        )
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_label.setStyleSheet("font-size: 18px; color: #9AA6B2;")
        welcome_label.setGeometry(200, 200, 800, 400)

        # Enter button
        enter_button = QPushButton("Enter", self)
        enter_button.setGeometry(500, 600, 200, 50)
        enter_button.setStyleSheet("""
            QPushButton {
                background-color: #D9EAFD;
                color: #1e1e1e;
                border-radius: 10px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #BCCCDC;
            }
        """)
        enter_button.clicked.connect(self.close)


class GalleryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi(global_var_gallery_ui_path, self)

        # Apply color palette
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #F8FAFC;
            }
            QFrame#leftSidebar {
                background-color: #D9EAFD;
                border-right: 1px solid #BCCCDC;
            }
            QPushButton {
                background-color: #D9EAFD;
                color: #1e1e1e;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #BCCCDC;
            }
            QLabel {
                color: #1e1e1e;
            }
            QScrollArea {
                border: none;
            }
            QTableWidget {
                gridline-color: #BCCCDC;
            }
        """)

        # Initialize data structures
        self.folders = []
        self.folder_data = {}  # Will store folder metadata
        self.image_data = {}  # Will store per-image data including extracted text
        self.hate_speech_images = []
        self.non_hate_speech_images = []

        # Setup UI elements
        self.setup_ui()

        # Load model
        try:
            self.detector = HateSpeechDetector(
                model_path=global_var_best_model_path, device="cpu"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            sys.exit(1)

        # Connect signals
        self.connect_signals()

        # Show dashboard by default
        self.show_dashboard()

    def setup_ui(self):
        # Add app title and logo to top bar
        self.top_bar = QWidget()
        self.top_bar_layout = QHBoxLayout(self.top_bar)

        self.logo_label = QLabel()
        pixmap = QPixmap(
            r"K:\Official_Thesis\GalleryApplication\assets\logo.png"
        ).scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio)
        self.logo_label.setPixmap(pixmap)
        self.top_bar_layout.addWidget(self.logo_label)

        self.app_title = QLabel("Hate Speech Detector")
        self.app_title.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.top_bar_layout.addWidget(self.app_title)
        self.top_bar_layout.addStretch()

        # Add top bar to each page
        for i in range(self.mainContentStack.count()):
            page = self.mainContentStack.widget(i)
            if hasattr(page, "layout"):
                page.layout().insertWidget(0, self.top_bar)

        # Create IPO page (not in UI file)
        self.create_ipo_page()

    def create_ipo_page(self):
        """Create the IPO module page."""
        self.ipo_page = QWidget()
        ipo_layout = QVBoxLayout(self.ipo_page)

        # Add scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Add IPO table with proper header setup
        self.ipo_table = QTableWidget()
        self.ipo_table.setColumnCount(3)
        self.ipo_table.setHorizontalHeaderLabels(["Input", "Process", "Output"])

        # Configure header styling
        header = self.ipo_table.horizontalHeader()
        header.setStyleSheet("""
            QHeaderView::section {
                background-color: #D9EAFD;
                color: #1e1e1e;
                padding: 5px;
                border: 1px solid #BCCCDC;
                font-weight: bold;
            }
        """)

        # Configure column sizes
        self.ipo_table.setColumnWidth(0, 200)  # Input column
        self.ipo_table.setColumnWidth(1, 400)  # Process column
        self.ipo_table.setColumnWidth(2, 200)  # Output column
        header.setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )  # Make process column stretchable

        self.ipo_table.verticalHeader().setVisible(False)
        self.ipo_table.setAlternatingRowColors(True)
        self.ipo_table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #F0F4F8;
                background-color: #F8FAFC;
                gridline-color: #BCCCDC;
            }
        """)

        scroll_layout.addWidget(self.ipo_table)
        scroll_area.setWidget(scroll_content)
        ipo_layout.addWidget(scroll_area)

        # Add IPO page to stack
        self.mainContentStack.addWidget(self.ipo_page)

    def connect_signals(self):
        self.dashboardButton.clicked.connect(self.show_dashboard)
        self.exploreButton.clicked.connect(self.show_explore)
        self.hatespeechImagesButton.clicked.connect(self.show_hate_speech_images)
        self.ipoButton.clicked.connect(self.show_ipo_module)
        self.scanFolderButton.clicked.connect(self.scan_folder)
        self.selectFolderButton.clicked.connect(self.select_folder)
        self.reportButton.clicked.connect(self.generate_report)
        self.menuButton.clicked.connect(self.toggle_sidebar)

    def toggle_sidebar(self):
        self.leftSidebar.setVisible(not self.leftSidebar.isVisible())

    def show_dashboard(self):
        self.mainContentStack.setCurrentIndex(0)
        self.update_title("Dashboard")
        self.update_button_styles("dashboard")
        self.update_dashboard()

    def update_dashboard(self):
        """Update the Dashboard page with statistics and graph."""
        # Clear existing content
        for i in reversed(range(self.dashboardContentLayout.count())):
            widget = self.dashboardContentLayout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Create container for the entire dashboard content
        dashboard_container = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_container)

        # Top section - Folder buttons (original layout)
        top_section = QWidget()
        top_layout = QHBoxLayout(top_section)

        if not self.folders:
            scan_button = QPushButton("Click to Scan Folder")
            scan_button.setStyleSheet(self.button_style())
            scan_button.clicked.connect(self.scan_folder)
            top_layout.addWidget(scan_button)
        else:
            # Add top 5 folders as buttons (original behavior)
            sorted_folders = sorted(
                self.folders,
                key=lambda f: (
                    self.folder_data[f]["hate_speech_count"]
                    / self.folder_data[f]["total_images"]
                    if self.folder_data[f]["total_images"] > 0
                    else 0
                ),
                reverse=True,
            )[:5]

            for folder in sorted_folders:
                folder_name = self.get_folder_name(folder)
                hate_speech_percentage = (
                    (
                        self.folder_data[folder]["hate_speech_count"]
                        / self.folder_data[folder]["total_images"]
                    )
                    * 100
                    if self.folder_data[folder]["total_images"] > 0
                    else 0
                )
                folder_button = QPushButton(
                    f"{folder_name}\n({hate_speech_percentage:.2f}% hate speech)"
                )
                folder_button.setStyleSheet(self.button_style())
                top_layout.addWidget(folder_button)

        dashboard_layout.addWidget(top_section)

        # Bottom section - Statistics and graph
        bottom_section = QWidget()
        bottom_layout = QVBoxLayout(bottom_section)

        # Add statistics
        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)

        total_folders = QLabel(f"Total Folders: {len(self.folders)}")
        total_images = sum(data["total_images"] for data in self.folder_data.values())
        total_images_label = QLabel(f"Total Images: {total_images}")
        hate_speech_count = sum(
            data["hate_speech_count"] for data in self.folder_data.values()
        )
        hate_speech_label = QLabel(f"Hate Speech Images: {hate_speech_count}")

        stats_layout.addWidget(total_folders)
        stats_layout.addWidget(total_images_label)
        stats_layout.addWidget(hate_speech_label)
        bottom_layout.addWidget(stats_widget)

        # Add graph (takes remaining space)
        self.create_dashboard_graph(bottom_layout)

        dashboard_layout.addWidget(bottom_section)

        # Add the container to the main layout
        self.dashboardContentLayout.addWidget(dashboard_container)

    def create_dashboard_graph(self, parent_layout):
        """Create a bar chart showing hate speech percentages by folder."""
        try:
            fig = Figure(figsize=(8, 4), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            # Prepare data
            folders = []
            percentages = []
            for folder in self.folders:
                total = self.folder_data[folder]["total_images"]
                if total > 0:
                    percentage = (
                        self.folder_data[folder]["hate_speech_count"] / total
                    ) * 100
                    folders.append(self.get_folder_name(folder))
                    percentages.append(percentage)

            # Create bar chart
            bars = ax.bar(folders, percentages, color="#D9EAFD")

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                )

            ax.set_title("Hate Speech Detection Accuracy by Folder")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)

            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            # Tight layout to prevent label cutoff
            fig.tight_layout()

            parent_layout.addWidget(canvas)

        except Exception as e:
            logging.error(f"Error creating dashboard graph: {e}")
            error_label = QLabel("Error generating graph")
            parent_layout.addWidget(error_label)

    def show_explore(self):
        self.mainContentStack.setCurrentIndex(1)
        self.update_title("Explore")
        self.update_button_styles("explore")
        self.update_explore()

    def show_hate_speech_images(self):
        if not self.folders and not self.hate_speech_images:
            QMessageBox.warning(self, "No Data", "No hate speech images detected yet.")
            return
        self.mainContentStack.setCurrentIndex(2)
        self.update_title("Hate Speech Images")
        self.update_button_styles("hatespeech")
        self.update_hate_speech_images()

    def show_ipo_module(self):
        self.mainContentStack.setCurrentWidget(self.ipo_page)
        self.update_title("IPO Module")
        self.update_button_styles("ipo")
        self.update_ipo_table()

    def update_button_styles(self, active_button):
        buttons = {
            "dashboard": self.dashboardButton,
            "explore": self.exploreButton,
            "hatespeech": self.hatespeechImagesButton,
            "ipo": self.ipoButton,
        }

        for name, button in buttons.items():
            if name == active_button:
                button.setStyleSheet("background-color: #BCCCDC;")
            else:
                button.setStyleSheet("background-color: transparent;")

    def update_title(self, title):
        title_label = self.findChild(QLabel, "dashboardTitleLabel")
        if title_label:
            title_label.setText(title)

    def scan_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.process_folder(folder)
            self.update_dashboard()
            self.update_explore()
            self.update_ipo_table()

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.process_folder(folder)
            self.update_explore()
            self.update_ipo_table()

    def process_folder(self, folder):
        if folder not in self.folders:
            self.folders.append(folder)
            self.folder_data[folder] = {
                "hate_speech_count": 0,
                "total_images": 0,
                "errors": 0,
            }

        error_messages = []

        for root, _, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)

                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    error_messages.append(f"Skipped non-image file: {file}")
                    continue

                try:
                    # Skip if we've already processed this image
                    if file_path in self.image_data:
                        continue

                    img = Image.open(file_path)
                    width, height = img.size
                    if width < 50 or height < 50:
                        raise ValueError(
                            "Image too small to process (min 50x50 pixels)"
                        )
                    if width > 5000 or height > 5000:
                        raise ValueError(
                            "Image too large to process (max 5000x5000 pixels)"
                        )

                    # Extract text and store it
                    extracted_text = self.detector.extract_text(file_path)
                    prediction = self.detector.predict(file_path, extracted_text)

                    # Store all image data
                    self.image_data[file_path] = {
                        "text": extracted_text,
                        "prediction": prediction,
                        "is_hate_speech": prediction > 0.5,
                        "folder": folder,
                    }

                    if prediction > 0.5:
                        self.hate_speech_images.append(file_path)
                        self.folder_data[folder]["hate_speech_count"] += 1
                    else:
                        self.non_hate_speech_images.append(file_path)
                    self.folder_data[folder]["total_images"] += 1

                except Exception as e:
                    logging.error(f"Error processing image {file_path}: {e}")
                    self.folder_data[folder]["errors"] += 1
                    error_messages.append(f"Error processing {file}: {str(e)}")

        if error_messages:
            self.show_error_dialog(
                "Processing Errors",
                f"Processed {self.folder_data[folder]['total_images']} images with {len(error_messages)} errors:",
                "\n".join(error_messages),
            )

        if error_messages:
            self.show_error_dialog(
                "Processing Errors",
                f"Processed {self.folder_data[folder]['total_images']} images with {len(error_messages)} errors:",
                "\n".join(error_messages),
            )

    def update_explore(self):
        for i in reversed(range(self.exploreScrollAreaLayout.count())):
            widget = self.exploreScrollAreaLayout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Selection widgets
        selection_widget = QWidget()
        selection_layout = QHBoxLayout(selection_widget)

        self.selection_combo = QComboBox()
        self.selection_combo.addItem("Select Folder")
        self.selection_combo.addItem("Select Image")
        selection_layout.addWidget(self.selection_combo)

        self.select_button = QPushButton("Select")
        self.select_button.clicked.connect(self.handle_selection)
        selection_layout.addWidget(self.select_button)

        self.view_prediction_button = QPushButton("View Prediction Scores")
        self.view_prediction_button.clicked.connect(self.show_prediction_scores)
        selection_layout.addWidget(self.view_prediction_button)

        scroll_layout.addWidget(selection_widget)

        # All images section
        all_images_label = QLabel("All Images")
        all_images_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        scroll_layout.addWidget(all_images_label)

        image_container = QWidget()
        image_layout = QGridLayout(image_container)
        image_layout.setHorizontalSpacing(10)
        image_layout.setVerticalSpacing(10)

        all_images = self.hate_speech_images + self.non_hate_speech_images
        for i, image_path in enumerate(all_images):
            try:
                pixmap = QPixmap(image_path)
                if pixmap.isNull():
                    raise ValueError("Invalid image file")

                if image_path in self.hate_speech_images:
                    image = Image.open(image_path)
                    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
                    blurred_image.save("temp_blurred.jpg")
                    pixmap = QPixmap("temp_blurred.jpg")

                image_label = QLabel()
                image_label.setPixmap(
                    pixmap.scaled(
                        200, 200, Qt.AspectRatioMode.KeepAspectRatioByExpanding
                    )
                )
                image_label.setFixedSize(200, 200)
                image_label.setStyleSheet("border: 1px solid #BCCCDC;")

                folder = self.get_image_folder(image_path)
                if folder:
                    image_label.setToolTip(f"Folder: {folder}")

                image_layout.addWidget(image_label, i // 4, i % 4)
            except Exception as e:
                error_label = QLabel(
                    f"Error loading image\n{os.path.basename(image_path)}"
                )
                error_label.setFixedSize(200, 200)
                error_label.setStyleSheet("border: 1px solid red; color: red;")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                image_layout.addWidget(error_label, i // 4, i % 4)

        scroll_layout.addWidget(image_container)
        self.exploreScrollAreaLayout.addWidget(scroll_content)

    def handle_selection(self):
        selection = self.selection_combo.currentText()
        if selection == "Select Folder":
            self.select_folder()
        else:
            self.select_image()

    def select_image(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file:
            try:
                # Skip if already processed
                if file in self.image_data:
                    return

                img = Image.open(file)
                width, height = img.size
                if width < 50 or height < 50:
                    raise ValueError("Image too small to process (min 50x50 pixels)")
                if width > 5000 or height > 5000:
                    raise ValueError(
                        "Image too large to process (max 5000x5000 pixels)"
                    )

                # Extract and store data
                extracted_text = self.detector.extract_text(file)
                prediction = self.detector.predict(file, extracted_text)

                self.image_data[file] = {
                    "text": extracted_text,
                    "prediction": prediction,
                    "is_hate_speech": prediction > 0.5,
                    "folder": None,  # Single image, no folder
                }

                if prediction > 0.5:
                    self.hate_speech_images.append(file)
                else:
                    self.non_hate_speech_images.append(file)

                self.update_explore()
                self.update_ipo_table()

            except Exception as e:
                self.show_error_dialog("Error", f"Failed to process image: {str(e)}")

    def update_hate_speech_images(self):
        for i in reversed(range(self.hateSpeechImagesScrollAreaLayout.count())):
            widget = self.hateSpeechImagesScrollAreaLayout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        if not self.hate_speech_images:
            no_images_label = QLabel("No hate speech images detected")
            no_images_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scroll_layout.addWidget(no_images_label)
        else:
            image_container = QWidget()
            image_layout = QGridLayout(image_container)
            image_layout.setHorizontalSpacing(10)
            image_layout.setVerticalSpacing(10)

            for i, image_path in enumerate(self.hate_speech_images):
                try:
                    # Apply blur effect to hate speech images
                    image = Image.open(image_path)
                    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
                    blurred_image.save("temp_blurred.jpg")
                    pixmap = QPixmap("temp_blurred.jpg")

                    if pixmap.isNull():
                        raise ValueError("Invalid image file")

                    image_label = QLabel()
                    image_label.setPixmap(
                        pixmap.scaled(
                            200, 200, Qt.AspectRatioMode.KeepAspectRatioByExpanding
                        )
                    )
                    image_label.setFixedSize(200, 200)
                    image_label.setStyleSheet("border: 1px solid #BCCCDC;")

                    folder = self.get_image_folder(image_path)
                    if folder:
                        image_label.setToolTip(f"Folder: {folder}")

                    image_layout.addWidget(image_label, i // 4, i % 4)
                except Exception as e:
                    error_label = QLabel(
                        f"Error loading image\n{os.path.basename(image_path)}"
                    )
                    error_label.setFixedSize(200, 200)
                    error_label.setStyleSheet("border: 1px solid red; color: red;")
                    error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    image_layout.addWidget(error_label, i // 4, i % 4)

        scroll_layout.addWidget(image_container)

        self.hateSpeechImagesScrollAreaLayout.addWidget(scroll_content)

    def update_ipo_table(self):
        self.ipo_table.setRowCount(0)

        # Add folder images
        for folder in self.folders:
            row = self.ipo_table.rowCount()
            self.ipo_table.insertRow(row)
            folder_item = QTableWidgetItem(f"Folder: {self.get_folder_name(folder)}")
            folder_item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.ipo_table.setSpan(row, 0, 1, 3)
            self.ipo_table.setItem(row, 0, folder_item)

            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(root, file)
                        self.add_ipo_row(image_path, folder)

        # Add single images
        single_images = [
            img
            for img in self.hate_speech_images + self.non_hate_speech_images
            if not any(img.startswith(folder) for folder in self.folders)
        ]

        if single_images:
            row = self.ipo_table.rowCount()
            self.ipo_table.insertRow(row)
            single_item = QTableWidgetItem("Single Images")
            single_item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.ipo_table.setSpan(row, 0, 1, 3)
            self.ipo_table.setItem(row, 0, single_item)

            for image_path in single_images:
                self.add_ipo_row(image_path)

    def add_ipo_row(self, image_path, folder=None):
        try:
            row = self.ipo_table.rowCount()
            self.ipo_table.insertRow(row)

            # Input column
            input_label = QLabel()
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                raise ValueError("Invalid image file")

            pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
            input_label.setPixmap(pixmap)
            input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ipo_table.setCellWidget(row, 0, input_label)

            # Process column - create a detailed process log
            process_log = QTextEdit()
            process_log.setReadOnly(True)
            process_log.setStyleSheet("""
                QTextEdit {
                    background-color: #F8FAFC;
                    border: 1px solid #BCCCDC;
                    font-family: Consolas, monospace;
                    font-size: 10pt;
                    color: #1e1e1e;
                    padding: 5px;
                }
            """)

            # Get or create image data
            if image_path not in self.image_data:
                process_log.append("🔄 Processing image...")
                process_log.append(f"📂 Image: {os.path.basename(image_path)}")

                # OCR Extraction
                process_log.append("\n🔍 Performing OCR text extraction...")
                text = self.detector.extract_text(image_path)
                process_log.append(f"📝 Extracted Text: {text}")

                # Prediction
                process_log.append("\n🧠 Running hate speech prediction...")
                prediction = self.detector.predict(image_path, text)
                process_log.append(f"📊 Prediction Score: {prediction:.4f}")
                process_log.append(
                    f"🚨 Hate Speech: {'YES' if prediction > 0.5 else 'NO'}"
                )

                self.image_data[image_path] = {
                    "text": text,
                    "prediction": prediction,
                    "is_hate_speech": prediction > 0.5,
                    "folder": folder,
                    "process_log": process_log.toPlainText(),
                }
            else:
                data = self.image_data[image_path]
                process_log.append(f"📂 Image: {os.path.basename(image_path)}")
                process_log.append("\n🔍 OCR Results (from cache):")
                process_log.append(f"📝 Extracted Text: {data['text']}")
                process_log.append("\n🧠 Prediction Results (from cache):")
                process_log.append(f"📊 Prediction Score: {data['prediction']:.4f}")
                process_log.append(
                    f"🚨 Hate Speech: {'YES' if data['is_hate_speech'] else 'NO'}"
                )
                process_log.append("\nℹ️ Using cached results from initial processing")

            process_log.append("\n" + "═" * 40)
            process_log.append("Processing complete")

            self.ipo_table.setCellWidget(row, 1, process_log)

            # Output column
            output_label = QLabel()
            is_hate_speech = self.image_data[image_path]["is_hate_speech"]

            if is_hate_speech:
                image = Image.open(image_path)
                blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
                blurred_image.save("temp_blurred.jpg")
                output_pixmap = QPixmap("temp_blurred.jpg")
                process_log.append("\n⚠️ Applying blur filter (hate speech detected)")
            else:
                output_pixmap = QPixmap(image_path)
                process_log.append("\n✅ No blur needed (no hate speech detected)")

            output_label.setPixmap(
                output_pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
            )
            output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ipo_table.setCellWidget(row, 2, output_label)

            self.ipo_table.resizeRowToContents(row)

        except Exception as e:
            error_item = QTableWidgetItem(f"Error processing image: {str(e)}")
            error_item.setForeground(QColor("red"))
            self.ipo_table.setItem(row, 1, error_item)

    def show_prediction_scores(self):
        if not self.folders:
            QMessageBox.information(
                self, "No Data", "No folders have been scanned yet."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Prediction Scores")
        dialog.setMinimumSize(400, 500)

        layout = QVBoxLayout()
        title = QLabel("Folder Prediction Scores")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        for folder in self.folders:
            folder_name = self.get_folder_name(folder)
            total = self.folder_data[folder]["total_images"]
            hate_count = self.folder_data[folder]["hate_speech_count"]
            percentage = (hate_count / total) * 100 if total > 0 else 0

            folder_widget = QWidget()
            folder_layout = QHBoxLayout(folder_widget)

            name_label = QLabel(folder_name)
            score_label = QLabel(f"{percentage:.1f}% hate speech")

            folder_layout.addWidget(name_label)
            folder_layout.addWidget(score_label)

            content_layout.addWidget(folder_widget)

        scroll.setWidget(content)
        layout.addWidget(scroll)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.exec()

    def generate_report(self):
        if not self.hate_speech_images:
            QMessageBox.warning(
                self, "No Data", "No hate speech images detected to generate a report."
            )
            return

        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, txt="Hate Speech Detection Report", ln=True, align="C")

            current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
            pdf.cell(
                200, 10, txt=f"Report Generated on: {current_time}", ln=True, align="C"
            )

            # Add statistics
            pdf.cell(200, 10, txt="Statistics:", ln=True, align="L")
            pdf.cell(
                200,
                10,
                txt=f"Total Folders Scanned: {len(self.folders)}",
                ln=True,
                align="L",
            )
            total_images = sum(
                data["total_images"] for data in self.folder_data.values()
            )
            pdf.cell(
                200, 10, txt=f"Total Images Scanned: {total_images}", ln=True, align="L"
            )
            hate_speech_count = sum(
                data["hate_speech_count"] for data in self.folder_data.values()
            )
            pdf.cell(
                200,
                10,
                txt=f"Hate Speech Images Detected: {hate_speech_count}",
                ln=True,
                align="L",
            )

            # Add folder percentages
            pdf.cell(200, 10, txt="Folder Statistics:", ln=True, align="L")
            for folder in self.folders:
                percentage = (
                    self.folder_data[folder]["hate_speech_count"]
                    / self.folder_data[folder]["total_images"]
                ) * 100
                pdf.cell(
                    200,
                    10,
                    txt=f"{self.get_folder_name(folder)}: {percentage:.1f}% hate speech",
                    ln=True,
                    align="L",
                )

            # Add hate speech images
            pdf.cell(200, 10, txt="Hate Speech Images:", ln=True, align="L")
            for image_path in self.hate_speech_images:
                # Original image
                pdf.cell(
                    200,
                    10,
                    txt=f"Original: {os.path.basename(image_path)}",
                    ln=True,
                    align="L",
                )
                pdf.image(image_path, x=10, y=pdf.get_y(), w=100)
                pdf.cell(200, 10, txt="", ln=True)  # Add space

                # Blurred image
                image = Image.open(image_path)
                blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
                blurred_image.save("temp_blurred.jpg")
                pdf.cell(
                    200,
                    10,
                    txt=f"Censored: {os.path.basename(image_path)}",
                    ln=True,
                    align="L",
                )
                pdf.image("temp_blurred.jpg", x=10, y=pdf.get_y(), w=100)
                pdf.cell(200, 10, txt="", ln=True)  # Add space

            report_path = "hate_speech_report.pdf"
            pdf.output(report_path)
            QMessageBox.information(
                self, "Report Generated", f"Report saved as {report_path}"
            )
        except Exception as e:
            self.show_error_dialog(
                "Report Error", f"Failed to generate report: {str(e)}"
            )

    def get_folder_name(self, folder_path):
        folder_name = os.path.basename(folder_path)
        words = folder_name.split()[:3]
        return " ".join(words) + ("..." if len(folder_name.split()) > 3 else "")

    def get_image_folder(self, image_path):
        for folder in self.folders:
            if image_path.startswith(folder):
                return self.get_folder_name(folder)
        return None

    def button_style(self):
        return """
            background-color: #D9EAFD;
            color: #1e1e1e;
            padding: 10px;
            border-radius: 5px;
            border: none;
        """

    def show_error_dialog(self, title, message, details=""):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout()

        message_label = QLabel(message)
        layout.addWidget(message_label)

        if details:
            details_text = QTextEdit()
            details_text.setPlainText(details)
            details_text.setReadOnly(True)
            layout.addWidget(details_text)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Show welcome screen first
    welcome = WelcomeScreen()
    welcome.show()
    app.exec()

    # Then show main application
    window = GalleryApp()
    window.show()
    sys.exit(app.exec())
