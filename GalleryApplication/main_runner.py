# main_runner.py (Merged Version)
# This Python file uses the following encoding: utf-8
import io
import sys
import os
import json
import time
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QLabel,
    QDialog,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QScrollArea,
    QMenu,
    QTableWidgetItem,
    QHeaderView,
    QTextEdit,
    QProgressBar,
    QSizePolicy,
)
from PyQt6.QtGui import QIcon, QPixmap, QAction, QMovie, QPainter, QColor
from PyQt6.QtCore import Qt, QSize, QDateTime, QTimer
from PyQt6.uic import loadUi
from PyQt6.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from fpdf import FPDF
from PIL import Image, ImageFilter
import torch
import warnings
from easy_ocr import OCRExtractor
from clip_and_hybrid_model import clip_and_hybrid_model

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Using a slow image processor.*")
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")


# region PATHS
DEVICE = torch.device("cpu")
GALLERY_UI_PATH = r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\gallery_application.ui"
STYLESHEET_PATH = r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\stylesheet.qss"
LIGHT_BLUE_THEME_PATH = r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\light_blue_theme.qss"
DARK_THEME_PATH = r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\dark_theme.qss"

# Logo and Icons
NEURALJAM_LOGO = (
    r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\NeuralJAM_Logo_60x60.png"
)
LOADING_GIF = r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\loading.gif"

# Model Path
MODEL_PATH = r"C:\Users\ACER\Desktop\Thesis\best_model.pth"

# Hamburger
HAMBURGER_ICON = {
    "black": r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\hamburgerIcon_black.png",
    "white": r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\hamburgerIcon_white.png",
}

# Sidebar Icons
SIDEBAR_ICONS = {
    "dashboard": {
        "black": r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\dashboard_black.png",
        "white": r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\dashboard_white.png",
    },
    "explore": {
        "black": r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\explore_black.png",
        "white": r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\explore_white.png",
    },
    "hate_speech": {
        "black": r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\hatespeechImage_black.png",
        "white": r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\hatespeechImage_white.png",
    },
    "inference": {
        "black": r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\IPO_black.png",
        "white": r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\IPO_white.png",
    },
    "info": {
        "black": r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\information_blue.png",
        "white": r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\information_blue.png",
    },
    "settings": {
        "black": r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\settings_black.png",
        "white": r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\settings_white.png",
    },
}
BACK_ICON = r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\back_black.png"

# Data Files
SCANNED_FOLDERS_FILE = "scanned_folders.json"

# SORT_BY_QSS = "font-size: 16px; color: #000000"
SORT_MENU_QSS = "color: #000000"
USER_GUIDE_QSS = "font-size: 16px"  # Set text color to black
USER_GUIDE_ICON_PATH = (
    r"C:\Users\ACER\Desktop\Thesis\git_Thesis\GalleryApplication\assets\information_blue.png"
)
# endregion


class ProcessingWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, folder_path=None, image_path=None):
        super().__init__()
        # self.detector = detector
        self.folder_path = folder_path
        self.image_path = image_path
        self._is_running = True

        # Add system files to ignore
        self.SYSTEM_FILES = {"desktop.ini", "thumbs.db", ".ds_store"}
        self.ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
        self.easy_ocr = OCRExtractor()
        self.hate_speech_detector = clip_and_hybrid_model()

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            if self.folder_path:
                self.process_folder()
            elif self.image_path:
                self.process_single_image()
        except Exception as e:
            self.error.emit(str(e))

    def process_folder(self):
        results = []
        hate_speech_count = 0
        total_images = 0
        time_start = time.time()

        try:
            # detector = (
            #     self.hate_speech_detector
            # )  # Avoid recreating the detector every time

            for root, _, files in os.walk(self.folder_path):
                for file in files:
                    if not self._is_running:
                        return

                    if file.lower() in self.SYSTEM_FILES:
                        continue

                    file_ext = os.path.splitext(file.lower())[1]
                    if file_ext in self.ALLOWED_EXTENSIONS:
                        file_path = os.path.join(root, file)
                        self.progress.emit(file_path)

                        # OCR text extraction
                        text = self.easy_ocr.extract_text(file_path)

                        # Predict with image + text using adaptive fusion
                        score = self.hate_speech_detector.predict_with_adaptive_fusion(
                            image_path=file_path, text=text
                        )

                        results.append({"path": file_path, "score": score})
                        if score > 0.5:
                            hate_speech_count += 1
                        total_images += 1

            elapsed_time = time.time() - time_start
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            processing_time = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

            print("\n" + "=" * 50)
            print("📊 Processing Summary:")
            print(f"⏱️ Total processing time: {processing_time}")
            print(f"📁 Folder: {os.path.basename(self.folder_path)}")
            print(f"🖼️ Total images processed: {total_images}")
            print(
                f"⚡ Average processing time per image: {elapsed_time / total_images:.2f}s"
                if total_images > 0
                else "N/A"
            )
            print("=" * 50 + "\n")

            self.finished.emit(
                {
                    "folder_path": self.folder_path,
                    "total_images": total_images,
                    "hate_speech_count": hate_speech_count,
                    "results": results,
                }
            )

        except Exception as e:
            elapsed_time = time.time() - time_start
            self.error.emit(
                f"Processing error after {elapsed_time:.1f} seconds: {str(e)}"
            )

    def process_single_image(self):
        # text = self.detector.extract_text(self.image_path)
        text = self.easy_ocr.extract_text(self.image_path)
        # score = self.detector.predict(self.image_path, text)

        score = self.hate_speech_detector.predict_with_adaptive_fusion(
            image_path=self.image_path, text=text
        )

        self.finished.emit(
            {
                "path": self.image_path,
                "score": score,
                "type": "single",
                "date": QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss"),
            }
        )


class ThemeManager:
    def __init__(self, parent):
        self.parent = parent
        self.current_theme = "default"
        self.themes = {
            "default": {
                "qss": STYLESHEET_PATH,
                "active_color": "#3B1E54",  # Purple
                "hover_color": "#9B7EBD",  # Light purple
                "text_color": "#FFFFFF",
            },
            "light_blue": {
                "qss": LIGHT_BLUE_THEME_PATH,
                "active_color": "#4682B4",  # Steel blue
                "hover_color": "#5F9EA0",  # Cadet blue
                "text_color": "#FFFFFF",
            },
            "dark": {
                "qss": DARK_THEME_PATH,
                "active_color": "#444444",  # Dark gray (changed from #5A3D7A)
                "hover_color": "#666666",  # Medium gray (changed from #7A5D9A)
                "text_color": "#FFFFFF",
            },
        }

    def load_theme(self, theme_name):
        if theme_name in self.themes:
            try:
                with open(self.themes[theme_name]["qss"]) as file:
                    self.parent.setStyleSheet(file.read())
                self.current_theme = theme_name
                print(f"Theme changed to: {theme_name}")

                # Update button colors after theme change
                if hasattr(self.parent, "previous_active_index"):
                    self.parent.update_button_colors_when_in_page(
                        self.parent.previous_active_index
                    )

                # ⚠️ Restore sidebar width (critical fix)
                if hasattr(self.parent, "frm_leftSideBar"):
                    if self.parent.state_leftSideBar:
                        self.parent.frm_leftSideBar.setFixedWidth(197)
                    else:
                        self.parent.frm_leftSideBar.setFixedWidth(55)

            except Exception as e:
                print(f"Error loading theme: {e}")
                QMessageBox.critical(
                    self.parent, "Error", f"Could not load theme:\n{e}"
                )

    def get_active_color(self):
        return self.themes[self.current_theme]["active_color"]

    def get_hover_color(self):
        return self.themes[self.current_theme]["hover_color"]

    def get_text_color(self):
        return self.themes[self.current_theme]["text_color"]


class GalleryApplication(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        loadUi(GALLERY_UI_PATH, self)

        # Initialize all data structures from old application
        self.folders = []
        self.folder_data = {}  # Will store folder metadata
        self.image_data = {}  # Will store per-image data including extracted text
        self.hate_speech_images = []
        self.non_hate_speech_images = []
        self.total_scanned_image_files = []
        self.total_scanned_hatespeech_images = []
        self.total_scanned_NOT_hatespeech_images = []
        self.scanned_image_files = []
        self.excluded_files = []
        self.currently_explored_folder_path = []

        """
            Allowed image extensions and ignored system files
        """
        self.ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        self.SYSTEM_FILES = [
            "desktop.ini",
            "thumbs.db",
            ".ds_store",
        ]  # Common system files to ignore

        # Remove User Guide button from sidebar layout
        for i in reversed(range(self.verticalLayout.count())):
            widget = self.verticalLayout.itemAt(i).widget()
            if widget and widget.objectName() == "bttn_sideBar_UserGuide":
                self.verticalLayout.removeWidget(widget)
                widget.deleteLater()
                break

        """
            For
                - App window title
                - Icon
                - Theme manager
        """
        self.setWindowTitle("NeuralJAM")
        self.setWindowIcon(QIcon(NEURALJAM_LOGO))

        # Get screen geometry
        screen = QApplication.primaryScreen()
        screen_size = screen.availableGeometry()

        # Set window size to, for example, 80% of screen size
        width = int(screen_size.width() * 0.8)
        height = int(screen_size.height() * 0.8)
        self.resize(width, height)

        # Optionally center the window
        self.move(
            screen_size.x() + (screen_size.width() - width) // 2,
            screen_size.y() + (screen_size.height() - height) // 2,
        )
        self.theme_manager = ThemeManager(self)

        """
            For hybrid model initialization and other instantiation
        """
        # try:
        #     self.detector = HateSpeechDetector(model_path=MODEL_PATH, device=DEVICE)
        # except Exception as e:
        #     QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
        #     sys.exit(1)

        self.easy_ocr = OCRExtractor()

        # DUMMY BUTTONS
        # Hide example folder buttons
        self.bttn_ScannedFolder_Example_1.setVisible(False)
        self.bttn_ScannedFolder_Example_2.setVisible(False)
        self.bttn_ScannedFolder_Example_3.setVisible(False)

        """
            Sidebar state and theme default setup
        """
        self.state_leftSideBar = False
        self.state_toggle_minfLeftSideBar = True
        self.sidebar_icon_button_mapping = {
            1: {
                "bttn_name": "bttn_sideBar_Dashboard",
                "icon_black": SIDEBAR_ICONS["dashboard"]["black"],  # Inactive
                "icon_white": SIDEBAR_ICONS["dashboard"]["white"],  # Active
            },
            2: {
                "bttn_name": "bttn_sideBar_Explore",
                "icon_black": SIDEBAR_ICONS["explore"]["black"],
                "icon_white": SIDEBAR_ICONS["explore"]["white"],
            },
            3: {
                "bttn_name": "bttn_sideBar_HateSpeechImages",
                "icon_black": SIDEBAR_ICONS["hate_speech"]["black"],
                "icon_white": SIDEBAR_ICONS["hate_speech"]["white"],
            },
            4: {
                "bttn_name": "bttn_sideBar_InferencePipeline",
                "icon_black": SIDEBAR_ICONS["inference"]["black"],
                "icon_white": SIDEBAR_ICONS["inference"]["white"],
            },
            5: {
                "bttn_name": "bttn_sideBar_Explore",
                "icon_black": SIDEBAR_ICONS["explore"]["black"],
                "icon_white": SIDEBAR_ICONS["explore"]["white"],
            },
            7: {
                "bttn_name": "bttn_sideBar_UserGuide",
                "icon_black": SIDEBAR_ICONS["info"]["black"],
                "icon_white": SIDEBAR_ICONS["info"]["white"],
            },
            8: {
                "bttn_name": "bttn_sideBar_Settings",
                "icon_black": SIDEBAR_ICONS["settings"]["black"],
                "icon_white": SIDEBAR_ICONS["settings"]["white"],
            },
        }

        # Add Settings button to sidebar
        self.bttn_sideBar_Settings = QPushButton("   Settings", self.frm_leftSideBar)
        self.bttn_sideBar_Settings.setIcon(QIcon(SIDEBAR_ICONS["settings"]["black"]))
        self.bttn_sideBar_Settings.setIconSize(QSize(20, 20))
        self.bttn_sideBar_Settings.setObjectName("bttn_sideBar_Settings")
        self.bttn_sideBar_Settings.setIcon(
            QIcon(self.sidebar_icon_button_mapping[8]["icon_black"])
        )
        self.bttn_sideBar_Settings.setIconSize(QSize(20, 20))
        self.bttn_sideBar_Settings.setVisible(True)
        self.bttn_sideBar_Settings.clicked.connect(
            self.show_theme_menu
        )  # Connect settings button to show theme menu
        self.verticalLayout.insertWidget(7, self.bttn_sideBar_Settings)

        self.bttn_sideBar_Hamburger.setIcon(QIcon(HAMBURGER_ICON["black"]))
        self.bttn_sideBar_Hamburger.setIconSize(QSize(20, 20))
        self.bttn_sideBar_Hamburger.setVisible(True)

        """
            SIDE BAR FUNCTIONS
        """
        self.bttn_sideBar_Dashboard.clicked.connect(
            lambda: self.pagesStackedWidget.setCurrentIndex(1)
        )
        self.bttn_sideBar_Explore.clicked.connect(
            lambda: self.pagesStackedWidget.setCurrentIndex(2)
        )
        self.bttn_sideBar_HateSpeechImages.clicked.connect(
            lambda: self.pagesStackedWidget.setCurrentIndex(3)
        )
        self.bttn_sideBar_InferencePipeline.clicked.connect(
            lambda: self.pagesStackedWidget.setCurrentIndex(4)
        )

        self.bttn_sideBar_Hamburger.clicked.connect(self.toggle_sidebar)
        self.pagesStackedWidget.currentChanged.connect(
            self.update_button_colors_when_in_page
        )

        """
            FOR UI DESIGN THEME
        """
        self.theme_manager.load_theme("default")

        """
            Welcome Page Part
            
            Initializes pagesStackedWidget index to 0.
            Sets the visiblity of left side bar to False.
        """

        self.frm_leftSideBar.setVisible(
            self.state_leftSideBar
        )  # False, so that left side bar does not show in welcome page
        self.pagesStackedWidget.setCurrentIndex(0)
        self.bttn_GetStarted.clicked.connect(
            self.switch_page_to_dashboard_and_toggle_leftSideBar_visiblity
        )

        # ALIGNMENTS
        self.lbl_NeuralJam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_Tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verticalLayout_Page_Welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_as_lbl_NeuralJAM_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_as_lbl_NeuralJAM_logo.setPixmap(QPixmap(NEURALJAM_LOGO))

        # Set welcome page layout properties
        welcome_container = self.findChild(QWidget, "Page_Welcome")
        if welcome_container:
            welcome_layout = welcome_container.layout()
            if welcome_layout:
                welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                for i in range(welcome_layout.count()):
                    item = welcome_layout.itemAt(i)
                    if item.widget():
                        item.widget().setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Processing Time
        self.processing_timer = QTimer()
        self.processing_timer.setSingleShot(True)
        # self.processing_timer.timeout.connect(self.show_processing_delay_warning)

        # region DASHBOARD
        """
            DASHBOARD
        """
        self.show_hate_speech_chart()  # Show bar chart

        # Add User Guide button to dashboard
        self.dashboard_user_guide_button = QPushButton(
            " User Guide ", self.dshbrd_upper
        )
        self.dashboard_user_guide_button.setObjectName("dashboard_user_guide_button")
        self.dashboard_user_guide_button.setIcon(QIcon(USER_GUIDE_ICON_PATH))
        self.dashboard_user_guide_button.setStyleSheet(USER_GUIDE_QSS)
        self.dashboard_user_guide_button.setIconSize(QSize(18, 18))
        self.horizontalLayout_2.addWidget(self.dashboard_user_guide_button)
        self.dashboard_user_guide_button.clicked.connect(self.show_user_guide)

        # Add sorting dropdown before user guide button
        self.sort_dropdown = QPushButton("Sort By ▼", self.dshbrd_upper)
        self.sort_dropdown.setObjectName("sort_dropdown")
        self.update_sort_dropdown_style()
        # self.sort_dropdown.setStyleSheet(
        #     f"""
        #         font-size: 16px;
        #         color: {"#f70000" if self.theme_manager.current_theme == "light_blue" else "#00ff77" if self.theme_manager.current_theme == "dark" else "#ff009d"};
        #     """
        # )
        self.horizontalLayout_2.insertWidget(
            self.horizontalLayout_2.count() - 1, self.sort_dropdown
        )

        # Create sort menu
        self.sort_menu = QMenu()
        self.sort_menu.addAction(
            "Hate Speech (High to Low)", lambda: self.sort_dashboard_folders("desc")
        )
        self.sort_menu.addAction(
            "Hate Speech (Low to High)", lambda: self.sort_dashboard_folders("asc")
        )
        self.sort_menu.addAction(
            "Most Recent", lambda: self.sort_dashboard_folders("recent")
        )
        self.sort_menu.addAction(
            "Oldest", lambda: self.sort_dashboard_folders("oldest")
        )

        self.sort_dropdown.clicked.connect(
            lambda: self.sort_menu.exec(
                self.sort_dropdown.mapToGlobal(self.sort_dropdown.rect().bottomLeft())
            )
        )

        """
            EXPLORE PAGE FUNCTIONS
        """
        self.bttn_Explore.clicked.connect(self.browse_folders)
        self.setup_initial_label()  # Adds a label if there is no scanned folders yet.
        open("scanned_folders.json", "w").close()
        self.load_scanned_folders()  # Loads the scanned folder.

        # Add User Guide button to explore page
        self.explore_user_guide_button = QPushButton(" User Guide ", self.explr_upper)
        self.explore_user_guide_button.setObjectName("explore_user_guide_button")
        self.explore_user_guide_button.setIcon(QIcon(USER_GUIDE_ICON_PATH))
        self.explore_user_guide_button.setStyleSheet(USER_GUIDE_QSS)
        self.explore_user_guide_button.setIconSize(QSize(18, 18))
        self.horizontalLayout_3.addWidget(self.explore_user_guide_button)
        self.explore_user_guide_button.clicked.connect(self.show_user_guide)

        """
            HATE SPEECH IMAGES PAGE FUNCTIONS
        """
        # Clear the default "Text Label" from sample image labels
        for label_name in [
            "img_HS_Sample",
            "img_HS_Sample_2",
            "img_HS_Sample_3",
            "img_HS_Sample_4",
        ]:
            label = self.findChild(QLabel, label_name)
            if label:
                label.clear()  # This removes the "Text Label" text
                label.setVisible(False)  # This hides the label completely

        # Add User Guide button to hate speech images page
        self.hs_user_guide_button = QPushButton(" User Guide ", self.hsImage_upper)
        self.hs_user_guide_button.setObjectName("hs_user_guide_button")
        self.horizontalLayout_4.addWidget(self.hs_user_guide_button)
        self.hs_user_guide_button.clicked.connect(self.show_user_guide)
        self.hs_user_guide_button.setIcon(QIcon(USER_GUIDE_ICON_PATH))
        self.hs_user_guide_button.setStyleSheet(USER_GUIDE_QSS)
        self.hs_user_guide_button.setIconSize(QSize(18, 18))
        self.bttn_GenerateReport.clicked.connect(self.generate_report)

        """
            INFERENCE PIPELINE PAGE FUNCTIONS
        """
        # Initialize IPO table
        self.ipo_table.setColumnCount(3)
        self.ipo_table.setHorizontalHeaderLabels(["Input", "Process", "Output"])
        self.ipo_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.ipo_table.verticalHeader().setVisible(False)

        # Hide the combo box
        self.comboBox.setVisible(False)  # Add this line to hide the combo box

        # Add User Guide button to IPO page
        self.ipo_user_guide_button = QPushButton(
            " User Guide ", self.inferencePipeline_upper
        )
        self.ipo_user_guide_button.setObjectName("ipo_user_guide_button")
        self.ipo_user_guide_button.setStyleSheet(USER_GUIDE_QSS)
        self.horizontalLayout_8.addWidget(self.ipo_user_guide_button)
        self.ipo_user_guide_button.clicked.connect(self.show_user_guide)
        self.ipo_user_guide_button.setIcon(QIcon(USER_GUIDE_ICON_PATH))
        self.ipo_user_guide_button.setStyleSheet(USER_GUIDE_QSS)
        self.ipo_user_guide_button.setIconSize(QSize(18, 18))

        # region EXPLORED FOLDER
        """
            EXPLORED FOLDER FUNCTIONS
        """
        self.bttn_Back.clicked.connect(self.back_to_folder_overview)
        self.bttn_DeleteThisFolder.clicked.connect(self.delete_current_folder)

        # Add User Guide button to explored folder page
        self.explored_folder_user_guide_button = QPushButton(
            " User Guide ", self.exploredFolder_upper
        )
        self.explored_folder_user_guide_button.setObjectName(
            "explored_folder_user_guide_button"
        )
        self.explored_folder_user_guide_button.setIcon(QIcon(USER_GUIDE_ICON_PATH))
        self.explored_folder_user_guide_button.setStyleSheet(USER_GUIDE_QSS)
        self.explored_folder_user_guide_button.setIconSize(QSize(18, 18))
        self.horizontalLayout_6.addWidget(self.explored_folder_user_guide_button)
        self.explored_folder_user_guide_button.clicked.connect(self.show_user_guide)

        # Add this to __init__ after loading the UI
        self.bttn_Info.setVisible(False)  # Hide the info button

        # In the __init__ method, after loading UI
        self.bttn_Back.setIcon(QIcon(BACK_ICON))
        self.bttn_Back.setIconSize(QSize(20, 20))

        # region END OF INIT

    """
        End of __init__
    """

    def update_sort_dropdown_style(self):
        if not hasattr(self, "sort_dropdown"):
            return  # Avoid error if not yet initialized

        color = (
            "#ffffff"  # for light blue
            if self.theme_manager.current_theme == "light_blue"
            else "#ffffff"  # for dark
            if self.theme_manager.current_theme == "dark"
            else "#000000"  # for default
        )
        self.sort_dropdown.setStyleSheet(
            f"""
                font-size: 16px;
                color: {color};
            """
        )

    def update_sort_menu_style(self):
        if not hasattr(self, "sort_menu"):
            return  # Avoid error if not yet initialized

        self.sort_menu.setStyleSheet(f"""
            QMenu {{
                background-color: {"#F0F8FF" if self.theme_manager.current_theme == "light_blue" else "#2D2D2D" if self.theme_manager.current_theme == "dark" else "#FFFFFF"};
                border: 1px solid {"#BCCCDC" if self.theme_manager.current_theme == "light_blue" else "#666666" if self.theme_manager.current_theme == "dark" else "#D4BEE4"};
            }}
            QMenu::item {{
                padding: 5px 25px 5px 20px;
                color: {"#000000" if self.theme_manager.current_theme == "light_blue" else "#FFFFFF" if self.theme_manager.current_theme == "dark" else "#1e1e1e"};
            }}
            QMenu::item:selected {{
                background-color: {self.theme_manager.get_hover_color()};
                color: {self.theme_manager.get_text_color()};
            }}
        """)

    def show_theme_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
        QMenu {{
            background-color: {"#F0F8FF" if self.theme_manager.current_theme == "light_blue" else "#2D2D2D" if self.theme_manager.current_theme == "dark" else "#FFFFFF"};
            border: 1px solid {"#BCCCDC" if self.theme_manager.current_theme == "light_blue" else "#666666" if self.theme_manager.current_theme == "dark" else "#D4BEE4"};
        }}
        QMenu::item {{
            padding: 5px 25px 5px 20px;
            color: {"#000000" if self.theme_manager.current_theme == "light_blue" else "#FFFFFF" if self.theme_manager.current_theme == "dark" else "#1e1e1e"};
        }}
        QMenu::item:selected {{
            background-color: {self.theme_manager.get_hover_color()};
            color: {self.theme_manager.get_text_color()};
        }}
    """)

        # Add theme actions
        default_theme = QAction("Default Theme", self)
        default_theme.triggered.connect(
            lambda: self.theme_manager.load_theme("default")
        )

        light_blue_theme = QAction("Light Blue Theme", self)
        light_blue_theme.triggered.connect(
            lambda: self.theme_manager.load_theme("light_blue")
        )

        dark_theme = QAction("Dark Theme", self)
        dark_theme.triggered.connect(lambda: self.theme_manager.load_theme("dark"))

        menu.addAction(default_theme)
        menu.addAction(light_blue_theme)
        menu.addAction(dark_theme)

        # Show menu below the button
        button_pos = self.bttn_sideBar_Settings.mapToGlobal(
            self.bttn_sideBar_Settings.rect().bottomLeft()
        )
        menu.exec(button_pos)

    def update_kpi_metrics(self):
        # Use the actual lists that track all images
        total_images = len(self.total_scanned_image_files)
        hate_speech_count = len(self.total_scanned_hatespeech_images)

        self.lbl_num_TotalImages.setText(str(total_images))
        self.lbl_num_TotalHSImages.setText(str(hate_speech_count))
        self.lbl_num_TotalNHSImages.setText(str(total_images - hate_speech_count))

        if total_images > 0:
            percentage = (hate_speech_count / total_images) * 100
            self.lbl_num_TotalHSPercentage.setText(f"{percentage:.1f}%")
        else:
            self.lbl_num_TotalHSPercentage.setText("0%")

        # Update chart with accurate data
        self.show_hate_speech_chart()

    def show_user_guide(self):
        current_page = self.pagesStackedWidget.currentIndex()
        guide_text = ""

        common_style = """
            <style>
                body { font-size: 16px; font-family: Arial, sans-serif; }
                h2 { font-size: 20px; margin-bottom: 10px; }
                ul { margin-top: 0; padding-left: 20px; }
                li { margin-bottom: 5px; }
            </style>
        """

        if current_page == 1:  # Dashboard
            guide_text = f"""
            {common_style}
            <h2>Dashboard User Guide</h2>
            <p>The Dashboard provides an overview of your hate speech detection results:</p>
            <ul>
                <li>View key statistics about scanned images</li>
                <li>See hate speech percentages by folder</li>
                <li>Quickly access your most problematic folders</li>
            </ul>
            """
        elif current_page == 2:  # Explore
            guide_text = f"""
            {common_style}
            <h2>Explore User Guide</h2>
            <p>The Explore module lets you browse all scanned images:</p>
            <ul>
                <li>Click 'Explore' to scan a new folder</li>
                <li>View thumbnails of all images</li>
                <li>Click on folders to see their contents</li>
            </ul>
            """
        elif current_page == 3:  # Hate Speech Images
            guide_text = f"""
            {common_style}
            <h2>Hate Speech Images User Guide</h2>
            <p>This module shows all detected hate speech images:</p>
            <ul>
                <li>View all flagged images in one place</li>
                <li>Use the 'Generate Report' button to create a PDF report</li>
                <li>Images are automatically blurred for safety</li>
            </ul>
            """
        elif current_page == 4:  # IPO Module
            guide_text = f"""
            {common_style}
            <h2>IPO Module User Guide</h2>
            <p>The Input-Process-Output module shows the detection pipeline:</p>
            <ul>
                <li>See the original input images</li>
                <li>View the extracted text and processing steps</li>
                <li>See the final output (blurred if hate speech detected)</li>
            </ul>
            """
        elif current_page == 5:  # Explored Folder
            guide_text = f"""
            {common_style}
            <h2>Explored Folder User Guide</h2>
            <p>This view shows contents of a specific folder:</p>
            <ul>
                <li>See all images in the selected folder</li>
                <li>Use the 'Delete this Folder' button to remove it from scans</li>
                <li>Click 'Back' to return to the folder overview</li>
            </ul>
            """

        self.show_custom_popup("User Guide", guide_text)

    # Globally functions
    # These functions worked across different kinds of pages throughout the application

    # region SIDEBAR
    """
        FOR SIDEBAR
    """

    def toggle_visiblity_leftSideBar(self):
        try:
            print(
                f"Toggling sidebar visibility. Current state: {self.state_leftSideBar}"
            )
            # Toggle visibility
            self.frm_leftSideBar.setVisible(self.state_leftSideBar)

            # Update icons
            self.set_sideBard_icons_to_black()
            if self.state_leftSideBar:  # Only set white icon if sidebar is visible
                self.bttn_sideBar_Dashboard.setIcon(
                    QIcon(self.sidebar_icon_button_mapping[1]["icon_white"])
                )

            # Toggle state
            self.state_leftSideBar = not self.state_leftSideBar
            print(f"Sidebar visibility toggled. New state: {self.state_leftSideBar}")
        except Exception as e:
            print(f"Error toggling sidebar: {str(e)}")
            raise

    # --------------------------------------------------------------------------------------------------------

    def switch_page_to_dashboard_and_toggle_leftSideBar_visiblity(self):
        try:
            print("Attempting to switch to dashboard...")
            # First show the sidebar
            self.state_leftSideBar = True
            self.frm_leftSideBar.setVisible(True)

            # Then switch pages
            self.pagesStackedWidget.setCurrentIndex(1)  # Dashboard
            if self.pagesStackedWidget.currentIndex() == 0:
                self.verticalLayout_Page_Welcome.setAlignment(
                    Qt.AlignmentFlag.AlignCenter
                )

            # Update button colors
            self.update_button_colors_when_in_page(1)
            print("Successfully switched to dashboard")
            self.show_custom_popup(
                title="Message",
                message="""
<span style="font-size:28px;">⚠️</span><br><br>
<b><span style="font-size:20px;">Bar Chart data are only dummy.</span></b><br><br>
<span style="font-size:16px;">Please explore folders first through the <b>"Explore"</b> page.</span><br><br>
<span style="font-size:16px; color: #666666;">This chart is displayed using placeholder values and does not reflect real data until folders are analyzed.</span>
""",
            )
        except Exception as e:
            print(f"Error during page switch: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to switch pages: {str(e)}")

    # --------------------------------------------------------------------------------------------------------

    def set_sideBard_icons_to_black(self):
        self.bttn_sideBar_Dashboard.setIcon(
            QIcon(self.sidebar_icon_button_mapping[1]["icon_black"])
        )
        self.bttn_sideBar_Explore.setIcon(
            QIcon(self.sidebar_icon_button_mapping[2]["icon_black"])
        )
        self.bttn_sideBar_HateSpeechImages.setIcon(
            QIcon(self.sidebar_icon_button_mapping[3]["icon_black"])
        )
        self.bttn_sideBar_InferencePipeline.setIcon(
            QIcon(self.sidebar_icon_button_mapping[4]["icon_black"])
        )
        # This needs to be updated if there is a userguide na
        self.bttn_sideBar_UserGuide.setIcon(
            QIcon(self.sidebar_icon_button_mapping[7]["icon_black"])
        )

    # --------------------------------------------------------------------------------------------------------

    def toggle_sidebar(self):
        """Properly toggle sidebar between expanded and collapsed states"""
        try:
            # Toggle the state
            self.state_leftSideBar = not self.state_leftSideBar

            if self.state_leftSideBar:
                # Expanded state
                self.frm_leftSideBar.setFixedWidth(197)
                self.lbl_sideBar_Icon.setPixmap(QPixmap(NEURALJAM_LOGO))
                # Show text
                self.bttn_sideBar_Hamburger.setIcon(QIcon(HAMBURGER_ICON["black"]))
                self.bttn_sideBar_Dashboard.setText("   Dashboard")
                self.bttn_sideBar_Explore.setText("   Explore")
                self.bttn_sideBar_HateSpeechImages.setText("   Hate Speech Images")
                self.bttn_sideBar_InferencePipeline.setText("   Inference Pipeline")
                self.bttn_sideBar_Settings.setText("   Settings")
            else:
                # Collapsed state
                self.frm_leftSideBar.setFixedWidth(55)
                self.lbl_sideBar_Icon.clear()
                # Hide text but keep icons
                self.bttn_sideBar_Hamburger.setIcon(QIcon(HAMBURGER_ICON["black"]))
                self.bttn_sideBar_Dashboard.setText("")
                self.bttn_sideBar_Explore.setText("")
                self.bttn_sideBar_HateSpeechImages.setText("")
                self.bttn_sideBar_InferencePipeline.setText("")
                self.bttn_sideBar_Settings.setText("")
                self.lbl_sideBar_Icon.setVisible(True)

            # Update button colors
            self.update_button_colors_when_in_page(
                self.pagesStackedWidget.currentIndex()
            )

        except Exception as e:
            print(f"Error toggling sidebar: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to toggle sidebar: {str(e)}")

    # --------------------------------------------------------------------------------------------------------

    def update_button_colors_when_in_page(self, index: int):
        # Get colors from theme manager
        active_color = self.theme_manager.get_active_color()
        text_color = self.theme_manager.get_text_color()

        # Reset styles for all buttons before applying the active color
        for key, button_data in self.sidebar_icon_button_mapping.items():
            button_name = button_data["bttn_name"]
            button = self.findChild(QPushButton, button_name)

            if button:
                # Reset to default style (will use the QSS stylesheet)
                button.setStyleSheet("")
                button.setIcon(QIcon(button_data["icon_black"]))

        # Apply the active color to the button corresponding to the active page
        if index in self.sidebar_icon_button_mapping:
            active_button_name = self.sidebar_icon_button_mapping[index]["bttn_name"]
            active_button = self.findChild(QPushButton, active_button_name)

            if active_button:
                # Apply active styling with a subtle effect
                active_button.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {active_color};
                        color: {text_color};
                        border: 1px solid {self.theme_manager.get_hover_color()};
                    }}
                    QPushButton:hover {{
                        background-color: {self.theme_manager.get_hover_color()};
                    }}
                """)
                active_button.setIcon(
                    QIcon(self.sidebar_icon_button_mapping[index]["icon_white"])
                )

        if index == 8:  # Assuming 8 is your settings page index
            self.bttn_sideBar_Settings.setStyleSheet(f"""
                QPushButton {{
                    background-color: {active_color};
                    color: {text_color};
                    border: 1px solid {self.theme_manager.get_hover_color()};
                }}
                QPushButton:hover {{
                    background-color: {self.theme_manager.get_hover_color()};
                }}
            """)
            self.bttn_sideBar_Settings.setIcon(
                QIcon(self.sidebar_icon_button_mapping[8]["icon_white"])
            )
        else:
            self.bttn_sideBar_Settings.setStyleSheet("")
            self.bttn_sideBar_Settings.setIcon(
                QIcon(self.sidebar_icon_button_mapping[8]["icon_black"])
            )

        self.update_sort_dropdown_style()
        self.update_sort_menu_style()
        # Store the new active index
        self.previous_active_index = index

    # --------------------------------------------------------------------------------------------------------

    # region CHART
    def show_hate_speech_chart(self):
        # Dummy data
        folders = ["Folder A", "Folder B", "Folder C", "Folder D"]
        hate_speech_perc = [35, 50, 20, 60]  # Percentages of hate speech
        not_hate_speech_perc = [65, 50, 80, 40]  # Percentages of not hate speech

        layout = self.frm_chart.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Create figure with better y-axis
        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Set y-axis to show 20, 40, 60, 80, 100
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_ylim(0, 100)

        # Set up bar chart
        x = range(len(folders))
        bar_width = 0.35

        ax.bar(
            [i - bar_width / 2 for i in x],
            hate_speech_perc,
            width=bar_width,
            label="Hate Speech",
            color="#FF0000",  # Purple
        )
        ax.bar(
            [i + bar_width / 2 for i in x],
            not_hate_speech_perc,
            width=bar_width,
            label="Not Hate Speech",
            color="#008000",  # Green
        )

        ax.set_xticks(x)
        ax.set_xticklabels(folders)
        ax.set_ylabel("Hate Speech Percentage (%)")
        ax.set_title("Hate Speech vs Not Hate Speech per Folder in Top 5 Folders")
        ax.legend()

        fig.tight_layout()

        # Add the new chart to the layout
        layout.addWidget(canvas)

    # --------------------------------------------------------------------------------------------------------

    # region EXPLORE PAGE
    """
        EXPLORE PAGE
    """

    def setup_initial_label(self):
        self.info_label = QLabel(
            "No scanned folder yet. Click 'Explore' to scan a folder."
        )
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("""
            font-size: 18px;
            font-family: "EXO Light";
            color: #666666;
        """)
        self.gridLayout_5.addWidget(self.info_label, 0, 0, 1, 3)  # Span 3 columns

        # Add the label to the layout
        layout = self.scrollArea_explr_Folders_Content.layout()
        layout.addWidget(self.info_label)

    # --------------------------------------------------------------------------------------------------------

    def remove_initial_label(self):
        # Remove the label if it's no longer needed
        if hasattr(self, "info_label") and self.info_label:
            print("🗑️ Removing the initial 'No folders' label.")
            layout = self.scrollArea_explr_Folders_Content.layout()
            layout.removeWidget(self.info_label)  # Remove it from the layout
            self.info_label.deleteLater()  # Mark it for deletion (cleans up memory)
            self.info_label = None  # Prevent accidental reuse

    # --------------------------------------------------------------------------------------------------------

    def browse_folders(self):
        """Complete folder/image selection from old app"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Input Type")
        dialog.setMinimumSize(600, 400)

        layout = QVBoxLayout(dialog)

        folder_btn = QPushButton("Select Folder")
        single_btn = QPushButton("Select Single Image")

        # Style and size policies
        for btn in [folder_btn, single_btn]:
            btn.setStyleSheet("font-size: 18px; font-weight: bold;")
            btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            btn.setMinimumHeight(100)  # Optional: gives them a base size

        # Increase font size
        folder_btn.setStyleSheet("font-size: 18px;")
        single_btn.setStyleSheet("font-size: 18px;")

        folder_btn.clicked.connect(lambda: self.handle_folder_selection(dialog))
        single_btn.clicked.connect(lambda: self.handle_single_image_selection(dialog))

        layout.addWidget(folder_btn)
        layout.addWidget(single_btn)

        layout.setSpacing(20)  # Optional: spacing between buttons
        layout.setContentsMargins(20, 20, 20, 20)

        dialog.exec()

    def add_selected_folder_button(self, folder_path):
        layout = self.scrollArea_explr_Folders_Content.layout()

        if layout is None:
            print("⚠️ No layout set on scrollArea_explr_Folders_Content!")
            return

        if not os.path.isdir(folder_path):
            print(f"❌ '{folder_path}' is not a valid folder.")
            return

        folder_name = os.path.basename(folder_path)
        parent = os.path.basename(os.path.dirname(folder_path))
        existing_names = {}

        # Step 1: Scan existing folder buttons
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if isinstance(widget, QPushButton):
                base_text = widget.text().split(" (")[0]
                if base_text not in existing_names:
                    existing_names[base_text] = []
                existing_names[base_text].append(widget)

        # Step 2: Handle duplicate
        if folder_name in existing_names:
            # Update ALL other matching buttons to include parent
            for btn in existing_names[folder_name]:
                if "(" not in btn.text():  # Only update if not already updated
                    old_path = btn.toolTip() if btn.toolTip() else ""
                    old_parent = (
                        os.path.basename(os.path.dirname(old_path))
                        if old_path
                        else "Unknown"
                    )
                    btn.setText(f"{folder_name} ({old_parent})")
            folder_name = f"{folder_name} ({parent})"

        # Step 3: Remove initial label if exists
        self.remove_initial_label()

        # Step 4: Compute row/col
        count = layout.count()
        row = count // 3
        col = count % 3

        button = QPushButton(folder_name)
        button = QPushButton(folder_name)
        button.setProperty("folder_path", folder_path)
        button.setToolTip(folder_path)  # helpful for future lookups
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        button.setStyleSheet("font-size: 19px")
        button.clicked.connect(lambda _, path=folder_path: self.explore_folder(path))

        layout.addWidget(button, row, col)
        print(f"✅ Added button for folder '{folder_name}' at row {row}, col {col}")

        self.save_scanned_folders(folder_path)
        self.update_dashboard_metrics()
        self.update_explore()

    # --------------------------------------------------------------------------------------------------------

    def on_processing_progress(self, file_path):
        if hasattr(self, "progress_bar") and self.progress_bar:
            if not hasattr(self, "total_files_count") or self.total_files_count == 0:
                self.total_files_count = 1

            self.processed_files_count += 1
            percentage = int(
                (self.processed_files_count / self.total_files_count) * 100
            )

            # Smooth animation by moving step-by-step
            current_value = self.progress_bar.value()
            if percentage > current_value:
                self.progress_bar.setValue(min(percentage, 100))

        # Show current filename
        if hasattr(self, "current_file_label") and self.current_file_label:
            filename = os.path.basename(file_path)
            self.current_file_label.setText(f"Processing: {filename}")

    def on_processing_finished(self, folder_result):
        self.processing_timer.stop()
        try:
            folder_path = folder_result["folder_path"]
            total_images = folder_result["total_images"]
            hate_speech_count = folder_result["hate_speech_count"]
            results = folder_result["results"]

            # Update data structures first
            if folder_path not in self.folders:
                self.folders.append(folder_path)

            self.folder_data[folder_path] = {
                "total_images": total_images,
                "hate_speech_count": hate_speech_count,
                "images": [],
            }

            for res in results:
                image_data = {
                    "path": res["path"],
                    "score": res["score"],
                    "type": "folder",
                    "date": QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss"),
                }

                # Update image lists
                if res["score"] > 0.5:
                    if res["path"] not in self.total_scanned_hatespeech_images:
                        self.hate_speech_images.append(image_data)
                        self.total_scanned_hatespeech_images.append(res["path"])
                else:
                    if res["path"] not in self.total_scanned_NOT_hatespeech_images:
                        self.non_hate_speech_images.append(image_data)
                        self.total_scanned_NOT_hatespeech_images.append(res["path"])

                self.folder_data[folder_path]["images"].append(image_data)
                if res["path"] not in self.total_scanned_image_files:
                    self.total_scanned_image_files.append(res["path"])

            # Update all UI elements
            self.update_dashboard_metrics()
            self.populate_images_from_folder(folder_path)
            self.add_selected_folder_button(folder_path)
            self.update_explore()
            self.update_hate_speech_images()
            self.update_ipo_table()

            # Only close loading dialog after ALL updates are complete
            if hasattr(self, "loading_movie"):
                self.loading_movie.stop()
            if hasattr(self, "loading_dialog"):
                self.loading_dialog.close()
            if hasattr(self, "overlay"):
                self.overlay.hide()

            # Clean up worker
            if hasattr(self, "worker") and self.worker is not None:
                if self.worker.isRunning():
                    self.worker.stop()
                    self.worker.wait()
                self.worker.deleteLater()
                self.worker = None

            print(f"✅ Finished processing {folder_path}")

        except Exception as e:
            print(f"Error in on_processing_finished: {str(e)}")
            self.show_custom_popup("Error", f"Failed to process results: {str(e)}")

    def on_processing_error(self, error_message):
        # Stop the loading animation
        if hasattr(self, "loading_movie"):
            self.loading_movie.stop()

        if hasattr(self, "loading_dialog"):
            self.loading_dialog.close()
        if hasattr(self, "overlay"):
            self.overlay.hide()
        self.show_custom_popup("Error", f"Failed to process folder:\n{error_message}")
        # Clean up worker
        if hasattr(self, "worker"):
            self.worker.deleteLater()
            self.worker = None

    def cancel_folder_processing(self):
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.stop()
            self.worker.quit()
            self.worker.wait()

            # Stop the loading animation
            if hasattr(self, "loading_movie"):
                self.loading_movie.stop()

            if hasattr(self, "loading_dialog"):
                self.loading_dialog.close()
            if hasattr(self, "overlay"):
                self.overlay.hide()
        self.show_custom_popup("Cancelled", "Folder processing has been cancelled.")

    # --------------------------------------------------------------------------------------------------------

    def populate_images_from_folder(self, folder_path):
        layout = self.scrllArea_explr_AllImages_Content.layout()
        if layout is None:
            print("⚠️ No layout found on scrllArea_explr_AllImages_Content!")
            return

        self.scanned_image_files = []
        self.excluded_files = []

        # Define valid extensions and system files to ignore
        SYSTEM_FILES = {"desktop.ini", "thumbs.db", ".ds_store"}

        files = os.listdir(folder_path)

        for file_name in files:
            # Skip system files
            if file_name.lower() in SYSTEM_FILES:
                continue

            full_path = os.path.join(folder_path, file_name)
            if os.path.isfile(full_path):
                if any(
                    file_name.lower().endswith(ext)
                    for ext in self.ALLOWED_IMAGE_EXTENSIONS
                ):
                    self.scanned_image_files.append(file_name)
                    if full_path not in self.total_scanned_image_files:
                        self.total_scanned_image_files.append(full_path)
                else:
                    self.excluded_files.append(file_name)

        if self.excluded_files:
            print(f"⚠️ Excluded files (non-image types): {self.excluded_files}")
        else:
            print("✅ All files in this folder are images!")

        if not self.scanned_image_files:
            print("⚠️ No image files found in the selected folder.")
            return

        # Step 1: Store old widgets
        old_widgets = []
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                old_widgets.append(widget)

        # Step 2: Add new images first
        label_width = 300
        label_height = 150

        for i, image_file in enumerate(self.scanned_image_files):
            image_path = os.path.normpath(os.path.join(folder_path, image_file))
            pixmap = QPixmap(image_path)

            # Use the new themed label
            label = self.create_themed_image_label(pixmap)

            count = layout.count()
            row = count // 3
            col = count % 3

            layout.addWidget(label, row, col)

        # Step 3: Re-add old widgets starting after the new ones
        offset = len(self.scanned_image_files)
        for i, widget in enumerate(old_widgets):
            row = (i + offset) // 3
            col = (i + offset) % 3
            layout.addWidget(widget, row, col)

        print(
            f"✅ {len(self.scanned_image_files)} image thumbnails successfully added to the content!"
        )
        total_images = len(self.total_scanned_image_files)
        print(f"🖼️ Total image files: {total_images}")
        self.lbl_num_TotalImages.setText(str(total_images))

    # --------------------------------------------------------------------------------------------------------

    def crop_and_scale_pixmap(self, pixmap, target_width, target_height):
        # Calculate aspect ratios
        source_ratio = pixmap.width() / pixmap.height()
        target_ratio = target_width / target_height

        # Scale to fit while maintaining aspect ratio
        if source_ratio > target_ratio:
            # Image is wider than target
            scaled = pixmap.scaledToWidth(
                target_width, Qt.TransformationMode.SmoothTransformation
            )
        else:
            # Image is taller than target
            scaled = pixmap.scaledToHeight(
                target_height, Qt.TransformationMode.SmoothTransformation
            )

        return scaled

    # --------------------------------------------------------------------------------------------------------

    def save_scanned_folders(self, new_folder):
        save_path = SCANNED_FOLDERS_FILE
        folders = []
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                folders = json.load(f)
        if new_folder not in folders:
            folders.append(new_folder)
            with open(save_path, "w") as f:
                json.dump(folders, f)

    def load_scanned_folders(self):
        save_path = SCANNED_FOLDERS_FILE
        try:
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                with open(save_path, "r") as f:
                    folders = json.load(f)
                    for folder in folders:
                        if os.path.isdir(folder):  # Check if folder still exists
                            self.add_selected_folder_button(folder)
                            self.populate_images_from_folder(folder)
            else:
                # Create empty file if it doesn't exist or is empty
                with open(save_path, "w") as f:
                    json.dump([], f)
        except json.JSONDecodeError:
            print("⚠️ JSON is invalid or corrupted, starting fresh.")
            with open(save_path, "w") as f:
                json.dump([], f)

    def delete_current_folder(self):
        folder_path = getattr(self, "currently_explored_folder_path", None)
        if not folder_path or not os.path.isdir(folder_path):
            print("❌ No valid folder to remove.")
            return

        reply = QMessageBox.question(
            self,
            "Remove Folder",
            "Are you sure you want to remove this folder from the application?\n\n(This will not delete the folder from your computer.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            print("⚠️ Folder removal cancelled.")
            return

        try:
            # Remove from folders list
            if folder_path in self.folders:
                self.folders.remove(folder_path)

            # Remove folder button from explore page
            explore_layout = self.scrollArea_explr_Folders_Content.layout()
            if explore_layout:
                for i in reversed(range(explore_layout.count())):
                    widget = explore_layout.itemAt(i).widget()
                    if isinstance(widget, QPushButton):
                        if widget.property("folder_path") == folder_path:
                            explore_layout.removeWidget(widget)
                            widget.deleteLater()

            self.refill_grid_layout(explore_layout, columns=3)

            # Remove all images from tracking lists
            images_to_remove = [
                img
                for img in self.total_scanned_image_files
                if img.startswith(folder_path)
            ]

            for img in images_to_remove:
                if img in self.total_scanned_image_files:
                    self.total_scanned_image_files.remove(img)
                if img in self.total_scanned_hatespeech_images:
                    self.total_scanned_hatespeech_images.remove(img)
                if img in self.total_scanned_NOT_hatespeech_images:
                    self.total_scanned_NOT_hatespeech_images.remove(img)

            # Remove from folder data
            if folder_path in self.folder_data:
                del self.folder_data[folder_path]

            # Remove from hate speech and non-hate speech lists
            self.hate_speech_images = [
                img
                for img in self.hate_speech_images
                if not img["path"].startswith(folder_path)
            ]
            self.non_hate_speech_images = [
                img
                for img in self.non_hate_speech_images
                if not img["path"].startswith(folder_path)
            ]

            # Remove from saved folders file
            # self.update_saved_folders(folder_path)

            # Update all views
            self.update_dashboard_metrics()
            self.update_explore()
            self.update_hate_speech_images()

            # Clear and rebuild IPO table
            self.ipo_table.setRowCount(0)
            self.update_ipo_table()

            # Return to Explore Page
            self.back_to_folder_overview()

            print(f"✅ Successfully removed folder: {folder_path}")

        except Exception as e:
            print(f"❌ Failed to remove folder: {e}")
            QMessageBox.critical(self, "Error", f"Could not remove folder:\n{e}")

        def update_saved_folders(self, removed_folder):
            """Update the saved folders JSON file after folder removal"""
            try:
                save_path = SCANNED_FOLDERS_FILE
                if os.path.exists(save_path):
                    with open(save_path, "r") as f:
                        folders = json.load(f)

                    # Remove the folder from the list if it exists
                    if removed_folder in folders:
                        folders.remove(removed_folder)

                    # Write back the updated list
                    with open(save_path, "w") as f:
                        json.dump(folders, f)

                    print(
                        f"✅ Successfully removed {removed_folder} from saved folders file"
                    )
            except Exception as e:
                print(f"❌ Error updating saved folders file: {e}")

    def refill_grid_layout(self, layout, columns=3):
        """Compact all widgets in the grid layout to remove empty gaps."""
        widgets = []
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget:
                layout.removeWidget(widget)
                widgets.append(widget)

        # Re-add widgets compactly
        for index, widget in enumerate(reversed(widgets)):
            row = index // columns
            col = index % columns
            layout.addWidget(widget, row, col)

    # --------------------------------------------------------------------------------------------------------

    # region EXPLORED FOLDER
    """
        EXPLORED FOLDER PAGE
    """

    def explore_folder(self, folder_path):
        # Switch to stacked widget index 5
        self.pagesStackedWidget.setCurrentIndex(5)

        # Update the label
        self.lbl_ExploredFolderPath.setText(
            f"  Explore >> {os.path.basename(folder_path)}"
        )

        # Make sure the back button is visible and properly styled
        self.bttn_Back.setVisible(True)  # Explicitly set visibility
        self.bttn_Back.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.1);
            }
        """)

        # Populate images
        self.populate_explored_folder_images(folder_path)

        self.currently_explored_folder_path = folder_path

    # --------------------------------------------------------------------------------------------------------

    def populate_explored_folder_images(self, folder_path):
        layout = self.scrllArea_exploredFolder_Content.layout()
        if layout is None:
            print("⚠️ No layout found on scrllArea_exploredFolder_Content!")
            return

        # Clear old widgets
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        scanned_images = []
        excluded_files = []

        for file_name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file_name)
            if os.path.isfile(full_path):
                if any(
                    file_name.lower().endswith(ext)
                    for ext in self.ALLOWED_IMAGE_EXTENSIONS
                ):
                    scanned_images.append(full_path)
                else:
                    excluded_files.append(file_name)

        if excluded_files:
            excluded_list_str = "\n".join(f"- {file}" for file in excluded_files)
            self.show_custom_popup(
                "Excluded non-image files",
                f"The following files were excluded:\n\n{excluded_list_str}",
            )

        if not scanned_images:
            print("⚠️ No valid images found.")
            return

        label_width = 300
        label_height = 150

        for i, image_path in enumerate(scanned_images):
            print(f"🖼️ Exploring: {image_path}")

            # Check if this is a hate speech image
            is_hate_speech = any(
                img["path"] == image_path and img["score"] > 0.5
                for img in self.hate_speech_images
            )

            if is_hate_speech:
                # Process with blur
                pixmap = self.process_image_for_display(image_path, blur=True)
            else:
                # Process without blur
                pixmap = self.process_image_for_display(image_path, blur=False)

            pixmap = self.crop_and_scale_pixmap(pixmap, label_width, label_height)

            label = QLabel()
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background-color: transparent")

            row = i // 3
            col = i % 3
            layout.addWidget(label, row, col)

        print(f"✅ {len(scanned_images)} images shown in explorer view.")

    # --------------------------------------------------------------------------------------------------------

    def back_to_folder_overview(self):
        self.pagesStackedWidget.setCurrentIndex(2)

    # --------------------------------------------------------------------------------------------------------

    """
        POP UP DIALOGUE BOXES
    """

    def show_custom_popup(self, title, message):
        # Create overlay
        self.overlay = QWidget(self)
        self.overlay.setGeometry(self.rect())  # Full window size
        self.overlay.setStyleSheet(
            "background-color: rgba(0, 0, 0, 120); font-size: 16px;"
        )
        self.overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.overlay.show()

        # Create modal dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)
        dialog.setFixedSize(600, 400)

        main_layout = QVBoxLayout()  # Create layout first

        # Scrollable message area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        lbl_message = QLabel(message)
        lbl_message.setText(message)  # Pass the HTML content here
        lbl_message.setWordWrap(True)
        content_layout.addWidget(lbl_message)

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        # OK button
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(dialog.accept)
        main_layout.addWidget(btn_ok)

        dialog.setLayout(main_layout)

        # Apply theme and font size
        # theme = ThemeManager(self).current_theme
        dialog.setStyleSheet("""
            * {
                font-family: "EXO Light";
                font-size: 20px;
            }
        """)

        # Show dialog
        dialog.exec()

        self.overlay.hide()

    def show_loading_dialog(self):
        self.overlay = QWidget(self)
        self.overlay.setGeometry(self.rect())
        self.overlay.setStyleSheet("background-color: rgba(0, 0, 0, 120);")
        self.overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.overlay.show()

        self.loading_dialog = QDialog(self)
        self.loading_dialog.setWindowTitle("Processing...")
        self.loading_dialog.setModal(True)
        self.loading_dialog.setFixedSize(400, 300)

        layout = QVBoxLayout(self.loading_dialog)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Loading GIF - Store the movie as an instance variable
        self.loading_label = QLabel(self.loading_dialog)
        self.loading_movie = QMovie(LOADING_GIF)  # Store as instance variable
        self.loading_label.setMovie(self.loading_movie)
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.loading_label)

        # Progress label
        self.progress_label = QLabel("Processing files...", self.loading_dialog)
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)

        # Add progress bar
        self.progress_bar = QProgressBar(self.loading_dialog)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #D4BEE4;
                width: 10px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Add current file label
        self.current_file_label = QLabel("Preparing...", self.loading_dialog)
        self.current_file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.current_file_label)

        # Cancel Button
        btn_cancel = QPushButton("Cancel", self.loading_dialog)
        btn_cancel.setFixedWidth(100)
        btn_cancel.clicked.connect(self.cancel_folder_processing)
        layout.addWidget(btn_cancel, alignment=Qt.AlignmentFlag.AlignCenter)

        self.loading_dialog.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
            }
            QPushButton {
                background-color: #D4BEE4;
                color: #1e1e1e;
                padding: 6px 12px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #9B7EBD;
            }
        """)

        # Start the animation
        self.loading_movie.start()
        self.loading_dialog.show()

    # ----------------------------------------------------------------------------------------------------------------------
    def process_image_for_display(self, image_path, blur=False):
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            if blur:
                # Create a more subtle blur effect
                image = image.filter(ImageFilter.GaussianBlur(radius=5))

            # Resize for display while maintaining aspect ratio
            max_size = (300, 300)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

            temp_path = "temp_processed.jpg"
            image.save(temp_path, "JPEG", quality=85)
            return QPixmap(temp_path)
        except Exception as e:
            print(f"Error processing image: {e}")
            # Return a placeholder image if processing fails
            placeholder = QPixmap(300, 300)
            placeholder.fill(Qt.GlobalColor.gray)
            painter = QPainter(placeholder)
            painter.drawText(
                placeholder.rect(), Qt.AlignmentFlag.AlignCenter, "Image Error"
            )
            painter.end()
            return placeholder

    def handle_folder_selection(self, dialog):
        dialog.close()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.process_folder(folder_path)

    def handle_single_image_selection(self, dialog):
        dialog.close()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.process_single_image(file_path)

    def process_folder(self, folder_path):
        try:
            # Count images in folder first
            image_count = sum(
                1
                for root, _, files in os.walk(folder_path)
                for f in files
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            )

            if image_count > 100:
                self.show_custom_popup(
                    "Too Many Images",
                    "Your folder contains over 100 images. \n"
                    + "For optimal performance, please select a folder with fewer images.\n\n"
                    + f"Current image count: {image_count}",
                )
                return

            # Check if folder already exists
            if folder_path in self.folders:
                reply = QMessageBox.question(
                    self,
                    "Folder Already Scanned",
                    "This folder has already been scanned. Would you like to rescan it?\n"
                    + "This will update all content and remove old data.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    # Remove all old data for this folder
                    self.remove_folder_data(folder_path)
                else:
                    return

            # Check if folder contains any valid images first
            has_images = False
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        has_images = True
                        break
                if has_images:
                    break

            if not has_images:
                self.show_custom_popup(
                    "No Images Found",
                    "The selected folder does not contain any supported image files.\n\n"
                    "Supported formats: PNG, JPG, JPEG, BMP",
                )
                return

            # Initialize progress tracking
            self.processed_files_count = 0
            self.total_files_count = sum(
                1
                for root, _, files in os.walk(folder_path)
                for f in files
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            )

            # Show loading dialog
            self.show_loading_dialog()

            # Clean up any existing worker properly
            if hasattr(self, "worker"):
                if self.worker is not None:
                    if self.worker.isRunning():
                        self.worker.stop()
                        self.worker.wait()
                    self.worker.deleteLater()
                self.worker = None  # Clear the reference

            # Create new worker
            self.worker = ProcessingWorker(folder_path=folder_path)

            # Connect signals
            self.worker.progress.connect(self.on_processing_progress)
            self.worker.finished.connect(self.on_processing_finished)
            self.worker.error.connect(self.on_processing_error)

            self.processing_timer.start(5000)
            self.worker.start()

            # Update IPO table for each image
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(root, file)
                        row = self.ipo_table.rowCount()
                        self.ipo_table.insertRow(row)
                        self.add_ipo_row(image_path, folder_path)

        except Exception as e:
            if hasattr(self, "loading_dialog"):
                self.loading_dialog.close()
            if hasattr(self, "overlay"):
                self.overlay.hide()
            self.show_custom_popup("Error", f"Failed to process folder: {str(e)}")

    def process_single_image(self, image_path):
        try:
            self.show_loading_dialog()

            # Clean up any existing worker properly
            if hasattr(self, "worker"):
                if self.worker is not None:
                    if self.worker.isRunning():
                        self.worker.stop()
                        self.worker.wait()
                    self.worker.deleteLater()
                self.worker = None  # Clear the reference

            # Create and start worker for single image
            self.worker = ProcessingWorker(image_path=image_path)
            self.worker.finished.connect(self.on_single_image_finished)
            self.worker.error.connect(self.on_processing_error)
            self.processing_timer.start(5000)
            self.worker.start()

        except Exception as e:
            if hasattr(self, "loading_dialog"):
                self.loading_dialog.close()
            if hasattr(self, "overlay"):
                self.overlay.hide()
            self.show_custom_popup("Error", f"Failed to process image: {str(e)}")

    def update_dashboard_metrics(self):
        # First, clear the existing folder frames
        layout = self.verticalLayout_9
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Create a special "Single Uploads" category for non-folder images
        single_images = [
            img
            for img in self.hate_speech_images + self.non_hate_speech_images
            if img["type"] == "single"
        ]
        if single_images:
            single_hs_count = len([img for img in single_images if img["score"] > 0.5])
            total_singles = len(single_images)
            percentage = (
                (single_hs_count / total_singles) * 100 if total_singles > 0 else 0
            )

            single_btn = QPushButton(
                f"Single Uploads\n"
                f"Hate Speech: {single_hs_count}/{total_singles}\n"
                f"({percentage:.1f}%)"
            )

        # Update KPI metrics
        total_images = len(self.total_scanned_image_files)
        hate_speech_count = len(self.total_scanned_hatespeech_images)

        self.lbl_num_TotalImages.setText(str(total_images))
        self.lbl_num_TotalHSImages.setText(str(hate_speech_count))
        self.lbl_num_TotalNHSImages.setText(str(total_images - hate_speech_count))

        if total_images > 0:
            percentage = (hate_speech_count / total_images) * 100
            self.lbl_num_TotalHSPercentage.setText(f"{percentage:.1f}%")
        else:
            self.lbl_num_TotalHSPercentage.setText("0%")

        # Add top 5 folders as buttons
        if not self.folders:
            no_folder_label = QLabel("No folders scanned yet")
            no_folder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.verticalLayout_9.addWidget(no_folder_label)
        else:
            # Sort folders by hate speech percentage (highest first)
            sorted_folders = sorted(
                self.folders,
                key=lambda f: (
                    len(
                        [
                            img
                            for img in self.hate_speech_images
                            if img["path"].startswith(f)
                        ]
                    )
                    / len(
                        [
                            img
                            for img in self.total_scanned_image_files
                            if img.startswith(f)
                        ]
                    )
                )
                if len(
                    [img for img in self.total_scanned_image_files if img.startswith(f)]
                )
                > 0
                else 0,
                reverse=True,
            )[:5]  # Take top 5

            # Group folders by name to detect duplicates
            folder_groups = {}
            for folder in sorted_folders:
                name = os.path.basename(folder)
                if name not in folder_groups:
                    folder_groups[name] = []
                folder_groups[name].append(folder)

            # For folders with same names, add parent folder info
            for folder in sorted_folders:
                folder_name = os.path.basename(folder)
                if len(folder_groups[folder_name]) > 1:
                    # Add parent folder name for disambiguation
                    parent = os.path.basename(os.path.dirname(folder))
                    display_name = f"{folder_name} ({parent})"
                else:
                    display_name = folder_name

                hs_count = len(
                    [
                        img
                        for img in self.hate_speech_images
                        if img["path"].startswith(folder)
                    ]
                )
                total = len(
                    [
                        img
                        for img in self.total_scanned_image_files
                        if img.startswith(folder)
                    ]
                )
                percentage = (hs_count / total) * 100 if total > 0 else 0

                btn = QPushButton(
                    f"{display_name}\n"
                    f"Hate Speech: {hs_count}/{total}\n"
                    f"({percentage:.1f}%)"
                )

                # Theme-aware styling
                if self.theme_manager.current_theme == "dark":
                    high_risk = "#8B0000"  # Dark red
                    medium_risk = "#8B4513"  # Dark orange
                    low_risk = "#006400"  # Dark green
                    text_color = "#FFFFFF"  # White text
                elif self.theme_manager.current_theme == "light_blue":
                    high_risk = "#FF6B6B"  # Light red
                    medium_risk = "#FFA07A"  # Light orange
                    low_risk = "#90EE90"  # Light green
                    text_color = "#000000"  # Black text
                else:  # Default theme
                    high_risk = "#FF9C9C"  # Original red
                    medium_risk = "#FFD59C"  # Original orange
                    low_risk = "#9CFF9C"  # Original green
                    text_color = "#000000"  # Black text

                # Style based on percentage with theme awareness
                if percentage > 50:
                    btn.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {high_risk};
                            border-radius: 10px;
                            color: {text_color};
                            padding: 10px;
                        }}
                        QPushButton:hover {{
                            background-color: {high_risk if self.theme_manager.current_theme == "dark" else "#FF4444"};
                        }}
                    """)
                elif percentage > 20:
                    btn.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {medium_risk};
                            border-radius: 10px;
                            color: {text_color};
                            padding: 10px;
                        }}
                        QPushButton:hover {{
                            background-color: {medium_risk if self.theme_manager.current_theme == "dark" else "#FF9966"};
                        }}
                    """)
                else:
                    btn.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {low_risk};
                            border-radius: 10px;
                            color: {text_color};
                            padding: 10px;
                        }}
                        QPushButton:hover {{
                            background-color: {low_risk if self.theme_manager.current_theme == "dark" else "#77DD77"};
                        }}
                    """)

                btn.clicked.connect(lambda _, f=folder: self.explore_folder(f))
                self.verticalLayout_9.addWidget(btn)

        # Update chart with accurate data
        self.update_dashboard_chart()

    def update_folder_buttons(self):
        # Clear existing folder frames
        for i in reversed(range(self.verticalLayout_9.count())):
            widget = self.verticalLayout_9.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Get top 5 folders sorted by hate speech percentage
        folder_stats = []
        for folder_path, data in self.folder_data.items():
            total = data["total_images"]
            if total > 0:
                percentage = (data["hate_speech_count"] / total) * 100
                folder_stats.append(
                    {
                        "path": folder_path,
                        "name": os.path.basename(folder_path),
                        "percentage": percentage,
                        "count": data["hate_speech_count"],
                    }
                )

        folder_stats.sort(key=lambda x: x["percentage"], reverse=True)
        top_folders = folder_stats[:5]

        # Add buttons for top folders
        for i, folder in enumerate(top_folders):
            button = QPushButton(
                f"{folder['name']}\n"
                f"Hate Speech: {folder['count']}/{self.folder_data[folder['path']]['total_images']}\n"
                f"({folder['percentage']:.1f}%)"
            )
            button.setStyleSheet("""
                QPushButton {
                    background-color: #FF9C9C;
                    border-radius: 10px;
                    padding: 10px;
                    text-align: center;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #FF7C7C;
                }
            """)
            button.clicked.connect(
                lambda _, path=folder["path"]: self.explore_folder(path)
            )
            self.verticalLayout_9.addWidget(button)

        # Update chart
        self.update_dashboard_chart()

    def update_dashboard_chart(self):
        """Complete dashboard chart update from old app"""
        # Get the existing layout from the frame
        layout = self.frm_chart.layout()
        if layout is None:
            print("⚠️ frm_chart has no layout set!")
            return

        # Remove existing widgets (previous canvas)
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Prepare data - get top 5 folders by hate speech percentage
        folder_stats = []
        for folder_path, data in self.folder_data.items():
            if data["total_images"] > 0:
                percentage = (data["hate_speech_count"] / data["total_images"]) * 100
                folder_stats.append(
                    {
                        "name": os.path.basename(folder_path),
                        "percentage": percentage,
                        "total": data["total_images"],
                        "hate_count": data["hate_speech_count"],
                    }
                )

        # Sort by percentage and take top 5
        folder_stats.sort(key=lambda x: x["percentage"], reverse=True)
        top_folders = folder_stats[:5]

        if not top_folders:
            # No data case
            no_data_label = QLabel(
                "No folder data available. Scan a folder to see statistics."
            )
            no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(no_data_label)
            return

        # Create the bar chart
        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Data for chart
        folder_names = [f["name"] for f in top_folders]
        hate_percent = [f["percentage"] for f in top_folders]
        not_hate_percent = [100 - p for p in hate_percent]

        # Set up bar chart
        x = range(len(folder_names))
        bar_width = 0.35

        ax.bar(
            [i - bar_width / 2 for i in x],
            hate_percent,
            width=bar_width,
            label="Hate Speech",
            color="#E74C3C",
        )
        ax.bar(
            [i + bar_width / 2 for i in x],
            not_hate_percent,
            width=bar_width,
            label="Not Hate Speech",
            color="#2ECC71",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(folder_names)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Hate Speech Distribution in Top 5 Folders")
        ax.legend()

        fig.tight_layout()

        # Add the new chart to the layout
        layout.addWidget(canvas)

    def update_explore(self):
        """Complete explore page update from old app"""
        # Clear existing content
        layout = self.scrllArea_explr_AllImages_Content.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Combine all images and sort by date (newest first)
        all_images = []
        for folder in self.folders:
            all_images.extend(self.folder_data[folder]["images"])
        all_images.extend(
            [
                img
                for img in self.hate_speech_images + self.non_hate_speech_images
                if img["type"] == "single"
            ]
        )

        # Sort by date (newest first)
        all_images.sort(key=lambda x: x["date"], reverse=True)

        # Add images to layout
        label_width = 300
        label_height = 200

        # region BLUR THRESHOLD
        for i, image_data in enumerate(all_images):
            image_path = image_data["path"]
            print(f"🖼️ Exploring: {image_path}")
            try:
                # Only blur if it's hate speech
                pixmap = self.process_image_for_display(
                    image_path,
                    blur=image_data["score"]
                    > 0.5,  # Only blur if score > 0.5 ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
                )
                pixmap = self.crop_and_scale_pixmap(pixmap, label_width, label_height)

                label = QLabel()
                label.setPixmap(pixmap)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setStyleSheet("background-color: transparent")

                # Add tooltip with prediction info
                tooltip = (
                    f"Score: {image_data['score']:.2f}\nDate: {image_data['date']}"
                )
                if image_data["type"] == "folder":
                    tooltip = (
                        f"Folder: {os.path.basename(os.path.dirname(image_path))}\n"
                        + tooltip
                    )
                label.setToolTip(tooltip)

                row = i // 3
                col = i % 3
                layout.addWidget(label, row, col)

            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
                error_label = QLabel(f"Error loading\n{os.path.basename(image_path)}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                error_label.setStyleSheet("color: red;")
                row = i // 3
                col = i % 3
                layout.addWidget(error_label, row, col)

    def update_hate_speech_images(self):
        layout = self.gridLayout_2

        for label_name in [
            "img_HS_Sample",
            "img_HS_Sample_2",
            "img_HS_Sample_3",
            "img_HS_Sample_4",
        ]:
            label = self.findChild(QLabel, label_name)
            if label:
                label.setText("")

        # Clear existing widgets including the default "Text Label"
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        if not self.hate_speech_images:
            # Create a more professional empty state message
            empty_label = QLabel("No hate speech images detected yet")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setStyleSheet("""
                QLabel {
                    color: #666666;
                    font-size: 20px;
                    font-family: Arial;
                    padding: 20px;
                    background: transparent;
                }
            """)
            empty_label.setVisible(True)
            layout.addWidget(empty_label, 0, 0, 1, 4)  # Span 4 columns
            return

        # Sort by most recent first
        sorted_images = sorted(
            self.hate_speech_images, key=lambda x: x["date"], reverse=True
        )

        # Add images in grid (4 columns)
        for i, image_data in enumerate(sorted_images):
            row = i // 4
            col = i % 4

            try:
                # Process image with blur
                pixmap = self.process_image_for_display(image_data["path"], blur=True)
                # Use themed label with smaller size for hate speech images
                label = self.create_themed_image_label(pixmap, size=(200, 200))
                layout.addWidget(label, row, col)
            except Exception as e:
                error_label = QLabel("Error loading image")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(error_label, row, col)

    # region GENERATE REPORT
    def generate_report(self):
        try:
            pdf = FPDF()
            pdf.add_page()

            # Fix logo path and specify format
            pdf.image(NEURALJAM_LOGO, x=10, y=8, w=30, type="PNG")
            pdf.set_font("Arial", "B", 16)
            pdf.cell(
                0, 10, "NeuralJAM Hate Speech Detection Report", ln=True, align="C"
            )

            # Add date
            pdf.set_font("Arial", "", 12)
            pdf.cell(
                0,
                10,
                f"Report generated on: {QDateTime.currentDateTime().toString('yyyy-MM-dd HH:mm:ss')}",
                ln=True,
                align="C",
            )
            pdf.ln(10)

            # Add Dashboard Statistics Section
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Dashboard Overview", ln=True)
            pdf.set_font("Arial", "", 12)

            # Overall Statistics
            total_images = len(self.total_scanned_image_files)
            hate_speech_count = len(self.total_scanned_hatespeech_images)
            non_hate_speech_count = total_images - hate_speech_count
            hate_speech_percentage = (
                (hate_speech_count / total_images * 100) if total_images > 0 else 0
            )

            pdf.cell(0, 10, f"Total Images Analyzed: {total_images}", ln=True)
            pdf.cell(0, 10, f"Hate Speech Images: {hate_speech_count}", ln=True)
            pdf.cell(0, 10, f"Non-Hate Speech Images: {non_hate_speech_count}", ln=True)
            pdf.cell(
                0,
                10,
                f"Overall Hate Speech Percentage: {hate_speech_percentage:.1f}%",
                ln=True,
            )
            pdf.ln(5)

            # Top 5 Folders Section
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Top 5 Folders by Hate Speech Percentage", ln=True)
            pdf.set_font("Arial", "", 12)

            # Sort folders by hate speech percentage
            folder_stats = []
            for folder in self.folders:
                folder_images = [
                    img
                    for img in self.total_scanned_image_files
                    if img.startswith(folder)
                ]
                folder_hate_speech = [
                    img
                    for img in self.total_scanned_hatespeech_images
                    if img.startswith(folder)
                ]
                if folder_images:
                    percentage = (len(folder_hate_speech) / len(folder_images)) * 100
                    folder_stats.append(
                        {
                            "name": os.path.basename(folder),
                            "total": len(folder_images),
                            "hate_count": len(folder_hate_speech),
                            "percentage": percentage,
                        }
                    )

            # Sort by percentage and take top 5
            folder_stats.sort(key=lambda x: x["percentage"], reverse=True)
            for folder in folder_stats[:5]:
                pdf.cell(0, 10, f"{folder['name']}", ln=True)
                pdf.cell(0, 10, f"    Total Images: {folder['total']}", ln=True)
                pdf.cell(
                    0, 10, f"    Hate Speech Images: {folder['hate_count']}", ln=True
                )
                pdf.cell(0, 10, f"    Percentage: {folder['percentage']:.1f}%", ln=True)
                pdf.ln(3)

            pdf.ln(10)

            # Your existing Folder Uploads section
            folder_hs = [
                f for f in self.folders if self.folder_data[f]["hate_speech_count"] > 0
            ]
            if folder_hs:
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Folder Uploads", ln=True)
                pdf.set_font("Arial", "", 12)

                for folder in folder_hs:
                    pdf.cell(0, 10, f"Folder: {os.path.basename(folder)}", ln=True)
                    pdf.cell(0, 10, f"Path: {folder}", ln=True)
                    pdf.cell(
                        0,
                        10,
                        f"Hate Speech Images: {self.folder_data[folder]['hate_speech_count']}/{self.folder_data[folder]['total_images']}",
                        ln=True,
                    )
                    pdf.ln(5)

                    # Add images from this folder
                    for root, _, files in os.walk(folder):
                        for file in files:
                            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                                img_path = os.path.join(root, file)
                                if img_path in [
                                    hs["path"]
                                    for hs in self.hate_speech_images
                                    if hs["type"] == "folder"
                                ]:
                                    # Find the image data to get score
                                    img_data = next(
                                        hs
                                        for hs in self.hate_speech_images
                                        if hs["path"] == img_path
                                    )
                                    pdf.cell(
                                        0,
                                        10,
                                        f"Image: {file} (Score: {img_data['score']:.2f})",
                                        ln=True,
                                    )
                                    pdf.image(img_path, x=10, y=pdf.get_y(), w=50)
                                    pdf.ln(60)  # Space for image

                    pdf.ln(10)

            # Your existing Single Image Uploads section
            single_hs = [hs for hs in self.hate_speech_images if hs["type"] == "single"]
            if single_hs:
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Single Image Uploads", ln=True)
                pdf.set_font("Arial", "", 12)

                for img in single_hs:
                    pdf.cell(
                        0,
                        10,
                        f"Image: {os.path.basename(img['path'])} (Score: {img['score']:.2f})",
                        ln=True,
                    )
                    pdf.image(img["path"], x=10, y=pdf.get_y(), w=50)
                    pdf.ln(60)  # Space for image
                    pdf.cell(0, 10, f"Uploaded: {img['date']}", ln=True)
                    pdf.ln(10)

            # Save PDF
            report_path = os.path.join(os.getcwd(), "hate_speech_report.pdf")
            pdf.output(report_path)
            self.show_custom_popup(
                "Success", f"Report generated successfully at:\n{report_path}"
            )

        except Exception as e:
            self.show_custom_popup("Error", f"Failed to generate report:\n{str(e)}")

    # region IPO
    def update_ipo_table(self):
        """Complete IPO table update with simplified grid styling"""
        self.ipo_table.setRowCount(0)

        # Configure table properties
        self.ipo_table.setColumnCount(3)
        self.ipo_table.setHorizontalHeaderLabels(["Input", "Process", "Output"])

        # Use built-in grid styling
        self.ipo_table.setShowGrid(True)
        self.ipo_table.setStyleSheet("""
            QHeaderView::section {
                background-color: #F8FAFC;
                color: #000000;
                font-family: "Jura";
                font-size: 18px;
                font-weight: bold;
                padding: 4px;
            }
        """)

        # Configure column resize modes
        self.ipo_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed
        )
        self.ipo_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.ipo_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Fixed
        )

        # Set column widths
        self.ipo_table.setColumnWidth(0, 250)  # Input column width
        self.ipo_table.setColumnWidth(2, 250)  # Output column width
        self.ipo_table.setWordWrap(True)  # Enable text wrapping

        # Collect all image data
        all_images = []
        for folder in self.folders:
            all_images.extend(self.folder_data[folder]["images"])

        # Add single images not in folders
        all_images.extend(
            [
                img
                for img in self.hate_speech_images + self.non_hate_speech_images
                if img["type"] == "single"
            ]
        )

        # Sort by date (newest first) by default
        all_images.sort(key=lambda x: x["date"], reverse=True)

        # Populate table
        for img_data in all_images:
            row = self.ipo_table.rowCount()
            self.ipo_table.insertRow(row)
            self.ipo_table.setRowHeight(row, 250)

            # Input column - original image
            input_container = QWidget()
            input_layout = QVBoxLayout(input_container)
            input_layout.setContentsMargins(5, 5, 5, 5)
            input_container.setStyleSheet("background-color: #F8FAFC;")

            input_label = QLabel()
            try:
                pixmap = QPixmap(img_data["path"])
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(240, 240, Qt.AspectRatioMode.KeepAspectRatio)
                    input_label.setPixmap(pixmap)
                    input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            except Exception as e:
                input_label.setText(f"Error loading image\n{str(e)}")

            input_layout.addWidget(input_label)
            self.ipo_table.setCellWidget(row, 0, input_container)

            # Process column - information
            process_widget = QWidget()
            process_layout = QVBoxLayout(process_widget)
            process_widget.setStyleSheet("background-color: #F8FAFC;")

            process_text_edit = QLabel()
            process_text_edit.setWordWrap(True)
            process_text_edit.setStyleSheet("""
                QLabel {
                    background-color: #F8FAFC;
                    font-family: "Jura";
                    font-size: 16px;
                    color: #000000;
                    padding: 10px;
                    line-height: 1.5;
                }
            """)

            # Add the formatted text
            process_text = (
                "PROCESSING INFORMATION:\n"
                f"📄 File: {os.path.basename(img_data['path'])}\n"
            )

            if img_data["type"] == "folder":
                process_text += f"📁 Folder: {os.path.basename(os.path.dirname(img_data['path']))}\n\n"

            # text = self.detector.extract_text(img_data["path"])
            text = self.easy_ocr.extract_text(img_data["path"])

            # process_text += f"📝 Extracted Text:\n{text}\n\n"

            process_text += (
                "TEXT EXTRACTION:\n"
                "🔍 Performing OCR text extraction...\n"
                f"📝 Extracted Text:\n{text}\n\n"
                "HATE SPEECH ANALYSIS:\n"
                f"📊 Prediction Score: {img_data['score']:.4f}\n"
                f"🚨 Conclusion: {'HATE SPEECH DETECTED' if img_data['score'] > 0.5 else 'No hate speech detected'}\n"
                f"\n📍 Path: {img_data['path']}"
            )
            process_text_edit.setText(process_text)

            process_layout.addWidget(process_text_edit)
            process_layout.setContentsMargins(0, 0, 0, 0)
            self.ipo_table.setCellWidget(row, 1, process_widget)

            # Output column - blurred if hate speech
            output_widget = QWidget()
            output_layout = QVBoxLayout(output_widget)
            output_layout.setContentsMargins(5, 5, 5, 5)
            output_widget.setStyleSheet("background-color: #F8FAFC;")

            # Status Label
            is_hate = img_data["score"] > 0.5
            status_label = QLabel("🚨 HATE SPEECH" if is_hate else "✅ CLEAN")
            status_label.setFixedHeight(20)
            status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_label.setStyleSheet(
                f"""
                background-color: #F8FAFC;
                font-weight: bold;
                font-family: "Jura";
                color: {"red" if is_hate else "green"};
                font-size: 16px;
                padding: 3px;
                """
            )
            output_layout.addWidget(status_label)

            # Image Label
            output_label = QLabel()
            try:
                if is_hate:
                    image = Image.open(img_data["path"])
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
                    blurred_image.save("temp_blurred.jpg")
                    output_pixmap = QPixmap("temp_blurred.jpg")
                else:
                    output_pixmap = QPixmap(img_data["path"])

                output_pixmap = output_pixmap.scaled(
                    240, 240, Qt.AspectRatioMode.KeepAspectRatio
                )
                output_label.setPixmap(output_pixmap)
                output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            except Exception as e:
                output_label.setText(f"Error loading image\n{str(e)}")

            output_layout.addWidget(output_label)
            self.ipo_table.setCellWidget(row, 2, output_widget)

            # Set row height based on content
            self.ipo_table.setRowHeight(row, process_widget.sizeHint().height() + 10)

    def add_ipo_row(self, image_path, folder=None):
        try:
            row = self.ipo_table.rowCount()
            self.ipo_table.insertRow(row)

            # Set row height to accommodate the image
            self.ipo_table.setRowHeight(row, 220)

            # Input column - larger and centered
            input_label = QLabel()
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                raise ValueError("Invalid image file")

            # Scale image to fit but maintain aspect ratio
            pixmap = pixmap.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio)
            input_label.setPixmap(pixmap)
            input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            input_label.setStyleSheet("padding: 10px;")
            self.ipo_table.setCellWidget(row, 0, input_label)

            # Process column - more structured log
            process_log = QTextEdit()
            process_log.setReadOnly(True)
            process_log.setWordWrapMode(QTextEdit.WidgetWidth)
            process_log.setLineWrapMode(QTextEdit.WidgetWidth)
            process_log.setStyleSheet("""
                QTextEdit {
                    background-color: #F8FAFC;
                    border: 0.5px solid #000000;
                    font-family: "Jura";
                    font-size: 14px;
                    color: #000000;
                    padding: 5px;
                }
            """)

            # Get or create image data
            if image_path not in self.image_data:
                # Header
                process_log.append("╔══════════════════════════════════╗")
                process_log.append("║       PROCESSING INFORMATION     ║")
                process_log.append("╚══════════════════════════════════╝")
                process_log.append(f"\n📄 File: {os.path.basename(image_path)}")

                if folder:
                    process_log.append(f"📁 Folder: {self.get_folder_name(folder)}")

                # OCR Section
                process_log.append("\n┌──────────────────────────────────┐")
                process_log.append("│          TEXT EXTRACTION         │")
                process_log.append("└──────────────────────────────────┘")
                process_log.append("\n🔍 Performing OCR text extraction...")
                # text = self.detector.extract_text(image_path)
                text = self.easy_ocr.extract_text(image_path)
                process_log.append(f"\n📝 Extracted Text:\n{text}")

                # Prediction Section
                process_log.append("\n┌──────────────────────────────────┐")
                process_log.append("│        HATE SPEECH ANALYSIS      │")
                process_log.append("└──────────────────────────────────┘")
                process_log.append("\n🧠 Running hate speech prediction...")
                # prediction = self.detector.predict(image_path, text)
                prediction = self.hate_speech_detector.predict_with_adaptive_fusion(
                            image_path=image_path, text=text
                        )
                
                process_log.append(f"\n📊 Prediction Score: {prediction:.4f}")
                process_log.append(
                    f"🚨 Conclusion: {'HATE SPEECH DETECTED' if prediction > 0.5 else 'No hate speech detected'}"
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
                # Header
                process_log.append("╔══════════════════════════════════╗")
                process_log.append("║       CACHED RESULTS            ║")
                process_log.append("╚══════════════════════════════════╝")
                process_log.append(f"\n📄 File: {os.path.basename(image_path)}")

                if data["folder"]:
                    process_log.append(
                        f"📁 Folder: {self.get_folder_name(data['folder'])}"
                    )

                # OCR Section
                process_log.append("\n┌──────────────────────────────────┐")
                process_log.append("│          TEXT EXTRACTION         │")
                process_log.append("└──────────────────────────────────┘")
                process_log.append(f"\n📝 Extracted Text:\n{data['text']}")

                # Prediction Section
                process_log.append("\n┌──────────────────────────────────┐")
                process_log.append("│        HATE SPEECH ANALYSIS      │")
                process_log.append("└──────────────────────────────────┘")
                process_log.append(f"\n📊 Prediction Score: {data['prediction']:.4f}")
                process_log.append(
                    f"🚨 Conclusion: {'HATE SPEECH DETECTED' if data['is_hate_speech'] else 'No hate speech detected'}"
                )

            self.ipo_table.setCellWidget(row, 1, process_log)
            self.ipo_table.setRowHeight(
                row, int(process_log.document().size().height() + 10)
            )

            # Output column - larger and with status indicator
            output_widget = QWidget()
            output_layout = QVBoxLayout(output_widget)

            is_hate_speech = self.image_data[image_path]["is_hate_speech"]
            status_label = QLabel("🚨 HATE SPEECH" if is_hate_speech else "✅ CLEAN")
            status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_label.setStyleSheet(
                """
                font-weight: bold; 
                font-family: "Jura";
                color: {color};
                font-size: 16px;
                padding: 3px;
            """.format(color="red" if is_hate_speech else "green")
            )
            output_layout.addWidget(status_label)

            output_label = QLabel()
            if is_hate_speech:
                image = Image.open(image_path)

                if image.mode != "RGB":
                    image = image.convert("RGB")

                blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))
                blurred_image.save("temp_blurred.jpg")
                output_pixmap = QPixmap("temp_blurred.jpg")
            else:
                output_pixmap = QPixmap(image_path)

            output_pixmap = output_pixmap.scaled(
                300, 180, Qt.AspectRatioMode.KeepAspectRatio
            )
            output_label.setPixmap(output_pixmap)
            output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            output_layout.addWidget(output_label)
            output_widget.setLayout(output_layout)

            self.ipo_table.setCellWidget(row, 2, output_widget)

        except Exception as e:
            error_item = QTableWidgetItem(f"Error processing image: {str(e)}")
            error_item.setForeground(QColor("red"))
            self.ipo_table.setItem(row, 1, error_item)

    # region FUNCTIONS
    def on_single_image_finished(self, result):
        self.processing_timer.stop()
        try:
            if hasattr(self, "loading_dialog"):
                self.loading_dialog.close()
            if hasattr(self, "overlay"):
                self.overlay.hide()

            # Process the result
            image_data = {
                "path": result["path"],
                "score": result["score"],
                "type": "single",
                "date": result["date"],
            }

            # Update data structures
            if result["score"] > 0.5:
                self.hate_speech_images.append(image_data)
                self.total_scanned_hatespeech_images.append(result["path"])
            else:
                self.non_hate_speech_images.append(image_data)
                self.total_scanned_NOT_hatespeech_images.append(result["path"])

            if result["path"] not in self.total_scanned_image_files:
                self.total_scanned_image_files.append(result["path"])

            # Update UI
            self.update_dashboard_metrics()
            self.update_explore()
            self.update_hate_speech_images()
            self.update_ipo_table()

            # Clean up worker - only if it exists
            if hasattr(self, "worker") and self.worker is not None:
                self.worker.stop()  # Stop the worker if it's still running
                self.worker.wait()  # Wait for it to finish
                self.worker.deleteLater()  # Schedule for deletion
                self.worker = None  # Clear the reference

        except Exception as e:
            print(f"Error in on_single_image_finished: {str(e)}")
            self.show_custom_popup("Error", f"Failed to process image: {str(e)}")
            # Clean up worker even if there's an error
            if hasattr(self, "worker") and self.worker is not None:
                if self.worker.isRunning():
                    self.worker.stop()
                    self.worker.wait()
                self.worker.deleteLater()
                self.worker = None

    def create_themed_image_label(self, pixmap, size=(300, 150)):
        label = QLabel()
        label.setFixedSize(size[0], size[1])

        # Make everything transparent
        label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                border: none;
                padding: 5px;
            }
        """)

        # Scale the pixmap to fit while maintaining aspect ratio
        scaled_pixmap = self.crop_and_scale_pixmap(pixmap, size[0] - 10, size[1] - 10)
        label.setPixmap(scaled_pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        return label

    def remove_folder_data(self, folder_path):
        """Remove all data associated with a folder"""
        if folder_path in self.folders:
            self.folders.remove(folder_path)

        # Remove images from all tracking lists - only those that are actually in the folder
        self.total_scanned_image_files = [
            img
            for img in self.total_scanned_image_files
            if not (img.startswith(folder_path) and os.path.dirname(img) != folder_path)
        ]

        self.total_scanned_hatespeech_images = [
            img
            for img in self.total_scanned_hatespeech_images
            if not (img.startswith(folder_path) and os.path.dirname(img) != folder_path)
        ]

        self.total_scanned_NOT_hatespeech_images = [
            img
            for img in self.total_scanned_NOT_hatespeech_images
            if not (img.startswith(folder_path) and os.path.dirname(img) != folder_path)
        ]

        # Remove from hate speech and non-hate speech lists
        self.hate_speech_images = [
            img
            for img in self.hate_speech_images
            if not (
                img["path"].startswith(folder_path)
                and os.path.dirname(img["path"]) != folder_path
            )
        ]

        self.non_hate_speech_images = [
            img
            for img in self.non_hate_speech_images
            if not (
                img["path"].startswith(folder_path)
                and os.path.dirname(img["path"]) != folder_path
            )
        ]

        # Clear folder data
        if folder_path in self.folder_data:
            del self.folder_data[folder_path]

        # Update all views
        self.update_dashboard_metrics()
        self.update_explore()
        self.update_hate_speech_images()
        self.update_ipo_table()

        # Clean up worker
        if hasattr(self, "worker") and self.worker is not None:
            if self.worker.isRunning():
                self.worker.stop()
                self.worker.wait()
            self.worker.deleteLater()
            self.worker = None

    def sort_dashboard_folders(self, sort_type):
        if not self.folders:
            return

        # Get all folder statistics first
        folder_stats = []
        for folder_path in self.folders:
            folder_images = [
                img
                for img in self.total_scanned_image_files
                if img.startswith(folder_path)
            ]
            folder_hate_speech = [
                img
                for img in self.hate_speech_images
                if img["path"].startswith(folder_path)
            ]
            total = len(folder_images)
            hate_count = len(folder_hate_speech)

            if total > 0:
                percentage = (hate_count / total) * 100
                folder_stats.append(
                    {
                        "path": folder_path,
                        "name": os.path.basename(folder_path),
                        "total": total,
                        "hate_count": hate_count,
                        "percentage": percentage,
                        "modified_time": os.path.getmtime(folder_path),
                    }
                )

        # Sort based on selected criteria
        if sort_type == "desc":
            # Sort by hate speech percentage (highest first)
            folder_stats.sort(key=lambda x: x["percentage"], reverse=True)
        elif sort_type == "asc":
            # Sort by hate speech percentage (lowest first)
            folder_stats.sort(key=lambda x: x["percentage"])
        elif sort_type == "recent":
            # Sort by most recent modification time
            folder_stats.sort(key=lambda x: x["modified_time"], reverse=True)
        else:  # oldest
            # Sort by oldest modification time
            folder_stats.sort(key=lambda x: x["modified_time"])

        # Clear existing folder buttons
        layout = self.verticalLayout_9
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add sorted folder buttons
        for folder in folder_stats[:5]:  # Only show top 5
            btn = QPushButton(
                f"{folder['name']}\n"
                f"Hate Speech: {folder['hate_count']}/{folder['total']}\n"
                f"({folder['percentage']:.1f}%)"
            )

            # Theme-aware styling
            if self.theme_manager.current_theme == "dark":
                high_risk = "#8B0000"  # Dark red
                medium_risk = "#8B4513"  # Dark orange
                low_risk = "#006400"  # Dark green
                text_color = "#FFFFFF"  # White text
            elif self.theme_manager.current_theme == "light_blue":
                high_risk = "#FF6B6B"  # Light red
                medium_risk = "#FFA07A"  # Light orange
                low_risk = "#90EE90"  # Light green
                text_color = "#000000"  # Black text
            else:  # Default theme
                high_risk = "#FF9C9C"  # Original red
                medium_risk = "#FFD59C"  # Original orange
                low_risk = "#9CFF9C"  # Original green
                text_color = "#000000"  # Black text

            # Style based on percentage
            if folder["percentage"] > 50:
                color = high_risk
                hover_color = (
                    "#FF4444"
                    if self.theme_manager.current_theme != "dark"
                    else high_risk
                )
            elif folder["percentage"] > 20:
                color = medium_risk
                hover_color = (
                    "#FF9966"
                    if self.theme_manager.current_theme != "dark"
                    else medium_risk
                )
            else:
                color = low_risk
                hover_color = (
                    "#77DD77"
                    if self.theme_manager.current_theme != "dark"
                    else low_risk
                )

            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    border-radius: 10px;
                    color: {text_color};
                    padding: 10px;
                }}
                QPushButton:hover {{
                    background-color: {hover_color};
                }}
            """)

            btn.clicked.connect(
                lambda checked, path=folder["path"]: self.explore_folder(path)
            )
            layout.addWidget(btn)

        # Update the chart to reflect the new order
        self.update_dashboard_chart()

    # def show_processing_delay_warning(self):
    #     reply = QMessageBox.question(
    #         self,
    #         "Long Processing Time",
    #         "Processing is taking longer than expected.\nWould you like to continue?",
    #         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    #     )
    #     if reply == QMessageBox.StandardButton.No:
    #         self.cancel_folder_processing()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GalleryApplication()
    window.show()
    sys.exit(app.exec())
