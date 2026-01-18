# main_runner.py (Merged Version)
# This Python file uses the following encoding: utf-8

# Standard library imports
import io
import sys
import os
import time
from datetime import datetime
import warnings

# Third-party library imports
from fpdf import FPDF
from PIL import Image, ImageFilter
import torch
import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# PyQt6 imports
from PyQt6.QtCore import Qt, QSize, QDateTime, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QAction, QMovie, QPainter
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
    QProgressBar,
    QSizePolicy,
)
from PyQt6.uic import loadUi

# Local application imports
from easy_ocr import OCRExtractor
from clip_and_hybrid_model import clip_and_hybrid_model


warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Using a slow image processor.*")
try:
    if sys.stdout:
        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")
except (AttributeError, ValueError):
    pass


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# region
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GALLERY_UI_PATH = resource_path("ui_files/gallery_application.ui")
STYLESHEET_PATH = resource_path("qss_files/stylesheet.qss")
LIGHT_BLUE_THEME_PATH = resource_path("qss_files/light_blue_theme.qss")
DARK_THEME_PATH = resource_path("qss_files/dark_theme.qss")

NEURALJAM_LOGO = resource_path("assets/NeuralJAM_Logo_60x60.png")
LOADING_GIF = resource_path("assets/loading.gif")

HAMBURGER_ICON = {
    "black": resource_path("assets/hamburgerIcon_black.png"),
    "white": resource_path("assets/hamburgerIcon_white.png"),
}

SIDEBAR_ICONS = {
    "dashboard": {
        "black": resource_path("assets/dashboard_black.png"),
        "white": resource_path("assets/dashboard_white.png"),
    },
    "explore": {
        "black": resource_path("assets/explore_black.png"),
        "white": resource_path("assets/explore_white.png"),
    },
    "hate_speech": {
        "black": resource_path("assets/hatespeechImage_black.png"),
        "white": resource_path("assets/hatespeechImage_white.png"),
    },
    "inference": {
        "black": resource_path("assets/IPO_black.png"),
        "white": resource_path("assets/IPO_white.png"),
    },
    "info": {
        "black": resource_path("assets/information_blue.png"),
        "white": resource_path("assets/information_blue.png"),
    },
    "settings": {
        "black": resource_path("assets/settings_black.png"),
        "white": resource_path("assets/settings_white.png"),
    },
}

USER_GUIDE_ICON_PATH = resource_path("assets/information_blue.png")
BACK_ICON = resource_path("assets/back_black.png")
DROPDOWN_ICON_BLACK = resource_path("assets/dropdown_black.png")
DROPDOWN_ICON_WHITE = resource_path("assets/dropdown_white.png")

USER_GUIDE_QSS = "font-size: 16px"

# endregion


class ProcessingWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, folder_path=None, image_path=None):
        super().__init__()
        self.folder_path = folder_path
        self.image_path = image_path
        self._is_running = True

        # Add system files to ignore
        self.SYSTEM_FILES = {"desktop.ini", "thumbs.db", ".ds_store"}
        self.ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

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
            detector = (
                self.hate_speech_detector
            )  # Avoid recreating the detector every time

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
                        score, interpretation = detector.predict_with_adaptive_fusion(
                            image_path=file_path, text=text
                        )

                        results.append(
                            {
                                "path": file_path,
                                "score": score,
                                "interpretation": interpretation,
                            }
                        )
                        if score > 0.3:
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
        text = self.easy_ocr.extract_text(self.image_path)
        score, interpretation = self.hate_speech_detector.predict_with_adaptive_fusion(
            image_path=self.image_path, text=text
        )

        self.finished.emit(
            {
                "path": self.image_path,
                "score": score,
                "type": "single",
                "interpretation": interpretation,
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
                "active_color": "#3B1E54",
                "hover_color": "#9B7EBD",
                "text_color": "#FFFFFF",
            },
            "light_blue": {
                "qss": LIGHT_BLUE_THEME_PATH,
                "active_color": "#4682B4",
                "hover_color": "#5F9EA0",
                "text_color": "#FFFFFF",
            },
            "dark": {
                "qss": DARK_THEME_PATH,
                "active_color": "#444444",
                "hover_color": "#666666",
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

                if hasattr(self.parent, "update_combo_box_styles"):
                    self.parent.update_combo_box_styles()

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

    def get_combo_box_style(self):
        if self.current_theme == "dark":
            return f"""
                QComboBox {{
                    background-color: #444444;
                    color: #FFFFFF;
                    border: 1px solid #666666;
                    padding: 5px;
                    border-radius: 4px;
                    font-size: 16px;
                    padding-right: 25px; /* Make space for the dropdown icon */
                }}
                QComboBox:hover {{
                    background-color: #555555;
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: 20px;
                    background: transparent;
                }}
                QComboBox::down-arrow {{
                    image: url({DROPDOWN_ICON_WHITE});
                    width: 12px;
                    height: 12px;
                }}
                QComboBox QAbstractItemView {{
                    background-color: #444444;
                    color: #FFFFFF;
                    selection-background-color: #666666;
                    font-size: 16px;
                    outline: none;
                    border: 1px solid #666666;
                }}
            """
        elif self.current_theme == "light_blue":
            return f"""
                QComboBox {{
                    background-color: #F0F8FF;
                    color: #000000;
                    border: 1px solid #BCCCDC;
                    padding: 5px;
                    border-radius: 4px;
                    font-size: 16px;
                    padding-right: 25px;
                }}
                QComboBox:hover {{
                    background-color: #E0F0FF;
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: 20px;
                    background: transparent;
                }}
                QComboBox::down-arrow {{
                    image: url({DROPDOWN_ICON_BLACK});
                    width: 12px;
                    height: 12px;
                }}
                QComboBox QAbstractItemView {{
                    background-color: #F0F8FF;
                    color: #000000;
                    selection-background-color: #BCCCDC;
                    font-size: 16px;
                    outline: none;
                    border: 1px solid #BCCCDC;
                }}
            """
        else:  # default theme
            return f"""
                QComboBox {{
                    background-color: #FFFFFF;
                    color: #000000;
                    border: 1px solid #D4BEE4;
                    padding: 5px;
                    border-radius: 4px;
                    font-size: 16px;
                    padding-right: 25px;
                }}
                QComboBox:hover {{
                    background-color: #F5F0FA;
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: 20px;
                    background: transparent;
                }}
                QComboBox::down-arrow {{
                    image: url({DROPDOWN_ICON_BLACK});
                    width: 12px;
                    height: 12px;
                }}
                QComboBox QAbstractItemView {{
                    background-color: #FFFFFF;
                    color: #000000;
                    selection-background-color: #D4BEE4;
                    font-size: 16px;
                    outline: none;
                    border: 1px solid #D4BEE4;
                }}
            """


class GalleryApplication(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        loadUi(GALLERY_UI_PATH, self)

        """
            For hybrid model initialization and other instantiation
        """
        self.easy_ocr = OCRExtractor()

        """
            Data structures initialization
        """

        self.folders = []
        self.folder_data = {}
        self.image_data = {}
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
        self.ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
        self.SYSTEM_FILES = [
            "desktop.ini",
            "thumbs.db",
            ".ds_store",
        ]

        """
            For
                - App window title
                - Icon
                - Theme manager
        """
        self.setWindowTitle("NeuralJAM")
        self.setWindowIcon(QIcon(NEURALJAM_LOGO))

        # Get screen geometry
        self.showMaximized()

        self.theme_manager = ThemeManager(self)

        """
            Sidebar state and theme default setup
        """
        self.state_leftSideBar = False
        self.state_toggle_minfLeftSideBar = True
        self.sidebar_icon_button_mapping = {
            1: {
                "bttn_name": "bttn_sideBar_Dashboard",
                "icon_black": SIDEBAR_ICONS["dashboard"]["black"],
                "icon_white": SIDEBAR_ICONS["dashboard"]["white"],
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
            6: {
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
            QIcon(self.sidebar_icon_button_mapping[6]["icon_black"])
        )
        self.bttn_sideBar_Settings.setIconSize(QSize(20, 20))
        self.bttn_sideBar_Settings.setVisible(True)
        self.bttn_sideBar_Settings.clicked.connect(self.show_theme_menu)
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

        # Add User Guide button to hate speech images page
        self.hs_user_guide_button = QPushButton(" User Guide ", self.hsImage_upper)
        self.hs_user_guide_button.setObjectName("hs_user_guide_button")
        self.horizontalLayout_4.addWidget(self.hs_user_guide_button)
        self.hs_user_guide_button.clicked.connect(self.show_user_guide)
        self.hs_user_guide_button.setIcon(QIcon(USER_GUIDE_ICON_PATH))
        self.hs_user_guide_button.setStyleSheet(USER_GUIDE_QSS)
        self.hs_user_guide_button.setIconSize(QSize(18, 18))

        """
            INFERENCE PIPELINE PAGE FUNCTIONS
        """
        # Initialize IPO table with 3 columns
        self.ipo_table.setColumnCount(3)
        self.ipo_table.setHorizontalHeaderLabels(["Input", "Process", "Output"])
        self.ipo_table.verticalHeader().setVisible(False)
        self.ipo_table.setWordWrap(True)
        self.ipo_table.setShowGrid(True)  # Optional: cleaner look without grid lines
        self.bttn_GenerateReport.clicked.connect(self.generate_report)

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
        self.ipo_user_guide_button.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )

        # For Filters combo boxes
        self.filterByComboBox.setVisible(True)
        self.sortByComboBox.setVisible(True)
        self.groupByComboBox.setVisible(True)
        self.setup_combo_boxes_theme()

        self.setup_combo_boxes_functions()  # Connect signals for filtering/sorting
        self.update_group_by_combo_box()  # Populate group by dropdown

        # Configure header appearance
        self.ipo_table.setStyleSheet("""
            QHeaderView::section {
                background-color: #F8FAFC;
                color: #000000;
                font-family: "EXO Light";
                font-size: 18px;
                font-weight: bold;
                padding: 4px;
                border-bottom: 1px solid #000000;
                border-left: none;
                border-right: none;
                border-top: none;
            }
        """)

        # Configure column resize behavior
        header = self.ipo_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # Input
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Process
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)  # Output

        # Set specific column widths for Input and Output
        self.ipo_table.setColumnWidth(0, 250)
        self.ipo_table.setColumnWidth(2, 250)

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

        self.update_hate_speech_images()
        self.update_dashboard_metrics()

        # In the __init__ method, after loading UI
        self.bttn_Back.setIcon(QIcon(BACK_ICON))
        self.bttn_Back.setIconSize(QSize(20, 20))

        # region END OF INIT

    def setup_combo_boxes_functions(self):
        # Filter By ComboBox
        self.filterByComboBox.currentTextChanged.connect(self.update_ipo_table)

        # Group By ComboBox - populate with folders
        self.update_group_by_combo_box()
        self.groupByComboBox.currentTextChanged.connect(self.update_ipo_table)

        # Sort By ComboBox
        self.sortByComboBox.currentTextChanged.connect(self.update_ipo_table)

    def update_group_by_combo_box(self):
        """Update the Group By combo box with current folders"""
        self.groupByComboBox.clear()
        self.groupByComboBox.addItem("--Select--")
        self.groupByComboBox.addItem("All Content")

        # Add folders that have been scanned
        for folder in self.folders:
            folder_name = os.path.basename(folder)
            self.groupByComboBox.addItem(folder_name, folder)

    def setup_combo_boxes_theme(self):
        # Filter By ComboBox
        self.filterByComboBox.addItems(["--Select--", "Hate", "Not hate"])

        # Group By ComboBox
        self.groupByComboBox.addItems(["--Select--", "Folders", "Single image"])

        # Sort By ComboBox
        self.sortByComboBox.addItems(
            [
                "--Select--",
                "Hate Speech (High to Low)",
                "Hate Speech (Low to High)",
                "Most Recent",
                "Oldest",
            ]
        )

        # Apply initial theme
        self.update_combo_box_styles()

    def update_combo_box_styles(self):
        style = self.theme_manager.get_combo_box_style()
        self.filterByComboBox.setStyleSheet(style)
        self.groupByComboBox.setStyleSheet(style)
        self.sortByComboBox.setStyleSheet(style)

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

    def expand_sidebar(self):
        try:
            self.state_leftSideBar = True
            self.frm_leftSideBar.setVisible(True)
            self.frm_leftSideBar.setFixedWidth(197)
            self.lbl_sideBar_Icon.setPixmap(QPixmap(NEURALJAM_LOGO))
            self.bttn_sideBar_Hamburger.setIcon(QIcon(HAMBURGER_ICON["black"]))
            self.bttn_sideBar_Dashboard.setText("   Dashboard")
            self.bttn_sideBar_Explore.setText("   Explore")
            self.bttn_sideBar_HateSpeechImages.setText("   Hate Speech Images")
            self.bttn_sideBar_InferencePipeline.setText("   Inference Pipeline")
            self.bttn_sideBar_Settings.setText("   Settings")
            self.set_sideBard_icons_to_black()
        except Exception as e:
            print(f"Error expanding sidebar: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to expand sidebar: {str(e)}")

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

    def switch_page_to_dashboard_and_toggle_leftSideBar_visiblity(self):
        try:
            print("Attempting to switch to dashboard...")
            # First show the sidebar
            self.expand_sidebar()

            # Then switch pages
            self.pagesStackedWidget.setCurrentIndex(1)  # Dashboard
            if self.pagesStackedWidget.currentIndex() == 0:
                self.verticalLayout_Page_Welcome.setAlignment(
                    Qt.AlignmentFlag.AlignCenter
                )

            # Update button colors
            self.update_button_colors_when_in_page(1)
            print("Successfully switched to dashboard")

        except Exception as e:
            print(f"Error during page switch: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to switch pages: {str(e)}")

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

        if index == 6:
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
                QIcon(self.sidebar_icon_button_mapping[6]["icon_white"])
            )
        else:
            self.bttn_sideBar_Settings.setStyleSheet("")
            self.bttn_sideBar_Settings.setIcon(
                QIcon(self.sidebar_icon_button_mapping[6]["icon_black"])
            )

        self.update_sort_dropdown_style()
        self.update_sort_menu_style()
        # Store the new active index
        self.previous_active_index = index

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
        self.gridLayout_5.addWidget(self.info_label, 0, 0, 1, 3)

        layout = self.scrollArea_explr_Folders_Content.layout()
        layout.addWidget(self.info_label)

    def remove_initial_label(self):
        if hasattr(self, "info_label") and self.info_label:
            print("🗑️ Removing the initial 'No folders' label.")
            layout = self.scrollArea_explr_Folders_Content.layout()
            layout.removeWidget(self.info_label)
            self.info_label.deleteLater()
            self.info_label = None

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
            btn.setMinimumHeight(100)

        # Increase font size
        folder_btn.setStyleSheet("font-size: 18px; font-weight: 300")
        single_btn.setStyleSheet("font-size: 18px; font-weight: 300")

        folder_btn.clicked.connect(lambda: self.handle_folder_selection(dialog))
        single_btn.clicked.connect(lambda: self.handle_single_image_selection(dialog))

        layout.addWidget(folder_btn)
        layout.addWidget(single_btn)

        layout.setSpacing(20)
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

        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if isinstance(widget, QPushButton):
                base_text = widget.text().split(" (")[0]
                if base_text not in existing_names:
                    existing_names[base_text] = []
                existing_names[base_text].append(widget)

        if folder_name in existing_names:
            for btn in existing_names[folder_name]:
                if "(" not in btn.text():
                    old_path = btn.toolTip() if btn.toolTip() else ""
                    old_parent = (
                        os.path.basename(os.path.dirname(old_path))
                        if old_path
                        else "Unknown"
                    )
                    btn.setText(f"{folder_name} ({old_parent})")
            folder_name = f"{folder_name} ({parent})"

        self.remove_initial_label()

        count = layout.count()
        row = count // 3
        col = count % 3

        button = QPushButton(folder_name)
        button.setProperty("folder_path", folder_path)
        button.setToolTip(folder_path)
        button.setFixedSize(370, 100)
        button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        button.setStyleSheet(
            'font-family: "EXO Light"; font-size: 21px; font-weight: 600;'
        )
        button.clicked.connect(lambda _, path=folder_path: self.explore_folder(path))

        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)

        layout.addWidget(button, row, col)
        print(f"✅ Added button for folder '{folder_name}' at row {row}, col {col}")

        self.update_dashboard_metrics()
        self.update_explore()

    def populate_images_from_folder(self, folder_path):
        layout = self.scrllArea_explr_AllImages_Content.layout()
        if layout is None:
            print("⚠️ No layout found on scrllArea_explr_AllImages_Content!")
            return

        self.scanned_image_files = []
        self.excluded_files = []

        files = os.listdir(folder_path)

        for file_name in files:
            # Skip system files
            if file_name.lower() in self.SYSTEM_FILES:
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

        old_widgets = []
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                old_widgets.append(widget)

        for i, image_file in enumerate(self.scanned_image_files):
            image_path = os.path.normpath(os.path.join(folder_path, image_file))
            pixmap = QPixmap(image_path)

            label = self.create_themed_image_label(pixmap)

            count = layout.count()
            row = count // 3
            col = count % 3

            layout.addWidget(label, row, col)

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

    def crop_and_scale_pixmap(self, pixmap, target_width, target_height):
        source_ratio = pixmap.width() / pixmap.height()
        target_ratio = target_width / target_height

        if source_ratio > target_ratio:
            scaled = pixmap.scaledToWidth(
                target_width, Qt.TransformationMode.SmoothTransformation
            )
        else:
            scaled = pixmap.scaledToHeight(
                target_height, Qt.TransformationMode.SmoothTransformation
            )

        return scaled

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
            if folder_path in self.folders:
                self.folders.remove(folder_path)

            explore_layout = self.scrollArea_explr_Folders_Content.layout()
            if explore_layout:
                for i in reversed(range(explore_layout.count())):
                    widget = explore_layout.itemAt(i).widget()
                    if isinstance(widget, QPushButton):
                        if widget.property("folder_path") == folder_path:
                            explore_layout.removeWidget(widget)
                            widget.deleteLater()

            self.refill_grid_layout(explore_layout, columns=3)

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

            if folder_path in self.folder_data:
                del self.folder_data[folder_path]

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

            self.update_dashboard_metrics()
            self.update_explore()
            self.update_hate_speech_images()
            self.setup_initial_label()

            self.ipo_table.setRowCount(0)
            self.update_ipo_table()

            self.back_to_folder_overview()

            print(f"✅ Successfully removed folder: {folder_path}")

        except Exception as e:
            print(f"❌ Failed to remove folder: {e}")
            QMessageBox.critical(self, "Error", f"Could not remove folder:\n{e}")

    def refill_grid_layout(self, layout, columns=3):
        widgets = []
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget:
                layout.removeWidget(widget)
                widgets.append(widget)

        for index, widget in enumerate(reversed(widgets)):
            row = index // columns
            col = index % columns
            layout.addWidget(widget, row, col)

    def explore_folder(self, folder_path):
        self.pagesStackedWidget.setCurrentIndex(5)

        self.lbl_ExploredFolderPath.setText(
            f"  Explore >> {os.path.basename(folder_path)}"
        )

        self.bttn_Back.setVisible(True)
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

    def populate_explored_folder_images(self, folder_path):
        layout = self.scrllArea_exploredFolder_Content.layout()
        if layout is None:
            print("⚠️ No layout found on scrllArea_exploredFolder_Content!")
            return

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

        label_width = 400
        label_height = 400

        for i, image_path in enumerate(scanned_images):
            print(f"🖼️ Exploring: {image_path}")

            is_hate_speech = any(
                img["path"] == image_path and img["score"] > 0.3
                for img in self.hate_speech_images
            )

            if is_hate_speech:
                pixmap = self.process_image_for_display(image_path, blur=True)
            else:
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

    def back_to_folder_overview(self):
        self.pagesStackedWidget.setCurrentIndex(2)

    def show_custom_popup(self, title, message):
        self.overlay = QWidget(self)
        self.overlay.setGeometry(self.rect())
        self.overlay.setStyleSheet(
            "background-color: rgba(0, 0, 0, 120); font-size: 16px;"
        )
        self.overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.overlay.show()

        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)
        dialog.setFixedSize(600, 400)

        main_layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        lbl_message = QLabel(message)
        lbl_message.setText(message)
        lbl_message.setWordWrap(True)
        content_layout.addWidget(lbl_message)

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(dialog.accept)
        main_layout.addWidget(btn_ok)

        dialog.setLayout(main_layout)

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

    def process_image_for_display(self, image_path, blur=False):
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            if blur:
                # Create a more subtle blur effect
                image = image.filter(ImageFilter.GaussianBlur(radius=10))

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
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
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
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
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
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
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
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
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
                        row = self.ipo_table.rowCount()
                        self.ipo_table.insertRow(row)

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
            no_folder_label = QLabel(
                'No folders scanned yet.\n Please go to "Explore Page" and Scan a folder.'
            )
            no_folder_label.setStyleSheet("""
                QLabel {
                    color: #666666;
                    font-size: 17px;
                    font-family: Arial;
                    padding: 20px;
                    background: transparent;
                }
            """)

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
                'No folders scanned yet.\n  Please go to "Explore Page" and scan a folder to see BAR CHART statistics.'
            )
            no_data_label.setStyleSheet("""
                QLabel {
                    color: #666666;
                    font-size: 17px;
                    font-family: Arial;
                    padding: 20px;
                    background: transparent;
                }
            """)
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
        label_width = 400
        label_height = 400

        for i, image_data in enumerate(all_images):
            image_path = image_data["path"]
            print(f"🖼️ Exploring: {image_path}")
            try:
                # Only blur if it's hate speech
                pixmap = self.process_image_for_display(
                    image_path,
                    blur=image_data["score"]
                    > 0.3,  # Only blur if score > 0.3 ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
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

        # Clear existing widgets including the default "Text Label"
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        if not self.hate_speech_images:
            # Create a more professional empty state message
            empty_label = QLabel("No hate speech images detected yet.")
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
                label = self.create_themed_image_label(pixmap, size=(400, 400))
                layout.addWidget(label, row, col)
            # except Exception as e:
            except Exception:
                error_label = QLabel("Error loading image")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(error_label, row, col)

    def generate_report(self):
        try:
            # First show confirmation dialog
            filter_text = self.filterByComboBox.currentText()
            group_text = self.groupByComboBox.currentText()
            sort_text = self.sortByComboBox.currentText()

            confirm_msg = f"""
            <div style="font-size:19px;">
                <b>Report Generation Confirmation</b><br><br>
                This report will be generated based on:<br>
                - <b>Filter By:</b> {filter_text if filter_text != "--Select--" else "All"}<br>
                - <b>Group By:</b> {group_text if group_text != "--Select--" else "All Content"}<br>
                - <b>Sort By:</b> {sort_text if sort_text != "--Select--" else "Default"}<br><br>
                The report will include:<br>
                1. Dashboard overview statistics<br>
                2. Detailed analysis of processed images<br><br>
                <i>Would you like to proceed?</i>
            </div>
            """

            reply = QMessageBox.question(
                self,
                "Confirm Report Generation",
                confirm_msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

            # Create PDF with Unicode support in LANDSCAPE orientation
            pdf = FPDF(orientation="L")
            pdf.add_page()
            pdf.set_font("Arial", "", 12)

            # Add page number function
            pdf.alias_nb_pages()

            # Create a footer with page numbers
            def add_page_number(pdf):
                pdf.set_y(-15)  # Position at 15 mm from bottom
                pdf.set_font("Arial", "I", 8)
                pdf.cell(0, 10, f"Page {pdf.page_no()}/{{nb}}", 0, 0, "R")

            # Set footer function
            pdf.set_auto_page_break(True, 15)
            pdf._footer = add_page_number

            # Header with logo and title
            pdf.image(NEURALJAM_LOGO, x=10, y=8, w=30)
            pdf.set_font("Arial", "B", 16)
            pdf.cell(
                0, 10, "NeuralJAM Hate Speech Detection Report", ln=True, align="C"
            )

            # Report metadata
            pdf.set_font("Arial", "", 12)
            pdf.cell(
                0,
                10,
                f"Report generated on: {QDateTime.currentDateTime().toString('yyyy-MM-dd HH:mm:ss')}",
                ln=True,
                align="C",
            )
            pdf.cell(
                0,
                10,
                f"Filter by: {filter_text if filter_text != '--Select--' else 'All'} | "
                f"Group by: {group_text if group_text != '--Select--' else 'All Content'} | "
                f"Sort by: {sort_text if sort_text != '--Select--' else 'Default'}",
                ln=True,
                align="C",
            )
            pdf.ln(15)

            # Dashboard Overview Section - First Page Only
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "1. Summary Statistics", ln=True)
            pdf.set_font("Arial", "", 12)

            # Calculate statistics
            total_images = len(self.total_scanned_image_files)
            hate_speech_count = len(self.total_scanned_hatespeech_images)
            non_hate_speech_count = total_images - hate_speech_count
            hate_speech_percentage = (
                (hate_speech_count / total_images * 100) if total_images > 0 else 0
            )

            # Create summary table
            col_width = pdf.w / 2 - 15
            row_height = 10

            # Table header
            pdf.set_fill_color(200, 200, 200)
            pdf.cell(col_width, row_height, "Metric", 1, 0, "C", 1)
            pdf.cell(col_width, row_height, "Value", 1, 1, "C", 1)

            # Table rows
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(col_width, row_height, "Total Images Processed", 1, 0, "L", 1)
            pdf.cell(col_width, row_height, str(total_images), 1, 1, "C")

            pdf.cell(col_width, row_height, "Hate Speech Detections", 1, 0, "L", 1)
            pdf.cell(col_width, row_height, str(hate_speech_count), 1, 1, "C")

            pdf.cell(col_width, row_height, "Clean Images", 1, 0, "L", 1)
            pdf.cell(col_width, row_height, str(non_hate_speech_count), 1, 1, "C")

            pdf.cell(col_width, row_height, "Hate Speech Percentage", 1, 0, "L", 1)
            pdf.cell(col_width, row_height, f"{hate_speech_percentage:.1f}%", 1, 1, "C")

            # Add some space before the detailed analysis section starts on next page
            pdf.ln(20)
            pdf.set_font("Arial", "I", 10)
            pdf.cell(
                0,
                10,
                "Detailed analysis begins on the next page...",
                ln=True,
                align="C",
            )

            # Image Analysis Section - Starts on new page
            rows = self.ipo_table.rowCount()
            if rows > 0:
                # Get the current IPO table data
                entries_per_page = 2  # Show 2 entries per page
                current_entry = 0

                while current_entry < rows:
                    pdf.add_page()

                    # Section header for each page
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(
                        0,
                        10,
                        f"2. Detailed Image Analysis (Entries {current_entry + 1}-{min(current_entry + entries_per_page, rows)})",
                        ln=True,
                    )
                    pdf.set_font("Arial", "", 10)

                    # Create table header row
                    col_widths = [
                        60,
                        150,
                        60,
                    ]  # Wider columns for landscape orientation
                    pdf.set_fill_color(200, 200, 200)
                    pdf.cell(col_widths[0], 10, "Original Image", 1, 0, "C", 1)
                    pdf.cell(col_widths[1], 10, "Analysis Details", 1, 0, "C", 1)
                    pdf.cell(col_widths[2], 10, "Result", 1, 1, "C", 1)

                    # Set consistent min row height and line height
                    # min_row_height = 50  # Minimum row height in points
                    line_height = 6  # Height of each text line
                    gap = 5  # Gap between rows to prevent overlapping

                    # Calculate available height per entry (half of remaining space after header)
                    available_height = (
                        pdf.h - pdf.get_y() - 20
                    ) / entries_per_page  # Leave space for footer

                    # Process entries for this page
                    for i in range(entries_per_page):
                        if current_entry >= rows:
                            break

                        row = current_entry
                        current_entry += 1

                        # Get data from each column
                        input_widget = self.ipo_table.cellWidget(row, 0)
                        process_widget = self.ipo_table.cellWidget(row, 1)
                        output_widget = self.ipo_table.cellWidget(row, 2)

                        # Input column - get image
                        input_item = self.ipo_table.item(row, 0)
                        input_image_path = input_item.text() if input_item else None
                        if input_widget:
                            # Look for QLabel with pixmap in the input widget
                            for child in input_widget.findChildren(QLabel):
                                if child.pixmap() and not child.pixmap().isNull():
                                    # First try to get path from toolTip
                                    if child.toolTip() and os.path.exists(
                                        child.toolTip()
                                    ):
                                        input_image_path = child.toolTip()
                                        break
                                    # If not in toolTip, try to get it from property
                                    elif child.property("imagePath") and os.path.exists(
                                        child.property("imagePath")
                                    ):
                                        input_image_path = child.property("imagePath")
                                        break
                                    # If still not found, check parent widget properties
                                    elif input_widget.property(
                                        "imagePath"
                                    ) and os.path.exists(
                                        input_widget.property("imagePath")
                                    ):
                                        input_image_path = input_widget.property(
                                            "imagePath"
                                        )
                                        break

                            # Debug info - helpful for troubleshooting
                            if not input_image_path:
                                print("Could not find valid image path in input widget")

                        # Process column - extract simplified information
                        process_text = ""
                        file_type = "Single Image"
                        folder_name = ""
                        if process_widget:
                            process_label = process_widget.findChild(QLabel)
                            if process_label:
                                # Extract the simplified information
                                text = process_label.text()
                                lines = text.split("\n")

                                # Get date and time
                                date_line = next(
                                    (
                                        line
                                        for line in lines
                                        if "Date & Time Processed:" in line
                                    ),
                                    "",
                                )
                                date_time = (
                                    date_line.partition("Date & Time Processed:")[
                                        2
                                    ].strip()
                                    if date_line
                                    else "N/A"
                                )

                                # Get file info
                                file_line = next(
                                    (line for line in lines if "File:" in line), ""
                                )
                                file_name = (
                                    file_line.split(":")[1].strip()
                                    if file_line
                                    else "N/A"
                                )

                                # Get file type and folder
                                type_line = next(
                                    (line for line in lines if "File Type:" in line), ""
                                )
                                if type_line:
                                    if "Folder" in type_line:
                                        file_type = "Folder"
                                        folder_line = next(
                                            (
                                                line
                                                for line in lines
                                                if "Folder:" in line
                                            ),
                                            "",
                                        )
                                        folder_name = (
                                            folder_line.split(":")[1].strip()
                                            if folder_line
                                            else ""
                                        )

                                # Get extracted text (limit to 5 lines)
                                text_lines = []
                                in_text_section = False
                                text_line_count = 0
                                for line in lines:
                                    if "Extracted Text:" in line:
                                        in_text_section = True
                                    elif "HATE SPEECH ANALYSIS:" in line:
                                        in_text_section = False
                                    elif in_text_section and line.strip():
                                        if text_line_count < 5:  # Limit to 5 lines
                                            text_lines.append(line.strip())
                                            text_line_count += 1
                                        elif text_line_count == 5:
                                            text_lines.append("... [text truncated]")
                                            text_line_count += 1
                                extracted_text = (
                                    "\n".join(text_lines)
                                    if text_lines
                                    else "No text detected"
                                )

                                # Get prediction info
                                score_line = next(
                                    (
                                        line
                                        for line in lines
                                        if "Prediction Score:" in line
                                    ),
                                    "",
                                )
                                score = (
                                    score_line.split(":")[1].strip()
                                    if score_line
                                    else "N/A"
                                )

                                interpretation_line = next(
                                    (
                                        line
                                        for line in lines
                                        if "Interpretation:" in line
                                    ),
                                    "",
                                )
                                interpretation = (
                                    interpretation_line.split(":")[1].strip()
                                    if interpretation_line
                                    else "N/A"
                                )

                                conclusion_line = next(
                                    (line for line in lines if "Conclusion:" in line),
                                    "",
                                )
                                conclusion = (
                                    conclusion_line.split(":")[1].strip()
                                    if conclusion_line
                                    else "N/A"
                                )

                                path_line = next(
                                    (line for line in lines if "Path:" in line),
                                    "",
                                )
                                path = (
                                    path_line.split(":", 1)[1].strip()
                                    if path_line
                                    else "N/A"
                                )

                                # Build simplified process text
                                process_text = f"Date & Time: {date_time}\n"
                                process_text += f"File: {file_name}\n"
                                process_text += f"Type: {file_type}"
                                if file_type == "Folder":
                                    process_text += (
                                        f" ({folder_name})" if folder_name else ""
                                    )
                                process_text += "\n\nExtracted Text:\n" + extracted_text
                                process_text += "\n\nPrediction Score: " + score
                                process_text += "\nInterpretation: " + interpretation
                                process_text += "\nConclusion: " + conclusion
                                process_text += "\n\nFile Path: " + path

                        # Output column - get image and status
                        output_item = self.ipo_table.item(row, 2)
                        output_image_path = output_item.text() if output_item else None
                        # output_image_path = None
                        output_status = "No result"
                        if output_widget:
                            # First get status from status label if it exists
                            status_label = output_widget.findChild(QLabel)
                            if status_label:
                                output_status = (
                                    "HATE SPEECH"
                                    if "HATE" in status_label.text().upper()
                                    else "CLEAN"
                                )

                            # Then look for image in any QLabel with pixmap
                            for child in output_widget.findChildren(QLabel):
                                if child.pixmap() and child.toolTip():
                                    output_image_path = child.toolTip()
                                    if os.path.exists(output_image_path):
                                        break

                        # Calculate row height based on available space
                        row_height = min(
                            available_height - gap, 80
                        )  # Max 80 points per entry

                        # Store position for the row
                        x_pos = pdf.get_x()
                        y_pos = pdf.get_y()

                        # Original image cell
                        pdf.set_xy(x_pos, y_pos)
                        if input_image_path and os.path.exists(input_image_path):
                            try:
                                pdf.cell(col_widths[0], row_height, "", 1, 0)
                                # Calculate dimensions for the image while maintaining aspect ratio
                                img_width = col_widths[0] - 6
                                img_height = row_height - 6
                                pdf.image(
                                    input_image_path,
                                    x=x_pos + 3,
                                    y=y_pos + 3,
                                    w=img_width,
                                    h=img_height,
                                )
                            except Exception as e:
                                pdf.set_xy(x_pos, y_pos)
                                pdf.cell(
                                    col_widths[0],
                                    row_height,
                                    f"Image Error: {str(e)}",
                                    1,
                                    0,
                                    "C",
                                )
                        else:
                            pdf.cell(col_widths[0], row_height, "No Image", 1, 0, "C")

                        # Analysis text cell
                        pdf.set_xy(x_pos + col_widths[0], y_pos)

                        # Draw cell border first
                        pdf.rect(
                            x_pos + col_widths[0], y_pos, col_widths[1], row_height
                        )

                        # Add text with multi_cell inside the predefined boundaries
                        pdf.set_xy(
                            x_pos + col_widths[0] + 2, y_pos + 2
                        )  # Small padding
                        pdf.multi_cell(
                            col_widths[1] - 4, line_height, process_text, 0, "L"
                        )

                        # Result cell
                        pdf.set_xy(x_pos + col_widths[0] + col_widths[1], y_pos)

                        if output_image_path and os.path.exists(output_image_path):
                            try:
                                pdf.cell(col_widths[2], row_height, "", 1, 0)
                                # Add status text at the top of the cell
                                pdf.set_xy(
                                    x_pos + col_widths[0] + col_widths[1], y_pos + 2
                                )
                                pdf.set_font("Arial", "B", 10)
                                pdf.cell(col_widths[2], 6, output_status, 0, 0, "C")
                                pdf.set_font("Arial", "", 10)

                                # Calculate dimensions for result image
                                img_width = col_widths[2] - 6
                                img_height = row_height - 12  # Leave space for text

                                # Position image below the text
                                pdf.image(
                                    output_image_path,
                                    x=x_pos + col_widths[0] + col_widths[1] + 3,
                                    y=y_pos + 8,  # Position below status text
                                    w=img_width,
                                    h=img_height,
                                )
                            except Exception as e:
                                pdf.set_xy(x_pos + col_widths[0] + col_widths[1], y_pos)
                                pdf.cell(
                                    col_widths[2],
                                    row_height,
                                    f"{output_status}\nError: {str(e)}",
                                    1,
                                    0,
                                    "C",
                                )
                        else:
                            pdf.set_font("Arial", "B", 12)
                            pdf.cell(
                                col_widths[2], row_height, output_status, 1, 0, "C"
                            )
                            pdf.set_font("Arial", "", 10)

                        # Move to next row position
                        pdf.set_xy(x_pos, y_pos + row_height + gap)

            # Save PDF
            now = datetime.now()
            formatted_date = now.strftime("%Y-%m-%d")
            formatted_time = now.strftime("%H-%M-%S")
            report_filename = f"NeuralJAM_Report_{formatted_date}_{formatted_time}.pdf"
            report_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Report As",
                report_filename,
                "PDF Files (*.pdf)",
            )

            if report_path:
                pdf.output(report_path)
                self.show_custom_popup(
                    "Success", f"Report generated successfully at:\n{report_path}"
                )
            else:
                self.show_custom_popup("Cancelled", "Report generation was cancelled.")

        except Exception as e:
            self.show_custom_popup("Error", f"Failed to generate report:\n{str(e)}")

    def update_ipo_table(self):
        """Update the IPO table based on current filters, grouping and sorting"""
        try:
            # Get current selections
            filter_text = self.filterByComboBox.currentText()
            group_text = self.groupByComboBox.currentText()
            sort_text = self.sortByComboBox.currentText()

            # Collect all relevant image data
            all_images = []

            # Handle grouping first
            if group_text == "--Select--" or group_text == "All Content":
                # Show all images
                for folder in self.folders:
                    all_images.extend(self.folder_data[folder]["images"])
                all_images.extend(
                    [
                        img
                        for img in self.hate_speech_images + self.non_hate_speech_images
                        if img["type"] == "single"
                    ]
                )
            else:
                # Show only images from selected folder
                folder_path = self.groupByComboBox.currentData()
                if folder_path and folder_path in self.folder_data:
                    all_images = self.folder_data[folder_path]["images"]

            # Apply filtering
            if filter_text == "Hate":
                all_images = [img for img in all_images if img["score"] > 0.3]
            elif filter_text == "Not hate":
                all_images = [img for img in all_images if img["score"] <= 0.3]

            # Apply sorting
            if sort_text == "Hate Speech (High to Low)":
                all_images.sort(key=lambda x: x["score"], reverse=True)
            elif sort_text == "Hate Speech (Low to High)":
                all_images.sort(key=lambda x: x["score"])
            elif sort_text == "Most Recent":
                all_images.sort(key=lambda x: x["date"], reverse=True)
            elif sort_text == "Oldest":
                all_images.sort(key=lambda x: x["date"])

            # Clear and repopulate table
            self.populate_ipo_table_with_data(all_images)

        except Exception as e:
            print(f"Error updating IPO table: {str(e)}")
            self.show_custom_popup("Error", f"Failed to update table: {str(e)}")

    def populate_ipo_table_with_data(self, image_data_list):
        """Helper method to populate the IPO table with given data"""
        self.ipo_table.setRowCount(0)

        for img_data in image_data_list:
            row = self.ipo_table.rowCount()
            self.ipo_table.insertRow(row)
            self.ipo_table.setRowHeight(row, 250)

            self.ipo_table.setItem(
                row, 0, QTableWidgetItem(img_data["path"])
            )  # Input column
            self.ipo_table.setItem(
                row, 2, QTableWidgetItem(img_data["path"])
            )  # Output column

            # Input column - original image
            input_container = QWidget()
            input_layout = QVBoxLayout(input_container)
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
            process_text = "<pre style='font-size: 14pt;'>"
            process_text += "PROCESSING INFORMATION:\n"
            process_text += f"📅 Date & Time Processed: {img_data['date']}\n"
            process_text += f"📄 File: {os.path.basename(img_data['path'])}\n"

            if img_data["type"] == "folder":
                process_text += "🗃️ File Type: Folder\n"
                process_text += f"📁 Folder: {os.path.basename(os.path.dirname(img_data['path']))}\n\n"
            else:
                process_text += "🗃️ File Type: Single Image\n"
                process_text += "\n"

            process_text += "TEXT EXTRACTION:\n"
            process_text += "🔍 Performing OCR text extraction...\n"

            text = self.easy_ocr.extract_text(img_data["path"])

            if text == "":
                process_text += "📝 Extracted Text:\nNo text detected\n\n"
            else:
                process_text += f"📝 Extracted Text:\n{text}\n\n"

            process_text += "HATE SPEECH ANALYSIS:\n"
            process_text += f"📊 Prediction Score: {img_data['score']:.4f}\n"
            process_text += f"ℹ️ Interpretation: {img_data['interpretation']}\n"
            process_text += f"🚨 Conclusion: {'HATE SPEECH DETECTED' if img_data['score'] > 0.3 else 'No hate speech detected'}\n"
            process_text += f"\n📍 Path: {img_data['path']}"

            process_text_edit.setText(process_text)
            process_layout.addWidget(process_text_edit)
            self.ipo_table.setCellWidget(row, 1, process_widget)

            # Output column - blurred if hate speech
            output_widget = QWidget()
            output_layout = QVBoxLayout(output_widget)
            output_widget.setStyleSheet("background-color: #F8FAFC;")

            # Status Label
            is_hate = img_data["score"] > 0.3
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
                    "interpretation": res["interpretation"],
                    "date": QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss"),
                }

                # Update image lists
                if res["score"] > 0.3:
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

            self.update_group_by_combo_box()
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
                "interpretation": result["interpretation"],
                "date": result["date"],
            }

            # Update data structures
            if result["score"] > 0.3:
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

    def create_themed_image_label(self, pixmap, size=(400, 400)):
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
                high_risk = "#8B0000"
                medium_risk = "#8B4513"
                low_risk = "#006400"
                text_color = "#FFFFFF"
            elif self.theme_manager.current_theme == "light_blue":
                high_risk = "#FF6B6B"
                medium_risk = "#FFA07A"
                low_risk = "#90EE90"
                text_color = "#000000"
            else:  # Default theme
                high_risk = "#FF9C9C"
                medium_risk = "#FFD59C"
                low_risk = "#9CFF9C"
                text_color = "#000000"

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GalleryApplication()
    window.show()
    sys.exit(app.exec())
