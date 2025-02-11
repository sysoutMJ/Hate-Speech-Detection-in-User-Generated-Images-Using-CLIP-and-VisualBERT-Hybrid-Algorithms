import sys

from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel, QFrame, QFileDialog
from PyQt6.QtCore import Qt
# Get the UI class from the converted .ui to .py
from main_ui import Ui_MainWindow

# QMainWindow is a class from PyQt6 that generates windows application.
# This acts like a blank notebook for us. In here, you have the freedom to put any widgets you want.

# Ui_MainWindow is the class that contains the design you made using Qt creator.
# In here, merong dalawa right na required na input ng class. This is called MULTIPLE INHERITANCE.
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args,**kwargs):
        
        self.image_counter = 0
        self.frame_label_counter = 0
        
        # *args and **kwargs are passed as parameter. Why? It is because they allow flexibility in passing arguments when creating the MainWindow instance.
        # This is also needed to future proof your application.
        # Need siya para makapag create ka ng mga custom properties mo through arguments.
        # Kunware using this main window, gusto mo mag create ng another with the same property but with differnt title ganon.

        
        # This is required as this ensures that parent class (QMainWindow) runs first so that the windows are being set up correctly.
        # This calls the constructor of QMainWindow.
        # Diba ang constructor is like a method na laging natatawag and nag e-execute if that class with that said consturctor is called.
        # Example: class myClass(): myClass(){print("hotdog")}
        # So pag ginamit mo yung myClass(), laging ma-e-execute yung myClass(), which is the constructor. 
        # super is used para ma-inherit ng lahat yung methods and properties ng parent. In this case, para magamit yung lahat ng meron si QMainWindow.
        super().__init__(*args, **kwargs) 
        
        # setupUi is the method from the converted .ui to .py
        self.setupUi(self)
        
        self.uploadImageButton.clicked.connect(self.create_images_frame)
        # self.toggleButton.clicked.connect()
    
    def testPrint(self):
        print("Annyeong world!")
         
    def create_images_frame(self):
        self.name_of_frame = self.increment_frame_label()
        print(self.name_of_frame)

        # Create a new frame inside scrollAreaWidgetContents
        self.qframe_name = QFrame(parent=self.scrollAreaWidgetContents)
        self.qframe_name.setStyleSheet("background-color: white; border: 1px solid black;")
        self.qframe_name.setFrameShape(QFrame.Shape.StyledPanel)
        self.qframe_name.setFrameShadow(QFrame.Shadow.Raised)
        self.qframe_name.setObjectName(f"{self.name_of_frame}")
        self.qframe_name.setFixedHeight(150) 

        self.verticalLayout_3.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.verticalLayout_3.addWidget(self.qframe_name)

        # Ensure scroll area recognizes new content
        self.scrollAreaWidgetContents.adjustSize()
        
        # Make inside of the created frame an horizontal layout
        # How to adjust kung ilang pics kaya?
        #       if 300 plus na width nadagdag, get pic from the second frame and add it to the first one.
        # Then set an maximum amount of labels allowed per frame.
        # If max labels allowed reach, create new frame. 

        # If created frame does not contain image, do not create another frame. 
        
    def add_photo_inside_frame(self):
        self.name_of_label = self.increment_label()
        print(self.name_of_label)
        
        self.qlabel_name = QLabel(parent=self.frame_1)
        self.choose_photo_from_file_dialog()
        # self.qlabel_name.setStyleSheet("background-color:white;")
        self.qlabel_name.setText("")
        self.qlabel_name.setObjectName(f"{self.name_of_label}")
        # This expects an widget.
        self.horizontalLayout_3.addWidget(self.qlabel_name)
    
    def choose_photo_from_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:  # Check if user selected a file
            print("Selected file:", file_path)
            
        self.qlabel_name.setStyleSheet("background-color:white; background-image: url('{file_path}'); background-repeat: no-repeat; background-position: center; background-size: cover;")
     
    def increment_frame_label(self):
        self.frame_label_counter += 1
        self.frame_name = f"frame{self.frame_label_counter}"
        return self.frame_name
           
    def increment_label(self):
        self.image_counter += 1
        self.name = f"label_for_image{self.image_counter}"
        return self.name
    
    
        


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
        

