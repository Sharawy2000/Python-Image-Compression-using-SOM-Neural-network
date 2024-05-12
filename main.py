import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import qdarkstyle
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import SOM_NN


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Remove the window frame
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Set the dark mode style
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.pointer = 0
        self.flag1 = False
        self.flag2 = False

        # Set window title and size
        self.setWindowTitle('Parallel Processing')
        self.setGeometry(0, 0, 1340, 800)

        # Set window icon
        self.setWindowIcon(QIcon('Images/logo.png'))

        # Create labels to display the selected images
        self.res = QLabel(self)
        self.res.setGeometry(50, 100, 400, 200)
        pixmap = QPixmap('Images/error-image.png')

        self.img = QLabel(self)
        self.img.setGeometry(70, 180, 400, 400)
        self.img.setPixmap(pixmap)
        self.img.setScaledContents(True)

        self.img_com = QLabel(self)
        self.img_com.setGeometry(550, 180, 400, 400)
        self.img_com.setPixmap(QPixmap('Images/error-image.png'))
        self.img_com.setScaledContents(True)

        # Center window on screen
        self.center()

        # create widgets
        self.create_widgets()

    def center(self):
        # Get the screen geometry
        screen = QDesktopWidget().screenGeometry()

        # Calculate the center point
        center_x = (screen.width() - self.width()) // 2
        center_y = (screen.height() - self.height()) // 2

        # Move the window to the center
        self.move(center_x, center_y)

    def create_widgets(self):
        # layout for the main window
        layout = QVBoxLayout(self)

        # Add a label
        self.label = QLabel("Image Compression Project", self)
        self.label.setStyleSheet("font-size: 20pt; font-weight: bold;")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label, alignment=Qt.AlignTop)

        # Add buttons
        self.txt = QLabel("Iterations :", self)
        self.txt.setStyleSheet("font-size: 12pt;")
        self.txt.move(1000, 60)

        self.iters_input = QSpinBox(self)
        self.iters_input.move(1080, 55)
        self.iters_input.setMinimum(1000)
        self.iters_input.setMaximum(20000)
        self.iters_input.resize(175, 40)
        self.iters_input.setStyleSheet("font-size: 12pt;")

        #grid input

        self.grid_txt = QLabel("Grid :", self)
        self.grid_txt.setStyleSheet("font-size: 12pt;")
        self.grid_txt.move(1090, 130)

        # left grid
        self.Left_grid_input = QSpinBox(self)
        self.Left_grid_input.move(1150, 125)
        self.Left_grid_input.setMinimum(1)
        self.Left_grid_input.setValue(10)
        self.Left_grid_input.setMaximum(100)
        self.Left_grid_input.resize(60, 40)
        self.Left_grid_input.setStyleSheet("font-size: 12pt;")

        # right grid
        self.Right_grid_input = QSpinBox(self)
        self.Right_grid_input.move(1220, 125)
        self.Right_grid_input.setMinimum(1)
        self.Right_grid_input.setValue(10)
        self.Right_grid_input.setMaximum(100)
        self.Right_grid_input.resize(60, 40)
        self.Right_grid_input.setStyleSheet("font-size: 12pt;")

        ######################

        # grid input

        self.learning_txt = QLabel("Learning :", self)
        self.learning_txt.setStyleSheet("font-size: 12pt;")
        self.learning_txt.move(990, 205)

        self.learning_input = QDoubleSpinBox(self)
        self.learning_input.move(1070, 200)
        self.learning_input.setMinimum(0.1)
        self.learning_input.setValue(0.1)
        self.learning_input.setMaximum(1.0)
        self.learning_input.resize(60, 40)
        self.learning_input.setStyleSheet("font-size: 12pt;")

        self.sigma_txt = QLabel("Sigma :", self)
        self.sigma_txt.setStyleSheet("font-size: 12pt;")
        self.sigma_txt.move(1150, 205)

        # right grid
        self.Sigma_input = QDoubleSpinBox(self)
        self.Sigma_input.move(1220, 200)
        self.Sigma_input.setMinimum(0.1)
        self.Sigma_input.setValue(10)
        self.Sigma_input.setMaximum(10.0)
        self.Sigma_input.resize(60, 40)
        self.Sigma_input.setStyleSheet("font-size: 12pt;")

        ######################
        self.default_iters = QCheckBox("Default", self)
        self.default_iters.move(980, 130)
        self.default_iters.setChecked(False)
        self.default_iters.clicked.connect(self.check_iters_button)

        self.select_file_button = QPushButton('Select Image', self)
        self.select_file_button.setStyleSheet("font-size: 12pt;")
        self.select_file_button.resize(175, 50)
        self.select_file_button.move(1050, 300)
        self.select_file_button.clicked.connect(self.select_file)

        self.compress_button = QPushButton('Compress Image', self)
        self.compress_button.setStyleSheet("font-size: 12pt;")
        self.compress_button.resize(175, 50)
        self.compress_button.move(1050, 400)
        self.compress_button.clicked.connect(self.SOM)

        self.performance_button = QPushButton('Performance Measures ', self)
        self.performance_button.setStyleSheet("font-size: 12pt;")
        self.performance_button.resize(175, 50)
        self.performance_button.move(1050, 500)
        self.performance_button.clicked.connect(self.som_metrics)

        self.Display_button = QPushButton('Display', self)
        self.Display_button.setStyleSheet("font-size: 12pt;")
        self.Display_button.resize(175, 50)
        self.Display_button.move(1050, 600)
        self.Display_button.clicked.connect(self.show_fig)

        self.close_button = QPushButton('Quit', self)
        self.close_button.setStyleSheet("font-size: 12pt;")
        self.close_button.resize(175, 50)
        self.close_button.move(1050, 700)
        self.close_button.clicked.connect(self.close)

        self.dark = QRadioButton("Dark mode", self)
        self.dark.move(100, 700)
        self.dark.setChecked((True))
        self.dark.toggled.connect(self.setDark)

        self.light = QRadioButton("Light mode", self)
        self.light.move(100, 730)
        self.light.toggled.connect(self.setLight)

    # Define a method to select an image
    def select_file(self):
        self.filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')

        # If an image is selected, display it
        if self.filename:
            pixmap = QPixmap(self.filename)
            self.img.setPixmap(pixmap)
            self.img.setScaledContents(True)

            self.flag1 = True

            errcomimg = QPixmap('Images/error-compress.png')
            self.img_com.setPixmap(errcomimg)
            self.img_com.setScaledContents(True)

    # Define a method to perform K-means clustering on the image
    def SOM(self):
        if self.flag1:
            loading = QPixmap('Images/loading.jpg')
            self.img_com.setPixmap(loading)
            self.img_com.setScaledContents(True)
            self.img_com.show()
            self.flag2 = True

            if self.default_iters.isChecked():

                # Default value
                Som_NN(self.filename, num_iterations=10000, left_grid=10, right_grid=10, sigma=10, learning=0.1)

            else:

                Som_NN(self.filename, num_iterations=self.iters_input.value(), left_grid=self.Left_grid_input.value(),
                       right_grid=self.Right_grid_input.value(), sigma=self.Sigma_input.value()
                       , learning=self.learning_input.value())

            com = QPixmap("Results/compressed_image.jpg")
            self.img_com.setPixmap(com)
            self.img_com.setScaledContents(True)

            self.orgi = QLabel(self)
            self.orgi.setText('Original Image ({} iterations) (size: {:.2f} KB)')
            self.orgi.setGeometry(100, 890, 700, 50)
            self.orgi.setStyleSheet("font-size: 16pt; font-weight: bold;")
            self.orgi.show()
            self.pcom = QLabel(self)
            self.pcom.setText('Compressed Image ({} iterations) (size: {:.2f} KB)')
            self.pcom.setGeometry(950, 890, 700, 50)
            self.pcom.setStyleSheet("font-size: 16pt; font-weight: bold;")
            self.pcom.show()

            self.timetxt = QLabel("Time:", self)
            self.timetxt.setStyleSheet("font-size: 10pt;")
            self.timetxt.move(1650, 800)
            self.timetxt.show()
            self.tim = QLabel(self)
            # self.tim.setText(f'{minutes} min {seconds} sec')
            self.tim.move(1650, 825)
            self.tim.setStyleSheet("font-size: 10pt;")
            self.tim.show()

    # Desplay the WCSS curve
    def som_metrics(self):
        try:
            if self.filename is not None:

                plot_som_metrics(self.filename)

        except NameError:
            print("image is not defined")


    # Define a function for checking the state of the radio button
    def check_iters_button(self):
        if self.default_iters.isChecked():
            self.iters_input.hide()
            self.txt.hide()
            self.grid_txt.hide()
            self.Left_grid_input.hide()
            self.Right_grid_input.hide()
            self.learning_input.hide()
            self.learning_txt.hide()
            self.sigma_txt.hide()
            self.Sigma_input.hide()
            return True
        self.iters_input.show()
        self.txt.show()
        self.grid_txt.show()
        self.Left_grid_input.show()
        self.Right_grid_input.show()
        self.learning_input.show()
        self.learning_txt.show()
        self.sigma_txt.show()
        self.Sigma_input.show()

        return False

    def show_fig(self):
        try:
            if fig is not None:

                image = Image.open("Results/figure.jpg")

                # Show the image
                image.show()

        except NameError:
            print("fig is not defined")

    # Define a function for setting the dark theme
    def setDark(self):
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # Define a function for setting the light theme
    def setLight(self):
        self.setStyleSheet('')


# org_colors, org_size, n_clusters, compressed_img_size, it, minutes, seconds = None, None, None, None, 1, 0, 0

def Som_NN(path, num_iterations, left_grid, right_grid, sigma, learning):

    global som , fig
    img = plt.imread(path)

    som = SOM_NN.SOM(input_size=3, output_size=(left_grid, right_grid), num_iterations=num_iterations, learning_rate=learning, sigma=sigma)
    som.compress_image(input_image_path=path, output_image_path='Results/compressed_image.jpg')
    compressed_img = plt.imread("Results/compressed_image.jpg")

    # Get the file sizes of the original and compressed images in bytes
    original_size_bytes = os.path.getsize(path)
    compressed_size_bytes = os.path.getsize("Results/compressed_image.jpg")

    # Convert file sizes to kilobytes
    original_size_kb = original_size_bytes / 1024
    compressed_size_kb = compressed_size_bytes / 1024

    # global fig

    # Plot the original and compressed images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original image
    ax[0].imshow(img)
    ax[0].set_title(f'Original Image\nSize: {original_size_kb:.2f} KB')
    ax[0].axis('off')

    # Display the compressed image
    ax[1].imshow(compressed_img)
    ax[1].set_title(f'Compressed Image\nSize: {compressed_size_kb:.2f} KB')
    ax[1].axis('off')

    # Save the figure
    plt.savefig("Results/figure.jpg")
    print("Done :)")


def plot_som_metrics(path):

    plt.close()

    # Load the original and compressed images
    img = plt.imread(path)
    compressed_img = plt.imread("Results/compressed_image.jpg")

    # Get the file sizes of the original and compressed images in bytes
    original_size_bytes = os.path.getsize(path)
    compressed_size_bytes = os.path.getsize("Results/compressed_image.jpg")

    # Convert file sizes to kilobytes
    original_size_kb = original_size_bytes / 1024
    compressed_size_kb = compressed_size_bytes / 1024

    # Calculate Compression Ratio
    compression_ratio = original_size_kb / compressed_size_kb

    data = np.array(img) / 255.0  # Normalize pixel values
    data = data.reshape(-1, 3)  # Reshape data

    # Calculate Quantization Error (QE)
    qe_sum = 0.0

    for data_point in data:
        bmu_index = som.find_bmu(data_point)
        bmu = som.weights[bmu_index]
        qe_sum += np.linalg.norm(data_point - bmu)
    quantization_error = qe_sum / len(data)


    # Plot the compressed image and display metrics
    fig, ax = plt.subplots(figsize=(8, 6))

    # Display the compressed image
    ax.imshow(compressed_img)
    ax.set_title(f'Compressed Image\nSize: {compressed_size_kb:.2f} KB')

    # Add metrics as text annotations
    textstr = f'Compression Ratio: {compression_ratio:.2f}\n Quantizarion Error: {quantization_error:.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    # Hide axis
    ax.axis('off')

    plt.savefig("Results/metrics.jpg")

    plt.show()


if __name__ == '__main__':
    # Create a QApplication instance
    app = QApplication(sys.argv)
    # Create an instance of our window
    window = MyWindow()
    # Show the window
    window.show()
    # Start the event loop and exit the application when the loop is finished
    sys.exit(app.exec_())
