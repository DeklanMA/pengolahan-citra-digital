import sys
import cv2
from PyQt5 import QtCore,  QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import  loadUi

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('citra.ui', self)
        self.Image = None
        self.button_loadcitra.clicked.connect(self.fungsi)

    def fungsi(self):
        self.Image = cv2.imread('2.jfif')
        self.displayImage()

    def displayImage(self):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape)==3:
            if(self.Image.shape[2])==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)

        img = img.rgbSwapped()

        self.label.setPixmap(QPixmap.fromImage(img))

app = QtWidgets .QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('A2')
window.show()
sys.exit(app.exec_())