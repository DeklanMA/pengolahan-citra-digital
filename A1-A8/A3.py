import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('citra.ui', self)
        self.Image = None
        self.button_loadcitra.clicked.connect(self.fungsi)
        self.button_prosescitra.clicked.connect(self.grayscale)
    def fungsi(self):
        print('Printed')
        self.Image = cv2.imread('2.jfif')
        self.displayImage(1)

    def grayscale(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] + 0.587
                                     * self.Image[i, j, 1] + 0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        self.displayImage(2)

    def displayImage(self, windows):

        qformat = QImage.Format_Indexed8

        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0],
                     self.Image.strides[0], qformat)

        img = img.rgbSwapped()

        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)

        if windows == 2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Image 1')
window.show()
sys.exit(app.exec_())