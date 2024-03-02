import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt 

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('citra.ui', self)
        self.Image = None
        self.button_loadcitra.clicked.connect(self.fungsi)
        self.button_prosescitra.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Streching.triggered.connect(self.contrastStreching)
        self.actionnegative.triggered.connect(self.negative)
        self.actionoperasi_biner.triggered.connect(self.biner)
        self.actionhistogram_Grayscale.triggered.connect(self.histogramgrayscale)
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogramClicked)
        self.actionHistogram_Equalization.triggered.connect(self.EqualHistogramClicked)
        
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

    def brightness(self):

        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BAYER_BG2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        brightness = 80
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + brightness, 0, 255)

                self.Image.itemset((i, j), b)
        self.displayImage(1)

    def contrast(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BAYER_BG2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        contrast = 1.7
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a * contrast, 0, 255)

                self.Image.itemset((i, j), b)
        self.displayImage(1)

    def contrastStreching(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BAYER_BG2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        minV = np.min(self.Image)
        maxV = np.max(self.Image)

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = float(a - minV) / (maxV - minV) * 255

                self.Image.itemset((i, j), b)
        self.displayImage(1)
        
    def negative(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W, 3), np.uint8)
        for i in range(H):
            for j in range(W):
                for k in range(3):
                    gray [i, j, k] = np.clip(255 - self.Image[i, j, k], 0, 255)
        self.Image = gray
        self.displayImage(2)
        
    def biner(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BAYER_BG2GRAY)
        except:
            pass
        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                if a == 180 : 
                    a = 0
                elif a<180 : 
                    a=1
                else : 
                    a = 255
                self.Image.itemset((i, j), a)
        self.displayImage(1)  
        
    def histogramgrayscale(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] + 0.587
                                     * self.Image[i, j, 1] + 0.114 
                                     * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        self.displayImage(2)
        plt.hist(self.Image.ravel(), 255, [0, 255])
        plt.show()
        
    def RGBHistogramClicked(self):
        color = ('b', 'g', 'r')
        for i,col in enumerate(color):
                histo=cv2.calcHist([self.Image],[i],None,[256],[0,256])
                plt.plot(histo,color=col)
                plt.xlim([0,256])
                self.displayImage(2)
                plt.show()

    def EqualHistogramClicked(self):
        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() -
    cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.Image = cdf[self.Image]
        self.displayImage(2)

        plt.plot(cdf_normalized, color='b')
        plt.hist(self.Image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

          
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