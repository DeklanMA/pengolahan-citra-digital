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
        # OperasiTITIK
        self.button_loadcitra.clicked.connect(self.fungsi)
        self.button_prosescitra.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Streching.triggered.connect(self.contrastStreching)
        self.actionnegative.triggered.connect(self.negative)
        self.actionoperasi_biner.triggered.connect(self.biner)
        # Histogram
        self.actionhistogram_Grayscale.triggered.connect(self.histogramgrayscale)
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogramClicked)
        self.actionHistogram_Equalization.triggered.connect(self.EqualHistogramClicked)
        # Operasi Geometri
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action90_derajat.triggered.connect(self.rotasi90derajat)
        self.action_90_derajat.triggered.connect(self.rotasimin90derajat)
        self.action45_derajat.triggered.connect(self.rotasi45derajat)
        self.action_45_derajat.triggered.connect(self.rotasimin45derajat)
        self.action180_derajat.triggered.connect(self.rotasi180derajat)
        self.action2x.triggered.connect(self.zoom2x)
        self.action3x.triggered.connect(self.zoom3x)
        self.action4x.triggered.connect(self.zoom4x)
        self.action1_2.triggered.connect(self.zoomsatuperdua)
        self.action1_4.triggered.connect(self.zoomsatuperempat)
        self.action3_4.triggered.connect(self.zoomtigaperempat)
        self.action900x400.triggered.connect(self.dimensi900x400)
        self.actionCrop.triggered.connect(self.cropimage)
        # Operasi Arimatika
        self.actionTambah_Dan_Kurang.triggered.connect(self.aritmatika)
        self.actionKali_dan_Bagi.triggered.connect(self.aritmatika2)
        # Operasi Bolean
        self.actionOperasi_AND.triggered.connect(self.operasiAND)
        # operasi spasial
        self.actionKonvolusi_A.triggered.connect(self.FilteringCliked)
        self.actionKonvolusi_B.triggered.connect(self.Filterring2)
        self.actionKernel_1_9.triggered.connect(self.Mean3x3)
        self.actionKernel_1_4.triggered.connect(self.Mean2x2)
        self.actionGaussian_Filter.triggered.connect(self.Gaussian)
        self.actionke_i.triggered.connect(self.Sharpening1)
        self.actionke_ii.triggered.connect(self.Sharpening2)
        self.actionke_iii.triggered.connect(self.Sharpening3)
        self.actionke_iv.triggered.connect(self.Sharpening4)
        self.actionke_v.triggered.connect(self.Sharpening5)
        self.actionke_vi.triggered.connect(self.Sharpening6)
        self.actionLaplace.triggered.connect(self.Laplace)
    
        
    def fungsi(self):
        print('Printed')
        self.Image = cv2.imread('3.jpg')
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
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BAYER_BG2GRAY)
        except:
            pass
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

    def translasi(self):
        h,w=self.Image.shape[:2]
        quarter_h,quarter_w=h/4,w/4
        T=np.float32([[1,0,quarter_w],[0,1,quarter_h]])
        Img=cv2.warpAffine(self.Image,T,(w,h))
        self.Image = Img
        self.displayImage(2)
     
    def rotasi90derajat(self):
        self.rotasi(90)    
    def rotasimin90derajat(self):
        self.rotasi(-90)     
        
    def rotasimin45derajat(self):
        self.rotasi(-45)   
    def rotasi45derajat(self):
        self.rotasi(45)
        
    def rotasi180derajat(self):
        self.rotasi(180)            
    def rotasi(self, degree):
        h, w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, .7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (h,
    w))
        self.Image=rot_image    
        self.displayImage(2) 
        
    def zoom2x(self):
         self.zoomIn(2)   
    def zoom3x(self):
         self.zoomIn(3)  
    def zoom4x(self):
         self.zoomIn(4)   
    def zoomIn(self, skala):
        resize_Image = cv2.resize(self.Image, None, fx = skala , fy = skala,interpolation = cv2.INTER_CUBIC)
        cv2.imshow('original', self.Image)
        cv2.imshow('Zoom In', resize_Image)
        cv2.waitKey()
    
    def zoomsatuperdua(self):
         self.zoomOut(1/2)
    def zoomsatuperempat(self):
         self.zoomOut(1/4)
    def zoomtigaperempat(self):
         self.zoomOut(3/4)
    def zoomOut(self, skala):
        resize_Image = cv2.resize(self.Image, None, fx = skala , fy = skala)
        cv2.imshow('original', self.Image)
        cv2.imshow('Zoom oun', resize_Image)
        cv2.waitKey()
        
    def dimensi900x400(self):
        resize_Image=cv2.resize(self.Image,(900,400),interpolation=cv2.INTER_AREA)
        cv2.imshow('Dimensi 900x400', resize_Image)
        cv2.waitKey()
        
    def cropimage(self):
        h, w = self.Image.shape[:2]
        #get the strating point of pixel coord(top left)
        start_row, start_col=int(h*.1),int(w*.1)
        #get the ending point coord (botoom right)
        end_row, end_col=int(h*.5),int(w*.5)
        crop=self.Image[start_row:end_row,start_col:end_col]
        cv2.imshow('Original',self.Image)
        cv2.imshow('Crop Image',crop)
        
    def aritmatika(self):
        Image1 = cv2.imread('3.jpg', 0)
        Image2 = cv2.imread('4.jpg', 0)
        Image_tambah = Image1 + Image2
        Image_kurang = Image1 - Image2
        cv2.imshow('image 1 original', Image1)
        cv2.imshow('image 2 original', Image2)
        cv2.imshow('image Tambah', Image_tambah)
        cv2.imshow('image kurang', Image_kurang)
        cv2.waitKey()
        
    def aritmatika2(self):
        Image1 = cv2.imread('3.jpg', 0)
        Image2 = cv2.imread('4.jpg', 0)
        Image_Kali = Image1 * Image2
        Image_Bagi = Image1 / Image2
        cv2.imshow('image 1 original', Image1)
        cv2.imshow('image 2 original', Image2)
        cv2.imshow('image Tambah', Image_Kali)
        cv2.imshow('image kurang', Image_Bagi)
        cv2.waitKey()
     
    def operasiAND(self):   
        Image1 = cv2.imread('3.jpg', 1)
        Image2 = cv2.imread('4.jpg', 1)
        Image1=cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)
        Image2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2RGB)
        op_and=cv2.bitwise_and(Image1, Image2)
        cv2.imshow('image 1 original', Image1)
        cv2.imshow('image 2 original', Image2)
        cv2.imshow('image Operasi AND', op_and)
        cv2.waitKey()

    def Konvolusi(self, X,F): #Fungsi konvolusi 2D
        X_height = X.shape[0] #membaca ukuran tinggi dan lebar citra
        X_width = X.shape[1]

        F_height = F.shape[0] #membaca ukuran tinggi dan lebar kernel
        F_width = F.shape[1]

        H = (F_height) // 2
        W = (F_width) // 2

        out = np.zeros((X_height, X_width))

        for i in np.arange(H + 1, X_height - H): #mengatur pergerakan karnel
            for j in np.arange(W + 1, X_width - W):
                sum = 0
                for k in np.arange(-H, H + 1):
                    for l in np.arange(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a) #menampung nilai total perkalian w kali a
                out[i,j] = sum  #menampung hasil
        return out

    def FilteringCliked(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        kernel = np.array(
            [
                [1, 1, 1],    #array kernelnya
                [1, 1, 1],
                [1, 1, 1]
            ])

        img_out = self.Konvolusi(img, kernel)
    #    print('---Nilai Pixel Filtering A--- \n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show() #Menampilkan gambar
        
    def Filterring2(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        kernel = np.array(
            [
                [6, 0, -6],  # array kernelnya
                [6, 1, -6],
                [6, 0, -6]
            ]
        )

        img_out = self.Konvolusi(img, kernel)
    #    print('---Nilai Pixel Filtering B---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()  # Menampilkan gambar
        
    def Mean2x2(self): #Fungsi Mean  2x2       #D2
        mean = (1.0 / 4) * np.array(  # Penapis rerata 1/4
            [  # array kernel 3x3
                [0, 0, 0],
                [0, 1, 1],
                [0, 1, 1]
            ]
        )
        img = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        img_out = self.Konvolusi(img, mean)
        print('---Nilai Pixel Mean Filter 2x2 ---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()
        
    def Mean3x3(self):  # Fungsi Mean 3x3         #D2
        mean = (1.0 / 9) * np.array(  # Penapis rerata 1/9
            [ # array kernel 3x3
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]
        )
        img = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        img_out = self.Konvolusi(img, mean)
        print('---Nilai Pixel Mean Filter 3x3---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()
    
    def Gaussian(self):         #D3
        gausian = (1.0 / 345) * np.array(
            [ #Kernel gaussian
                [1, 5, 7, 5, 1],
                [5, 20, 33, 20, 5],
                [7, 33, 55, 33, 7],
                [5, 20, 33, 20, 5],
                [1, 5, 7, 5, 1]
            ]
        )
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        img_out = self.Konvolusi(img, gausian)
        print('---Nilai Pixel Gaussian ---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Sharpening1(self):    #D4
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        cv2.imshow('Original', img)
        print('---Nilai Pixel Kernel i ---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()
        
    def Sharpening2(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1]
            ])
        img_out = self.Konvolusi(img, sharpe)
        cv2.imshow('Original', img)
        print('---Nilai Pixel Kernel ii ---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Sharpening3(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        print('---Nilai Pixel Kernel iii ---\n', img_out)
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Sharpening4(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [1, -2, 1],
                [-2, 5, -2],
                [1, 2, 1]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        print('---Nilai Pixel Kernel iv ---\n', img_out)
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Sharpening5(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [1, -2, 1],
                [-2, 4, -2],
                [1, -2, 1]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        print('---Nilai Pixel Kernel v ---\n', img_out)
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Sharpening6(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        print('---Nilai Pixel Kernel vi ---\n', img_out)
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Laplace(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        sharpe = (1.0 / 16) * np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0]
            ])
        img_out = self.Konvolusi(img, sharpe)
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()


        
    def displayImage(self, windows):
        qformat = QImage.Format_Indexed8
        if len(self.Image.shape)==3:
            if(self.Image.shape[2])==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)
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