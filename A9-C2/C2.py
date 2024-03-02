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
        self.actionTambah_Dan_Kurang.triggered.connect(self.aritmatika)
        self.actionKali_dan_Bagi.triggered.connect(self.aritmatika2)
        self.actionOperasi_AND.triggered.connect(self.operasiAND)
        self.actionOperasi_OR.triggered.connect(self.operasiOR)
        self.actionOperasi_XOR.triggered.connect(self.operasiXOR)
        self.actionOperasi_NOT.triggered.connect(self.operasinot)
        self.actionTranspose.triggered.connect(self.Transpose)

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

    def Transpose(self):
        #    trans_img = cv2.transpose(self.Image)
        #    self.Image = trans_img
        #    self.displayImage(2)
        img = cv2.imread('4.jpg')
        window_nama = 'Image Hasil'
        image = cv2.transpose(img)
        cv2.imshow(window_nama, image)
        cv2.waitKey(0)
        
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
        cv2.imshow('image kali', Image_Kali)
        cv2.imshow('image bagi', Image_Bagi)
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

    def operasiOR(self):
        Image1 = cv2.imread('3.jpg', 1)
        Image2 = cv2.imread('4.jpg', 1)
        Image1=cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)
        Image2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2RGB)
        op_and=cv2.bitwise_or(Image1, Image2)
        cv2.imshow('image 1 original', Image1)
        cv2.imshow('image 2 original', Image2)
        cv2.imshow('image Operasi AND', op_and)
        cv2.waitKey()

    def operasiXOR(self):
        Image1 = cv2.imread('3.jpg', 1)
        Image2 = cv2.imread('4.jpg', 1)
        Image1=cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)
        Image2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2RGB)
        op_and=cv2.bitwise_xor(Image1, Image2)
        cv2.imshow('image 1 original', Image1)
        cv2.imshow('image 2 original', Image2)
        cv2.imshow('image Operasi AND', op_and)
        cv2.waitKey()

    def operasinot(self):
        Image1 = cv2.imread('3.jpg', 1)
        Image2 = cv2.imread('4.jpg', 1)
        Image1=cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)
        Image2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2RGB)
        op_and=cv2.bitwise_not(Image1, Image2)
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
        plt.show()   #Menampilkan gambar
        
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