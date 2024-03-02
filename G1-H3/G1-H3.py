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
        self.actionMedianFilter.triggered.connect(self.Median)
        self.actionMaxFilter.triggered.connect(self.Max)
        self.actionMinFilter.triggered.connect(self.Min)
        # Tranformasi Fourier Diskrit
        self.actionDFT_Smooting_Image.triggered.connect(self.SmothImage)
        self.actionDFT_Edge_Detection.triggered.connect(self.EdgeDetec)
        # Deteksi Tepi
        self.actionOperasi_Sobel.triggered.connect(self.Opsobel)
        self.actionOperasi_Prewitt.triggered.connect(self.Opprewitt)
        self.actionOperasi_Robert.triggered.connect(self.Oprobert)
        self.actionOperasi_Canny.triggered.connect(self.OpCanny)
        # Morfologi
        self.actionDelasi.triggered.connect(self.MortlgiDilasi)
        self.actionErosi.triggered.connect(self.MorflgiErosi)
        self.actionOpening.triggered.connect(self.MorflgiOpening)
        self.actionClosing.triggered.connect(self.MorflgiClosing)
        self.actionSkeletonizing.triggered.connect(self.MorfSkeleton)
        # Segmentasi Citra
        self.actionBinary.triggered.connect(self.Binary)
        self.actionBinary_Invers.triggered.connect(self.BinaryInvers)
        self.actionTrunc.triggered.connect(self.Trunc)
        self.actionTo_Zero.triggered.connect(self.ToZero)
        self.actionTo_Zero_Invers.triggered.connect(self.ToZeroInvers)
        self.actionMean_Thresholding.triggered.connect(self.Meanthres)
        self.actionGaussian_Thresholding.triggered.connect(self.Gausthres)
        self.actionOtsu_Thresholding.triggered.connect(self.Otsuthres)
        self.actionContur.triggered.connect(self.Contour)
        
    def fungsi(self):
        print('Printed')
        self.Image = cv2.imread('po.jpg')
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

    def Median(self): #D5
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY) #mengubah citra ke grayscale
        img_out = img.copy()
        H, W = img.shape[:2] #tinggi dan lebar citra

        for i in np.arange(3, H - 3):
            for j in np.arange(3, W - 3):
                neighbors = [] #menampung nilai pixel
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l) # menampung hasil
                        neighbors.append(a)   #menambahkan a ke neighbors
                neighbors.sort() #untuk mengurutkan neighbors
                median = neighbors[24]
                b = median
                img_out.itemset((i, j), b)
        print('---Nilai Pixel Median Filter---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Max(self): #D6
        img = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        img_out = img.copy()
        H, W = img.shape[:2]

        for i in np.arange(3, H - 3): #mengecek nilai setiap pixel
            for j in np.arange(3, W - 3):
                max = 0
                for k in np.arange(-3, 4): #mencari nilai maximun
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l) #untuk menampung nilai hasil baca pixel
                        if a > max:
                            max = a
                        b = max
                img_out.itemset((i, j), b)
        print('---Nilai Pixel Maximun Filter ---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Min(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        img_out = img.copy()
        H, W = img.shape[:2]

        for i in np.arange(3, H - 3):  # mengecek nilai setiap pixel
            for j in np.arange(3, W - 3):
                min = 255
                for k in np.arange(-3, 4):  # mencari nilai maximun
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)  # untuk menampung nilai hasil baca pixel
                        if a < min:
                            min = a
                        b = min
                img_out.itemset((i, j), b)
        print('---Nilai Pixel Minimun Filter ---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()


        # E1
    def SmothImage(self):
            x = np.arange(256)
            y = np.sin(2 * np.pi * x / 3)

            y += max(y)

            img = np.array([[y[j] * 127 for j in range(256)] for i in
                            range(256)], dtype=np.uint8)

            plt.imshow(img)
            img = cv2.imread('orang.jpeg',0)

            dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))

            rows, cols = img.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols, 2), np.uint8)
            r = 120
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) * 2 + (y - center[1]) * 2 <= r * r

            mask[mask_area] = 1

            fshift = dft_shift * mask
            fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
            f_ishift = np.fft.ifftshift(fshift)

            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(img, cmap='gray')
            ax1.title.set_text('Input Image')
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(magnitude_spectrum, cmap='gray')
            ax2.title.set_text('FFT of Image')
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.imshow(fshift_mask_mag, cmap='gray')
            ax3.title.set_text('FFT + Mask')
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.imshow(img_back, cmap='gray')
            ax4.title.set_text('Inverse Fourier')
            plt.show()

        # E2
    def EdgeDetec(self):
            x = np.arange(256)
            y = np.sin(2 * np.pi * x / 3)

            y += max(y)

            img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)

            plt.imshow(img)
            img = cv2.imread("5.jpg", 0)

            dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))

            rows, cols = img.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.ones((rows, cols, 2), np.uint8)
            r = 120
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0] * 2 + (y - center[1])) * 2 <= r * r

            mask[mask_area] = 1

            fshift = dft_shift * mask
            fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
            f_ishift = np.fft.ifftshift(fshift)

            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(img, cmap='gray')
            ax1.title.set_text('Input Image')
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(magnitude_spectrum, cmap='gray')
            ax2.title.set_text('FFT of Image')
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.imshow(fshift_mask_mag, cmap='gray')
            ax3.title.set_text('FFT + Mask')
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.imshow(img_back, cmap='gray')
            ax4.title.set_text('Inverse fourier')
            plt.show()

            # F1

    def Opsobel(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        X = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
        Y = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]])
        img_Gx = self.Konvolusi(img, X)
        img_Gy = self.Konvolusi(img, Y)
        img_out = np.sqrt((img_Gx * img_Gx) + (img_Gy * img_Gy))
        img_out = (img_out / np.max(img_out)) * 255
        print('---Nilai Pixel Operasi Sobel--- \n', img_out)
        self.Image = img
        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()

    def Opprewitt(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        prewit_X = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])
        prewit_Y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]])
        img_Gx = self.Konvolusi(img, prewit_X)
        img_Gy = self.Konvolusi(img, prewit_Y)
        img_out = np.sqrt((img_Gx * img_Gx) + (img_Gy * img_Gy))
        img_out = (img_out / np.max(img_out)) * 255
        print('---Nilai Pixel Operasi Prewitt --- \n', img_out)
        self.Image = img
        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()

    def Oprobert(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        RX = np.array([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 0]])
        RY = np.array([[0, 1, 0],
                       [-1, 0, 0],
                       [0, 0, 0]])
        img_Gx = self.Konvolusi(img, RX)
        img_Gy = self.Konvolusi(img, RY)
        img_out = np.sqrt((img_Gx * img_Gx) + (img_Gy * img_Gy))
        img_out = (img_out / np.max(img_out)) * 255
        print('---Nilai Pixel Operasi Robert--- \n', img_out)
        self.Image = img
        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()

    def OpCanny(self):
        # Langkah ke 1 (Reduksi Noise0
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        gaus = (1.0 / 57) * np.array(
            [[0, 1, 2, 1, 0],
             [1, 3, 5, 3, 1],
             [2, 5, 9, 5, 2],
             [1, 3, 5, 3, 1],
             [0, 1, 2, 1, 0]])
        img_out = self.Konvolusi(img, gaus)
        img_out = img_out.astype("uint8")
        cv2.imshow("Noise Reduction", img_out)

        # Langkah ke 2 (Finding Gradien)
        Gx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Gy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
        konvolusi_x = self.Konvolusi(img, Gx)
        konvolusi_y = self.Konvolusi(img, Gy)
        theta = np.arctan2(konvolusi_y, konvolusi_x)
        theta = theta.astype("uint8")
        cv2.imshow("Finding Gradien", theta)

        # Langkah Ke 3 (Non Maximum suppression)
        H, W = img.shape[:2]
        Z = np.zeros((H, W), dtype=np.int32)

        angle = theta * 180. / np.pi
        angle[angle < 0] += 180
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255

                    # Angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img_out[i, j + 1]
                        r = img_out[i, j - 1]
                    # Angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img_out[i + 1, j - 1]
                        r = img_out[i - 1, j + 1]
                    # Angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img_out[i + 1, j]
                        r = img_out[i - 1, j]
                    # Angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img_out[i + 1, j - 1]
                        r = img_out[i - 1, j + 1]
                    if (img_out[i, j] >= q) and (img_out[i, j] >= r):
                        Z[i, j] = img_out[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        img_N = Z.astype("uint8")
        cv2.imshow("Non Maximum Supression", img_N)

        # Langkah ke 4 (Hysterisis Tresholding)
        weak = 80
        strong = 110
        for i in np.arange(H):
            for j in np.arange(W):
                a = img_N.item(i, j)
                if (a > weak):
                    b = weak
                    if (a > strong):
                        b = 255
                else:
                    b = 0
                img_N.itemset((i, j), b)
        img_H1 = img_N.astype("uint8")
        cv2.imshow("Hysterisis part 1", img_H1)
        print('---Nilai Pixel Hysterisis Part 1--- \n', img_H1)

        # Hysteresis Thresholding eliminasi titk tepi lemah jika tidak terhubung dengan tetangga tepi kuat
        strong = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if (img_H1[i, j] == weak):
                    try:
                        if ((img_H1[i + 1, j - 1] == strong) or
                                (img_H1[i + 1, j] == strong) or
                                (img_H1[i + 1, j + 1] == strong) or
                                (img_H1[i, j - 1] == strong) or
                                (img_H1[i, j + 1] == strong) or
                                (img_H1[i - 1, j - 1] == strong) or
                                (img_H1[i - 1, j] == strong) or
                                (img_H1[i - 1, j + 1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass
        img_H2 = img_H1.astype("uint8")
        cv2.imshow("Hysteresis part 2", img_H2)
        print('---Nilai Pixel Hysterisis Part 1--- \n', img_H2)


    # G1
    def MortlgiDilasi(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        ret, imgthres = cv2.threshold(img, 127, 255, 0)
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilasi = cv2.dilate(imgthres, strel, iterations=1)
        cv2.imshow("Hasil Dilasi", dilasi)
        cv2.imshow('Original', img)
        print('---Nilai Pixel Dilasi--- \n', dilasi)

    def MorflgiErosi(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        ret, imgthres = cv2.threshold(img, 127, 255, 0)
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        erosi = cv2.erode(imgthres, strel, iterations=1)
        cv2.imshow("Hasil Erosi", erosi)
        cv2.imshow('Original', img)
        print('---Nilai Pixel Erosi--- \n', erosi)

    def MorflgiOpening(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        ret, imgthres = cv2.threshold(img, 127, 255, 0)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        opening = cv2.morphologyEx(imgthres, cv2.MORPH_OPEN, strel)
        cv2.imshow("Hasil Opening", opening)
        cv2.imshow('Original', img)
        print('---Nilai Pixel Opening--- \n', opening)

    def MorflgiClosing(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        ret, imgthres = cv2.threshold(img, 127, 255, 0)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        closing = cv2.morphologyEx(imgthres, cv2.MORPH_CLOSE, strel)
        cv2.imshow("Hasil Closing", closing)
        cv2.imshow('Original', img)
        print('---Nilai Pixel Closing--- \n', closing)

    def MorfSkeleton(self):
        img = cv2.imread("10.jpeg")
        imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, imgg = cv2.threshold(imgg, 127, 255, 0)

        skel = np.zeros(imgg.shape, np.uint8)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            open = cv2.morphologyEx(imgg, cv2.MORPH_OPEN, strel)
            # mengurangi gambar dari yang asli
            temp = cv2.subtract(imgg, open)
            # erosi gambar aslidan perbaikan kerangka
            eroded = cv2.erode(imgg, strel)
            skel = cv2.bitwise_or(skel, temp)
            imgg = eroded.copy()
            # jika tidak ada pixel putih yang tersisa
            if cv2.countNonZero(imgg) == 0:
                break
            cv2.imshow("Skeleton", skel)
            print("---------Nilai pixel------\n", skel)
            cv2.imshow("Origin", img)
            print("---------Nilai pixel------\n", img)

    # H1
    def Binary(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        T = 127  # nilai ambang
        MAX = 255
        ret, thres = cv2.threshold(img, T, MAX, cv2.THRESH_BINARY)
        cv2.imshow('Binary', thres)
        cv2.imshow('Original', img)
        print("---------Nilai pixel Binary------\n", thres)

    def BinaryInvers(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        T = 127
        MAX = 255
        ret, thres = cv2.threshold(img, T, MAX, cv2.THRESH_BINARY_INV)
        cv2.imshow('Binaryinvers', thres)
        cv2.imshow('Original', img)
        print("---------Nilai pixel Binary Invers------\n", thres)

    def Trunc(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        T = 127
        MAX = 255
        ret, thres = cv2.threshold(img, T, MAX, cv2.THRESH_TRUNC)
        cv2.imshow('Trunc', thres)
        cv2.imshow('Original', img)
        print("---------Nilai pixel Trunc------\n", thres)

    def ToZero(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        T = 127
        MAX = 255
        ret, thres = cv2.threshold(img, T, MAX, cv2.THRESH_TOZERO)
        cv2.imshow('ToZero', thres)
        cv2.imshow('Original', img)
        print("---------Nilai pixel ToZero------\n", thres)

    def ToZeroInvers(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        T = 127
        MAX = 255
        ret, thres = cv2.threshold(img, T, MAX, cv2.THRESH_TOZERO_INV)
        cv2.imshow('ToZeroInvers', thres)
        cv2.imshow('Original', img)
        print("---------Nilai pixel ToZero Invers------\n", thres)

    # H2
    def Meanthres(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        thres = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
        cv2.imshow('MeanThresholding', thres)
        cv2.imshow('Original', img)
        print("---------Nilai pixel MeanThres------\n", thres)

    def Gausthres(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        thres = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 6)
        cv2.imshow('GaussianTresholding', thres)
        cv2.imshow('Original', img)
        print("---------Nilai pixel GaussThress------\n", thres)

    def Otsuthres(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        T = 130
        ret, thres = cv2.threshold(img, T, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('OtsuTresholding', thres)
        cv2.imshow('Original', img)
        print('---Nilai Pixel Closing--- \n', thres)

    # H3
    def Contour(self):
        img = cv2.imread('9.jpeg')
        # img = cv2.resize(img1, (900, 600), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T = 127
        max = 255
        ret, hasil = cv2.threshold(gray, T, max, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=hasil, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        image_copy = img.copy()
        i = 0
        for contour in contours:
            if i == 0:
                i = 1
                continue
            approx = cv2.approxPolyDP(
                contour, 0.01 * cv2.arcLength(contour, True), True)
            cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
            if len(approx) == 3:
                cv2.putText(image_copy, 'Triangle', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (17, 32, 242), 2)
            elif len(approx) == 4:
                cv2.putText(image_copy, 'Rectangle', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (17, 32, 242), 2)
            elif len(approx) == 10:
                cv2.putText(image_copy, 'Star', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (17, 32, 242), 2)
            elif len(approx) == 4:
                cv2.putText(image_copy, 'rectangle', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (17, 32, 242), 2)
            else:
                cv2.putText(image_copy, 'circle', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (17, 32, 242), 2)
        cv2.imshow('Original', img)
        cv2.imshow('Contour', image_copy)

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