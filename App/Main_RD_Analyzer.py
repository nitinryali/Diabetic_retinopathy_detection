import Background
# import Kemendikbiud_image
# import unib
from PyQt5 import QtCore, QtGui, QtWidgets
import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm,trange
from time import sleep
from Preprocess_Gaussian_Blur import load_ben_color
from Size_Normalize import del_black_or_white
from ContratsNormalize_CLAHE import CLAHE
import cv2
import numpy as np
import os
import tempfile


input_shape = (224,224,3)
model_input =tf.keras.Input(shape=input_shape)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 750)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        # self.label.setGeometry(QtCore.QRect(250, 10, 771, 21))
        self.label.setGeometry(QtCore.QRect(350, 20, 971, 31))

        font = QtGui.QFont()
        font.setFamily("Rockwell Extra Bold")
        font.setPointSize(26)
        #font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        # self.label_2 = QtWidgets.QLabel(self.centralwidget)
        # self.label_2.setGeometry(QtCore.QRect(0, 0, 211, 151))
        # self.label_2.setStyleSheet("image: url(:/newPrefix/mendikbud.png);")
        # self.label_2.setText("")
        # self.label_2.setObjectName("label_2")
        # self.label_3 = QtWidgets.QLabel(self.centralwidget)
        # self.label_3.setGeometry(QtCore.QRect(1170, 0, 161, 141))
        # self.label_3.setStyleSheet("image: url(:/kampus/UNIB.png);")
        # self.label_3.setText("")
        # self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        # self.label_4.setGeometry(QtCore.QRect(370, 40, 611, 31))
        self.label_4.setGeometry(QtCore.QRect(470, 60, 611, 31))

        font = QtGui.QFont()
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(320, 30, 871, 751))
        self.label_5.setStyleSheet("image: url(:/Background/Background.png);")
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(480, 70, 381, 41))
        font = QtGui.QFont()
        font.setFamily("Swis721 Hv BT")
        font.setPointSize(14)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.txt_V3_0 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_V3_0.setGeometry(QtCore.QRect(830, 360, 131, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_V3_0.setFont(font)
        self.txt_V3_0.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_V3_0.setObjectName("txt_V3_0")
        self.txt_V3_1 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_V3_1.setGeometry(QtCore.QRect(830, 390, 131, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_V3_1.setFont(font)
        self.txt_V3_1.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_V3_1.setObjectName("txt_V3_1")
        self.txt_V3_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_V3_3.setGeometry(QtCore.QRect(830, 420, 131, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_V3_3.setFont(font)
        self.txt_V3_3.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_V3_3.setObjectName("txt_V3_3")
        self.txt_V3_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_V3_4.setGeometry(QtCore.QRect(830, 450, 131, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_V3_4.setFont(font)
        self.txt_V3_4.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_V3_4.setObjectName("txt_V3_4")
        self.txt_V2_0 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_V2_0.setGeometry(QtCore.QRect(830, 540, 131, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_V2_0.setFont(font)
        self.txt_V2_0.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_V2_0.setObjectName("txt_V2_0")
        self.txt_V2_1 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_V2_1.setGeometry(QtCore.QRect(830, 570, 131, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_V2_1.setFont(font)
        self.txt_V2_1.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_V2_1.setObjectName("txt_V2_1")
        self.txt_V2_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_V2_2.setGeometry(QtCore.QRect(830, 600, 131, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_V2_2.setFont(font)
        self.txt_V2_2.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_V2_2.setObjectName("txt_V2_2")
        self.txt_V2_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_V2_3.setGeometry(QtCore.QRect(830, 630, 131, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_V2_3.setFont(font)
        self.txt_V2_3.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_V2_3.setObjectName("txt_V2_3")
        self.txt_201_1 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_201_1.setGeometry(QtCore.QRect(830, 210, 141, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_201_1.setFont(font)
        self.txt_201_1.setText("")
        self.txt_201_1.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_201_1.setObjectName("txt_201_1")
        self.txt_201_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_201_2.setGeometry(QtCore.QRect(830, 240, 141, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_201_2.setFont(font)
        self.txt_201_2.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_201_2.setObjectName("txt_201_2")
        self.txt_201_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_201_3.setGeometry(QtCore.QRect(830, 270, 141, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_201_3.setFont(font)
        self.txt_201_3.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_201_3.setObjectName("txt_201_3")
        self.txt_ESL_0 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_ESL_0.setGeometry(QtCore.QRect(1200, 350, 131, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_ESL_0.setFont(font)
        self.txt_ESL_0.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_ESL_0.setObjectName("txt_ESL_0")
        self.txt_ESL_1 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_ESL_1.setGeometry(QtCore.QRect(1200, 380, 131, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_ESL_1.setFont(font)
        self.txt_ESL_1.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_ESL_1.setObjectName("txt_ESL_1")
        self.txt_ESL_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_ESL_2.setGeometry(QtCore.QRect(1200, 410, 131, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_ESL_2.setFont(font)
        self.txt_ESL_2.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_ESL_2.setObjectName("txt_ESL_2")
        self.txt_ESL_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_ESL_4.setGeometry(QtCore.QRect(1180, 510, 161, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.txt_ESL_4.setFont(font)
        self.txt_ESL_4.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_ESL_4.setObjectName("txt_ESL_4")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(1240, 490, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.predict_img_lbl = QtWidgets.QLabel(self.centralwidget)
        self.predict_img_lbl.setGeometry(QtCore.QRect(240, 290, 201, 201))
        self.predict_img_lbl.setAutoFillBackground(True)
        self.predict_img_lbl.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.predict_img_lbl.setText("")
        self.predict_img_lbl.setObjectName("predict_img_lbl")
        self.Load_img_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Load_img_btn.setGeometry(QtCore.QRect(0, 610, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.Load_img_btn.setFont(font)
        self.Load_img_btn.setObjectName("Load_img_btn")
        self.Predict_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Predict_btn.setGeometry(QtCore.QRect(290, 610, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.Predict_btn.setFont(font)
        self.Predict_btn.setObjectName("Predict_btn")
        self.txt_ESL_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_ESL_3.setGeometry(QtCore.QRect(1200, 440, 131, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_ESL_3.setFont(font)
        self.txt_ESL_3.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_ESL_3.setObjectName("txt_ESL_3")
        self.txt_201_0 = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_201_0.setGeometry(QtCore.QRect(830, 180, 141, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.txt_201_0.setFont(font)
        self.txt_201_0.setText("")
        self.txt_201_0.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_201_0.setObjectName("txt_201_0")
        self.Clr_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Clr_btn.setGeometry(QtCore.QRect(200, 630, 75, 23))
        self.Clr_btn.setObjectName("Clr_btn")
        self.Load_image_lbl_2 = QtWidgets.QLabel(self.centralwidget)
        self.Load_image_lbl_2.setGeometry(QtCore.QRect(10, 290, 211, 201))
        self.Load_image_lbl_2.setAutoFillBackground(True)
        self.Load_image_lbl_2.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.Load_image_lbl_2.setText("")
        self.Load_image_lbl_2.setObjectName("Load_image_lbl_2")
        self.Preprocess_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Preprocess_btn.setGeometry(QtCore.QRect(140, 540, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.Preprocess_btn.setFont(font)
        self.Preprocess_btn.setObjectName("Preprocess_btn")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(30, 250, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(280, 250, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_5.raise_()
        self.label.raise_()
        # self.label_2.raise_()
        # self.label_3.raise_()
        self.label_4.raise_()
        self.label_6.raise_()
        self.txt_V3_0.raise_()
        self.txt_V3_1.raise_()
        self.txt_V3_3.raise_()
        self.txt_V3_4.raise_()
        self.txt_V2_0.raise_()
        self.txt_V2_1.raise_()
        self.txt_V2_2.raise_()
        self.txt_V2_3.raise_()
        self.txt_201_1.raise_()
        self.txt_201_2.raise_()
        self.txt_201_3.raise_()
        self.txt_ESL_0.raise_()
        self.txt_ESL_1.raise_()
        self.txt_ESL_2.raise_()
        self.txt_ESL_4.raise_()
        self.label_7.raise_()
        self.predict_img_lbl.raise_()
        self.Load_img_btn.raise_()
        self.Predict_btn.raise_()
        self.txt_ESL_3.raise_()
        self.txt_201_0.raise_()
        self.Clr_btn.raise_()
        self.Load_image_lbl_2.raise_()
        self.Preprocess_btn.raise_()
        self.label_8.raise_()
        self.label_9.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1345, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.Load_img_btn.clicked.connect(self.setImage)
        self.Preprocess_btn.clicked.connect(self.PraProcess)
        self.Predict_btn.clicked.connect(self.Deteksi)
        self.Clr_btn.clicked.connect(self.Clear)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DR-DETECTION"))
        self.label.setText(_translate("MainWindow", "DIABETIC RETINOPATHY CLASSIFIER"))
        self.label_4.setText(_translate("MainWindow", "      Ensemble Architecture"))
       # self.label_6.setText(_translate("MainWindow", "(FAKULTAS TEKNIK-UNIB)"))
        self.label_7.setText(_translate("MainWindow", "Output"))
        self.Load_img_btn.setText(_translate("MainWindow", "LOAD"))
        self.Predict_btn.setText(_translate("MainWindow", "PREDICT"))
        self.Clr_btn.setText(_translate("MainWindow", "Clear"))
        self.Preprocess_btn.setText(_translate("MainWindow", "Preprocess"))
        self.label_8.setText(_translate("MainWindow", "    ORIGINAL"))
        self.label_9.setText(_translate("MainWindow", "   INPUT"))

    def Clear (self):
        self.txt_ESL_0.clear()
        self.txt_ESL_1.clear()
        self.txt_ESL_2.clear()
        self.txt_ESL_3.clear()
        self.txt_ESL_4.clear()
        self.txt_201_0.clear()
        self.txt_201_1.clear()
        self.txt_201_2.clear()
        self.txt_201_3.clear()
        self.txt_V3_0.clear()
        self.txt_V3_1.clear()
        self.txt_V3_3.clear()
        self.txt_V3_4.clear()
        self.txt_V2_0.clear()
        self.txt_V2_1.clear()
        self.txt_V2_2.clear()
        self.txt_V2_3.clear()
        self.predict_img_lbl.clear()
        self.Load_image_lbl_2.clear()


    def setImage(self):
        global fileName
        fileName,_=QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files(*.png *.jpg *.jpeg *.bmp *.tif)")
        if fileName:
            pixmap=QtGui.QPixmap(fileName)
            pixmap=pixmap.scaled(self.Load_image_lbl_2.width(),self.Load_image_lbl_2.height(),QtCore.Qt.KeepAspectRatio)
            self.Load_image_lbl_2.setPixmap(pixmap)
            self.Load_image_lbl_2.setAlignment(QtCore.Qt.AlignCenter)
            print(fileName)

    # def PraProcess(self):
    #     global PreprocessFile
    #     img= cv2.imread(fileName)
    #     crop_size=1000
    #     image1=del_black_or_white(img)
    #     min_width_heigt=min(image1.shape[0],image1.shape[1])
    #     image_size_before_hough=crop_size*2
    #     if min_width_heigt<100:
    #         crop_ratio=image_size_before_hough/min_width_heigt
    #         image1=cv2.resize(image1,None, fx=crop_ratio, fy=crop_ratio)
            
    #     dim=(224,224)
    #     fundus1 = cv2.resize(image1, dim, interpolation = cv2.INTER_AREA)
    #     imag=cv2.imwrite('Lokasi/Size_normalize.jpg',fundus1)
    #     gambar='Lokasi/Size_normalize.jpg'
    #     Clahe=CLAHE(gambar)
    #     imag=cv2.imwrite('Lokasi/CLAHE.jpg',Clahe)
    #     img='Lokasi/CLAHE.jpg'
    #     img2= load_ben_color(img,sigmaX=10)
    #     cv2.imwrite('Lokasi/Gaussian_BLUR.jpg',img2)

    #     PreprocessFile='Lokasi/Gaussian_BLUR.jpg'
        
    #     pixmap=QtGui.QPixmap(PreprocessFile)
    #     pixmap=pixmap.scaled(self.predict_img_lbl.width(),self.predict_img_lbl.height(),QtCore.Qt.KeepAspectRatio)
    #     self.predict_img_lbl.setPixmap(pixmap)
    #     self.predict_img_lbl.setAlignment(QtCore.Qt.AlignCenter)

    def PraProcess(self):
        global PreprocessFile
        img = cv2.imread(fileName)
        crop_size = 1000
        image1 = del_black_or_white(img)
        min_width_height = min(image1.shape[0], image1.shape[1])
        image_size_before_hough = crop_size * 2
        if min_width_height < 100:
            crop_ratio = image_size_before_hough / min_width_height
            image1 = cv2.resize(image1, None, fx=crop_ratio, fy=crop_ratio)

        dim = (224, 224)
        fundus1 = cv2.resize(image1, dim, interpolation=cv2.INTER_AREA)
    
        # Temporary file paths
        temp_file1 = tempfile.mktemp('.jpg')
        temp_file2 = tempfile.mktemp('.jpg')

        imag = cv2.imwrite(temp_file1, fundus1)
        Clahe = CLAHE(temp_file1)
        imag = cv2.imwrite(temp_file2, Clahe)
        img2 = load_ben_color(temp_file2, sigmaX=10)

        # Temporary output file path
        temp_output_file = tempfile.mktemp('.jpg')
        cv2.imwrite(temp_output_file, img2)

        PreprocessFile = temp_output_file

        pixmap = QtGui.QPixmap(PreprocessFile)
        pixmap = pixmap.scaled(self.predict_img_lbl.width(), self.predict_img_lbl.height(), QtCore.Qt.KeepAspectRatio)
        self.predict_img_lbl.setPixmap(pixmap)
        self.predict_img_lbl.setAlignment(QtCore.Qt.AlignCenter)

    
        
    def Deteksi(self):
        IMG_SIZE=(224,224)
        img = tf.keras.preprocessing.image.load_img(PreprocessFile, target_size=IMG_SIZE)
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        Base_model1 =tf.keras.applications.DenseNet201(input_shape=input_shape, input_tensor=model_input, include_top=False, weights=None)
        for layer in Base_model1.layers:
            layer.trainable = True
        Base_model1_last_layer = Base_model1.get_layer('relu')
        Base_model1_last_output = Base_model1_last_layer.output
        x1 =tf.keras.layers.GlobalAveragePooling2D()(Base_model1_last_output)
        x1 =tf.keras.layers.Dropout(0.25)(x1)
        x1 =tf.keras.layers.Dense(512, activation='relu')(x1)
        x1 =tf.keras.layers.Dropout(0.25)(x1)
        final_output1 =tf.keras.layers.Dense(4, activation='softmax', name='final_output')(x1)
        DensNet201_model =tf.keras.models.Model(model_input, final_output1)
        metric_list = ["accuracy"]
        optimizer =tf.keras.optimizers.Adam()
        DensNet201_model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)
        if getattr(sys, 'frozen', False): 
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")
        weights_path = os.path.join(base_path, "WEIGHT", "Weight_DensNet201_Optimal_Ori.h5")

        DensNet201_model.load_weights(weights_path)
        print(DensNet201_model.summary)
        print ("DensNet201 Start predict.........")
        
        for i in trange(10):
            sleep(0.1)
            Predict1=DensNet201_model.predict(x)
            pass
        
        print ("Normal Probabality:",Predict1[0][0])
        print ("Mild Probabality:",Predict1[0][1])
        print ("Moderate Probabality:",Predict1[0][2])
        print ("Severe Probabality:",Predict1[0][3])

        self.txt_201_0.setText(str(Predict1[0][0]))
        self.txt_201_1.setText(str(Predict1[0][1]))
        self.txt_201_2.setText(str(Predict1[0][2]))
        self.txt_201_3.setText(str(Predict1[0][3]))


        Base_model2 =tf.keras.applications.InceptionV3(input_shape=input_shape, input_tensor=model_input, include_top=False, weights=None)
        for layer in Base_model2.layers:
            layer.trainable = True
        Base_model2_last_layer = Base_model2.get_layer('mixed10')
        Base_model2_last_output = Base_model2_last_layer.output
        x2 =tf.keras.layers.GlobalAveragePooling2D()(Base_model2_last_output)
        x2 =tf.keras.layers.Dropout(0.25)(x2)
        x2 =tf.keras.layers.Dense(1024, activation='relu')(x2)
        x2 =tf.keras.layers.Dropout(0.25)(x2)
        final_output2 =tf.keras.layers.Dense(4, activation='softmax', name='final_output2')(x2)
        InceptionV3_model =tf.keras.models.Model(model_input, final_output2)
        metric_list = ["accuracy"]
        optimizer = tf.keras.optimizers.Adam(1.0000e-06)
        InceptionV3_model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)
        if getattr(sys, 'frozen', False):  # Running as a bundled executable
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")

        # Construct the full path to the weights file
        weights_path1 = os.path.join(base_path, "WEIGHT", "Weight_InceptionV3_Optimal_Ori.h5")

        # Load the model weights
        InceptionV3_model.load_weights(weights_path1)
        print ("InceptionV3 Start predict.........")
        
        for i in trange(10):
            sleep(0.1)
            Predict2=InceptionV3_model.predict(x)
            pass
        
        print ("Normal Probabality:",Predict2[0][0])
        print ("Mild Probabality:",Predict2[0][1])
        print ("Moderate Probabality:",Predict2[0][2])
        print ("Severe Probabality:",Predict2[0][3])

        self.txt_V3_0.setText(str(Predict2[0][0]))
        self.txt_V3_1.setText(str(Predict2[0][1]))
        self.txt_V3_3.setText(str(Predict2[0][2]))
        self.txt_V3_4.setText(str(Predict2[0][3]))


        Base_model3 =tf.keras.applications.MobileNetV2(input_shape=input_shape, input_tensor=model_input, include_top=False, weights=None)
        for layer in Base_model3.layers:
            layer.trainable = True
        Base_model3_last_layer = Base_model3.get_layer('out_relu')
        Base_model3_last_output = Base_model3_last_layer.output
        x3 =tf.keras.layers.GlobalAveragePooling2D()(Base_model3_last_output)
        x3 =tf.keras.layers.Dropout(0.5)(x3)
        x3 =tf.keras.layers.Dense(512, activation='relu')(x3)
        x3 =tf.keras.layers.Dropout(0.5)(x3)
        final_output3 =tf.keras.layers.Dense(4, activation='softmax', name='final_output3')(x3)
        MobileNetV2_model =tf.keras.models.Model(model_input, final_output3)
        metric_list = ["accuracy"]
        optimizer = tf.keras.optimizers.Adam()
        MobileNetV2_model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)
        if getattr(sys, 'frozen', False):  # Running as a bundled executable
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")

        # Construct the full path to the weights file
        weights_path2 = os.path.join(base_path, "WEIGHT", "Weight_MobileNetV2_Optimal_(Ori).h5")

        # Load the model weights
        MobileNetV2_model.load_weights(weights_path2)
        print ("MobileNetV2 Start predict.........")
        
        for i in trange(10):
            sleep(0.1)
            Predict3=MobileNetV2_model.predict(x)
            pass
        print ("Normal Probabality:",Predict3[0][0])
        print ("Mild Probabality:",Predict3[0][1])
        print ("Moderate Probabality:",Predict3[0][2])
        print ("Severe Probabality:",Predict3[0][3])

        self.txt_V2_0.setText(str(Predict3[0][0]))
        self.txt_V2_1.setText(str(Predict3[0][1]))
        self.txt_V2_2.setText(str(Predict3[0][2]))
        self.txt_V2_3.setText(str(Predict3[0][3]))


        def ensemble(models, model_input):
            outputs = [model.outputs[0] for model in models]
            y =tf.keras.layers.Average()(outputs)
            model =tf.keras.Model(model_input,y,name='ensemble')
            return model

        ensemble_model = ensemble([DensNet201_model,InceptionV3_model,MobileNetV2_model], model_input)
        metric_list = ["accuracy"]
        optimizer =tf.keras.optimizers.Adam()
        ensemble_model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)

        print ("Ensemble Start predict.........")
        
        for i in trange(10):
            sleep(0.1)
            Predict=ensemble_model.predict(x)
            pass

        
        print ("Normal Probabality:",Predict[0][0])
        print ("Mild Probabality:",Predict[0][1])
        print ("Moderate Probabality:",Predict[0][2])
        print ("Severe Probabality:",Predict[0][3])
        
        self.txt_ESL_0.setText(str(Predict[0][0]))
        self.txt_ESL_1.setText(str(Predict[0][1]))
        self.txt_ESL_2.setText(str(Predict[0][2]))
        self.txt_ESL_3.setText(str(Predict[0][3]))


        if Predict[0][0]>=0.5:
            self.txt_ESL_4.setText("NORMAL (0)")
            print ("Diagnosis: NORMAL")
        elif Predict[0][1]>=0.5:
            self.txt_ESL_4.setText("MILD (1)")
            print ("Diagnosis: MILD")
        elif Predict[0][2]>=0.5:
            self.txt_ESL_4.setText("MODERATE (2)")
            print ("Diagnosis: MODERATE")
        elif Predict[0][3]>=0.5:
            self.txt_ESL_4.setText("SEVERE (3)")
            print ("Diagnosis: SEVERE")

            
    


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
