import sys
import cv2
import numpy as np
import imutils
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLCDNumber, QSlider, QLabel, \
    QCheckBox, QMainWindow
from PyQt5.QtGui import QIcon, QPixmap, QImage

from qt_gui.ImgProcessing import ImgProcessing


class MyWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.imglabel = QLabel(self)
        self.imglabel.setFixedSize(1200, 900)
        ori_img = cv2.imread("../resources/omr-1-ans-ori.png", cv2.IMREAD_COLOR)
        ori_img = imutils.resize(ori_img, height=960)
        self.gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
        self.gray_img_c = ori_img

        self.thresh = False
        self.thresh_karnel_size = 11

        self.init_ui()

    def init_ui(self):
        # lcd = QLCDNumber(self)
        hbox1 = QHBoxLayout()
        cb_thresh = QCheckBox('thresh', self)
        cb_thresh.setChecked(False)

        cb_thresh.stateChanged.connect(self.changeTitleThresh)
        hbox1.addWidget(cb_thresh)

        thresh_slider = QSlider(Qt.Horizontal, self)
        thresh_slider.setFocusPolicy(Qt.StrongFocus)
        thresh_slider.setTickPosition(QSlider.TicksBothSides)
        thresh_slider.setTickInterval(1)
        thresh_slider.setSingleStep(1)
        thresh_slider.setPageStep(1)
        thresh_slider.setMinimum(1)
        thresh_slider.setMaximum(127)
        thresh_slider.valueChanged[int].connect(self.threshSliderChangeValue)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addWidget(thresh_slider)
        vbox.addWidget(self.imglabel)
        self.setLayout(vbox)

        self.setGeometry(50, 50, 1200, 768)
        self.setWindowTitle('Learning PyQT5')
        self.updateImage()
        self.show()

    def changeTitleThresh(self, state):
        # print("thresh checkbox: ", state, Qt.Checked)
        if state == Qt.Checked:
            self.thresh = True
        else:
            self.thresh = False

    def threshSliderChangeValue(self, value):

        ksize = (value * 2) + 1
        print("ksize: ", ksize)
        if ksize > 1 and ksize % 2 != 0 and self.thresh:
            self.thresh_karnel_size = ksize
            self.gray_img = cv2.threshold(self.gray_img, self.thresh_karnel_size, 255, cv2.THRESH_BINARY)[1]
            self.gray_img_c = cv2.cvtColor(self.gray_img.copy(), cv2.COLOR_GRAY2BGR)
            self.updateImage()

    def updateImage(self):

        height, width, channel = self.gray_img_c.shape
        bytesPerLine = 3 * width
        qImg = QImage(self.gray_img_c.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixMap = QPixmap.fromImage(qImg)
        pixMap = pixMap.scaled(700, 500, Qt.KeepAspectRatio)
        self.imglabel.setPixmap(pixMap)
        self.imglabel.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWindow()
    sys.exit(app.exec_())
