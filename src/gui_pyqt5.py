import sys
import cv2
import imutils
from PyQt5 import QtCore, QtGui, QtWidgets


class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        ori_img = cv2.imread("../../resources/omr-imgs/omr-1-ans-ori.png", cv2.IMREAD_COLOR)
        self.original_image_color = imutils.resize(ori_img, height=900)
        self.original_image_gray = cv2.cvtColor(self.original_image_color, cv2.COLOR_BGR2GRAY)

        self.thresh = False
        self.addthresh = False
        self.thresh_karnel_size = 11
        self.addthresh_c_size = 8

        self.init_ui()

    def init_ui(self):
        self.imglabel = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.imglabel.setFixedSize(1200, 900)
        hbox1 = QtWidgets.QHBoxLayout()
        cb_thresh = QtWidgets.QCheckBox('thresh', checked=False)
        cb_thresh.stateChanged.connect(self.changeTitleThresh)
        cb_addthresh = QtWidgets.QCheckBox('adapthresh', checked=False)
        cb_addthresh.stateChanged.connect(self.changeTitleAdapThresh)
        hbox1.addWidget(cb_thresh)
        hbox1.addWidget(cb_addthresh)

        self.thresh_k_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, focusPolicy=QtCore.Qt.StrongFocus,
                                                 tickPosition=QtWidgets.QSlider.TicksBothSides,
                                                 tickInterval=1,
                                                 singleStep=1,
                                                 pageStep=1,
                                                 minimum=1,
                                                 maximum=255)
        self.thresh_c_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, focusPolicy=QtCore.Qt.StrongFocus,
                                                 tickPosition=QtWidgets.QSlider.TicksBothSides,
                                                 tickInterval=1,
                                                 singleStep=1,
                                                 pageStep=1,
                                                 minimum=1,
                                                 maximum=100)
        self.thresh_k_slider.valueChanged[int].connect(self.threshKSliderChangeValue)
        self.thresh_c_slider.valueChanged[int].connect(self.threshCSliderChangeValue)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addLayout(hbox1)
        vbox.addWidget(self.thresh_k_slider)
        vbox.addWidget(self.thresh_c_slider)

        vboxLabels = QtWidgets.QVBoxLayout(self)

        self.k_thresh_label = QtWidgets.QLabel("K val: ", alignment=QtCore.Qt.AlignCenter)
        self.c_thresh_label = QtWidgets.QLabel("C val: ", alignment=QtCore.Qt.AlignCenter)

        vboxLabels.addWidget(self.k_thresh_label)
        vboxLabels.addWidget(self.c_thresh_label)

        hbox_image_container = QtWidgets.QHBoxLayout()
        hbox_image_container.addWidget(self.imglabel)
        hbox_image_container.addLayout(vboxLabels)

        vbox.addLayout(hbox_image_container)

        # self.threshKSliderChangeValue(self.thresh_k_slider.value())
        # self.threshCSliderChangeValue(self.thresh_c_slider.value())
        self.updateImage(self.original_image_color)
        self.setGeometry(50, 50, 1200, 900)
        self.setWindowTitle('Learning PyQT5')
        self.show()

    @QtCore.pyqtSlot(int)
    def changeTitleThresh(self, state):
        self.thresh = state == QtCore.Qt.Checked
        self.threshKSliderChangeValue(self.thresh_k_slider.value())

    @QtCore.pyqtSlot(int)
    def changeTitleAdapThresh(self, state):
        self.addthresh = state == QtCore.Qt.Checked
        self.threshCSliderChangeValue(self.thresh_c_slider.value())

    @QtCore.pyqtSlot(int)
    def threshKSliderChangeValue(self, value):
        ksize = (value * 2) + 1
        if 1 < ksize <= 255 and ksize % 2 != 0 and self.addthresh:
            self.thresh_karnel_size = ksize
            self.k_thresh_label.setText("K val: " + str(self.thresh_karnel_size))
            gray_img = cv2.adaptiveThreshold(self.original_image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                             self.thresh_karnel_size, self.addthresh_c_size)
            gray_img_c = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            self.updateImage(gray_img_c)
        elif self.thresh and self.addthresh is not True:
            self.thresh_karnel_size = value
            self.k_thresh_label.setText("K val: " + str(self.thresh_karnel_size))
            _, gray_img = cv2.threshold(self.original_image_gray, self.thresh_karnel_size, 255, cv2.THRESH_BINARY)
            gray_img_c = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            self.updateImage(gray_img_c)

    @QtCore.pyqtSlot(int)
    def threshCSliderChangeValue(self, value):
        if self.addthresh:
            self.addthresh_c_size = value
            self.c_thresh_label.setText("C val: " + str(self.addthresh_c_size))
            gray_img = cv2.adaptiveThreshold(self.original_image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                             self.thresh_karnel_size, self.addthresh_c_size)
            gray_img_c = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            self.updateImage(gray_img_c)

    def updateImage(self, image):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        # pixMap = QtGui.QPixmap.fromImage(qImg).scaled(700, 500, QtCore.Qt.KeepAspectRatio)
        self.imglabel.setPixmap(QtGui.QPixmap.fromImage(qImg))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = MyWindow()
    sys.exit(app.exec_())
