import os
import pickle
import sys

import matplotlib
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class SelectableLabel(QtWidgets.QLabel):
    MIN_SELECT_WIDTH = 10

    def __init__(self, img: QtGui.QImage, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setPixmap(QtGui.QPixmap.fromImage(img))
        self.setScaledContents(True)
        self._pressed = False

        self._selectStart = None
        self._selectMid = None
        self._selectEnd = None

        self._painter = QtGui.QPainter()

    def mousePressEvent(self, evt: QtGui.QMouseEvent) -> None:
        self._selectStart = evt.position().x()
        self._selectEnd = None
        self._pressed = True

    def mouseMoveEvent(self, evt: QtGui.QMouseEvent) -> None:
        if self._pressed:
            self._selectMid = evt.position().x()
            if abs(self._selectStart - self._selectMid) < self.MIN_SELECT_WIDTH:
                self._selectMid = None
            self.repaint()

    def mouseReleaseEvent(self, evt: QtGui.QMouseEvent) -> None:
        if self._pressed:
            end = evt.position().x()
            if abs(self._selectStart - end) < self.MIN_SELECT_WIDTH:
                self._selectStart = None
                self._selectEnd = None
            else:
                self._selectEnd = end

            self._pressed = False
            self.repaint()
            selection = self.get_selection()
            if selection is not None:
                print(selection)

    def paintEvent(self, evt: QtGui.QPaintEvent) -> None:
        super().paintEvent(evt)

        if self._selectStart is None:
            return

        ending = self._selectEnd if (self._selectEnd is not None) else self._selectMid

        if ending is None:
            return

        starting, ending = sorted([self._selectStart, ending])

        self._painter.begin(self)
        highlight_color = (
            self.palette()
            .brush(QtGui.QPalette.Active, QtGui.QPalette.Highlight)
            .color()
        )
        highlight_color = QtGui.QColor(
            highlight_color.red(), highlight_color.green(), highlight_color.blue(), 155
        )

        self._painter.fillRect(
            starting, 0, ending - starting, self.height(), highlight_color
        )

        self._painter.end()

    def get_selection(self):
        if self._selectStart is None or self._selectEnd is None:
            return None
        else:
            return tuple(sorted([self._selectStart, self._selectEnd]))


class TestWidget(QtWidgets.QWidget):
    def __init__(self, image: QtGui.QImage, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self._sizer = QtWidgets.QVBoxLayout()

        self._scrollImg = QtWidgets.QScrollArea()
        self._scrollImg.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._scrollImg.setWidgetResizable(True)

        self._img = SelectableLabel(image)
        self._scrollImg.setWidget(self._img)

        self.button = QtWidgets.QPushButton("Click me!")

        self._sizer.addWidget(self._scrollImg)

        self.setLayout(self._sizer)

    def get_selection(self):
        return self._img.get_selection()

    def mouseReleaseEvent(self, evt: QtGui.QMouseEvent) -> None:
        if self._pressed:
            end = evt.position().x()
            if abs(self._selectStart - end) < self.MIN_SELECT_WIDTH:
                self._selectStart = None
                self._selectEnd = None
            else:
                self._selectEnd = end

            self._pressed = False
            self.repaint()
            selection = self.get_selection()
            if selection is not None:
                print("hello")
                print(selection)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    with open(f"{os.environ['HOME']}/example.pkl", "rb") as src:
        im = pickle.load(src)

    im = im - np.min(im)

    im = im / np.max(im)
    print(np.min(im), np.max(im))

    im = matplotlib.colormaps["viridis"](im, alpha=1, bytes=True)

    qt_img = QtGui.QImage(im, im.shape[1], im.shape[0], QtGui.QImage.Format_RGB32)

    test = TestWidget(qt_img)
    test.show()

    app.exec()
