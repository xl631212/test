import sys
from PyQt5 import QtWidgets
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QWidget()
    w.resize(400, 200)
    w.setWindowTitle("hello PyQt5")
    w.show()
    exit(app.exec_())
