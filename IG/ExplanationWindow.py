import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import random

class UI_Explanation(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Explanation')
        self.setGeometry(400, 200, 400, 331) 
        #self.setStyleSheet("""background: white;""")

        # a figure instance to plot on

        self.lbl1 = QtWidgets.QLabel("Résultat : classe1")
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.button = QPushButton('Sauvegarder')
        self.lbl2 = QtWidgets.QLabel("Poids : (ss8, 0.5), (ss1, 0.5), (ss3, 0.3), (ss6, 0.2), (ss2, 0.05)... ")

        data = [random.random() for i in range(11)]
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(data, '*-')
        self.canvas.draw()

        self.button.clicked.connect(self.plot)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.lbl1)
        layout.addWidget(self.canvas)
        layout.addWidget(self.lbl2)
        layout.addWidget(self.button)
        self.setLayout(layout)

    # Action of the button 'Sauvegarder'
    #def saveExplanation(self):

    def plot(self):
        ''' plot some random stuff '''
        # random data
        data = [random.random() for i in range(10)]
        # instead of ax.hold(False)
        self.figure.clear()
        # create an axis
        ax = self.figure.add_subplot(111)
        # plot data
        ax.plot(data, '*-')
        # refresh canvas
        self.canvas.draw()

    def showUI(self, explanation):
        self.show()

'''if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = UI_Explanation()
    main.show()

    sys.exit(app.exec_())'''