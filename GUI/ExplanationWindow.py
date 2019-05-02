import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QWidget, QFileDialog, QScrollArea
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class UI_Explanation(QWidget):

    weights = None

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Explanation')
        self.setGeometry(400, 200, 500, 400) 
        self.setStyleSheet(open("style.qss", "r").read())
        self.setWindowIcon(QtGui.QIcon("icons/TSExplanation.ico"))

        #self.lbl_class = QtWidgets.QLabel("Résultat : classe1")
        self.lbl_class = QtWidgets.QLabel("")
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        #self.lbl_weights = QtWidgets.QLabel("Poids : (ss8, 0.5), (ss1, 0.5), (ss3, 0.3), (ss6, 0.2), (ss2, 0.05)... ")
        self.lbl_weights = QtWidgets.QLabel("")
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.lbl_weights)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFixedHeight(50)
        self.btn_save = QPushButton('Sauvegarder')
        self.btn_save.clicked.connect(self.saveExplanation)


    # Action of the button 'Sauvegarder'
    def saveExplanation(self):
        fileName, _  = QFileDialog.getSaveFileName(self, "Sauvegarder un fichier","../","HTML (*.html)") # Créer dossier savedExplanations ??
        #if fileName:
        #    UI_Explanation.explanation.save_to_file(fileName)


    # Show the plot
    def plot(self, exp):
        layout = QVBoxLayout()
        layout.addWidget(self.lbl_class)
        layout.addWidget(self.canvas)
        layout.addWidget(self.scroll)
        layout.addWidget(self.btn_save)
        self.setLayout(layout)
        weights = UI_Explanation.weights = exp.as_list()
        sorted_weights = sorted(weights, key=lambda tup: tup[1], reverse=True) # Sort from the biggest weight to the lowest
        self.lbl_weights.setText("Poids : " + (str(sorted_weights))[1:(len(str(sorted_weights))-1)]) # Print the weights
        self.canvas.draw_idle()


    # Show the window
    def showUI(self, exp, result_class, myTs, num_cuts):
        #UI_Explanation.explanation = explanation
        #result_class = "classe1"
        self.canvas = exp.domain_mapper.as_pyplot(exp, myTs, num_cuts)
        self.lbl_class.setText("Résultat : " + result_class)
        self.plot(exp)
        self.show()

"""
# Test
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = UI_Explanation()
    main.showUI()
    sys.exit(app.exec_())
"""