import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QWidget, QFileDialog, QScrollArea
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class UI_Explanation(QWidget):

    result_class = ""
    explanation = None
    ts = None
    num_cuts = None

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Explanation')
        self.setGeometry(375, 200, 550, 450) 
        self.setStyleSheet(open("style.qss", "r").read())
        self.setWindowIcon(QtGui.QIcon("icons/TSExplanation.ico"))
        self.lbl_class = QtWidgets.QLabel("")
        self.lbl_class.setAlignment(Qt.AlignCenter)
        self.lbl_score = QtWidgets.QLabel("")
        self.lbl_score.setAlignment(Qt.AlignCenter)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.lbl_legende = QtWidgets.QLabel("Poids :   ")
        self.img_legende = QtWidgets.QLabel()
        self.img_legende.setPixmap(QPixmap("icons/Legende_GUI.png"))
        self.horizontalLayout.addWidget(self.lbl_legende)
        self.horizontalLayout.addWidget(self.img_legende)
        self.horizontalLayout.setAlignment(Qt.AlignCenter)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
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
        fileName, _  = QFileDialog.getSaveFileName(self, "Sauvegarder un fichier","../limeTS/SavedExplanations", "All Files (*)") # Créer dossier savedExplanations ??
        if fileName:
            UI_Explanation.explanation.domain_mapper.save_to_file(fileName, UI_Explanation.explanation, UI_Explanation.ts, UI_Explanation.num_cuts, UI_Explanation.result_class)


    # Show the plot
    def plot(self):
        layout = QVBoxLayout()
        layout.addWidget(self.lbl_class)
        layout.addWidget(self.lbl_score)
        layout.addLayout(self.horizontalLayout)
        layout.addWidget(self.canvas)
        layout.addWidget(self.scroll)
        layout.addWidget(self.btn_save)
        self.setLayout(layout)
        weights = UI_Explanation.explanation.as_list()
        sorted_weights = sorted(weights, key=lambda tup: tup[1], reverse=True) # Sort from the biggest weight to the lowest
        self.lbl_weights.setText("Poids : " + (str(sorted_weights))[1:(len(str(sorted_weights))-1)]) # Print the weights
        self.canvas.draw_idle()


    # Show the window
    def showUI(self, exp, result_class, myTs, num_cuts, score):
        UI_Explanation.explanation = exp
        UI_Explanation.result_class = result_class
        UI_Explanation.ts = myTs
        UI_Explanation.num_cuts = num_cuts
        self.canvas, _ = exp.domain_mapper.as_pyplot(exp, myTs, num_cuts)
        self.lbl_class.setText("Résultat : " + result_class)
        self.lbl_score.setText("Taux de classification correcte : " + str(round(score*100,2))+"%"+"\n")
        self.plot()
        self.show()

"""
# Test
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = UI_Explanation()
    main.showUI()
    sys.exit(app.exec_())
"""