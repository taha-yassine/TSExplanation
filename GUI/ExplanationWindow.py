import sys
sys.path.insert(0, "../limeTS")
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QWidget, QFileDialog, QScrollArea
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
#import explanation

class UI_Explanation(QWidget):

    explanation = None

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Explanation')
        self.setGeometry(400, 200, 400, 331) 

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

        layout = QVBoxLayout()
        layout.addWidget(self.lbl_class)
        layout.addWidget(self.canvas)
        layout.addWidget(self.scroll)
        layout.addWidget(self.btn_save)
        self.setLayout(layout)


    # Action of the button 'Sauvegarder'
    def saveExplanation(self):
        fileName, _  = QFileDialog.getSaveFileName(self, "Sauvegarder un fichier","../","HTML (*.html)") # Créer dossier savedExplanations ??
        #if fileName:
        #    UI_Explanation.explanation.save_to_file(fileName)


    # Show the plot
    def plot(self):
        ts = [-0.34086,-0.38038,-0.3458,-0.36556,-0.3458,-0.36556,-0.3952,-0.38038,-0.38532,-0.3952,-0.38038,-0.35568,-0.34086,-0.32604,-0.2964,-0.2964,-0.33098,-0.30134,-0.30134,-0.3211,-0.28652]
        #weights = UI_Explanation.explanation.as_list()
        weights = [("ss1",-0.8), ("ss2",-0.5), ("ss3",-0.1), ("ss4",0.0), ("ss5",0.1), ("ss6",0.3), ("ss7",0.5), ("ss8",0.8)] # List of tuples
        sorted_weights = sorted(weights, key=lambda tup: tup[1], reverse=True) # Sort from the biggest weight to the lowest
        self.lbl_weights.setText("Poids : " + (str(sorted_weights))[1:(len(str(sorted_weights))-1)]) # Print the weights
        #self.figure = UI_Explanation.explanation.as_pyplot_figure()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(ts, linestyle='-')
        self.canvas.draw()
        """
        #ts = [-0.34086,-0.38038,-0.3458,-0.36556,-0.3458,-0.36556,-0.3952,-0.38038,-0.38532,-0.3952,-0.38038,-0.35568,-0.34086,-0.32604,-0.2964,-0.2964,-0.33098,-0.30134,-0.30134,-0.3211,-0.28652, -0.34086,-0.32604,-0.2964, -0.32604, -0.32604]        
        #weights = [("ss1",-0.88), ("ss2",-0.5), ("ss3",-0.1), ("ss4",0.0), ("ss5",0.1), ("ss6",0.3), ("ss7",0.5), ("ss8",0.8)]
        colors = {-1:"#ff0000", -0.9:"#ff1400", -0.8:"#ff2801", -0.7:"#ff3c00", -0.6:"#ff5101", -0.5:"#ff6500", -0.4:"#ff7900", -0.3:"#ff8d01", -0.2:"#ffa100", -0.1:"#ffb501", 0:"#ffc900", 0.1:"#e8c403", 0.2:"#d2bd06", 0.3:"#bab709", 0.4:"#a4b10b", 0.5:"#8daa0e", 0.6:"#76a310", 0.7:"#5f9d13", 0.8:"#489816", 0.9:"#489816", 1:"#1b8b1b"}
        segment_length = int((len(ts)-1) / len(weights)) # gérer les arrondis comme dans lime ??
        k = 0
        for elem in weights:
            start = int(k*segment_length)
            if k==(len(weights) - 1):
                segment_length = len(ts) - 1 - k*segment_length
            weight = round(elem[1],1)
            x = np.arange(0.0, len(ts), 1)
            y1 = np.ma.masked_where((x<start), ts)
            curve = np.ma.masked_where((x>(start + segment_length)), y1)
            ax.plot(curve, linestyle='-', color=colors[weight])
            plt.axvline(x=start, linewidth=1, color="grey")
            k = k + 1
            #print(str(k)+" "+str(segment_length))
            # Changer le x sur les abscisses (pour compter le nb de segments) ?
        plt.axvline(x=(start+segment_length), linewidth=1, color="grey")
        """


    # Show the window
    def showUI(self):#, explanation, result_class):
        #UI_Explanation.explanation = explanation
        result_class = "classe1"
        self.lbl_class.setText("Résultat : " + result_class)
        self.plot()
        self.show()


# Test
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = UI_Explanation()
    main.showUI()
    sys.exit(app.exec_())