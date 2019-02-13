import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout
from PyQt5.QtGui import QIcon, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import random
#import GUI_Graphe

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title='TSExplanation'
        self.initUI()
        '''f = QtCore.QFile(QtCore.QDir.currentPath() + "/StyleSheet.qss")
        ts = QtCore.QTextStream(f)
        stylesheet = ts.readAll()
        self.setStyleSheet(stylesheet)'''
        self.setStyleSheet("""
			QTabWidget::tab-bar {
			    left: 5px; /* Move to the right by 5px */
			}
			QTabBar::tab {
			    border: 1px solid #B8CDDE;
			    border-top-left-radius: 4px;
			    border-top-right-radius: 4px;
			    min-width: 8ex;
			    padding: 2px;
			    width: 70px; /* Each tab has the same width */
			    height: 20px;
			    background: #D5ECFA;
			    margin-top: 5px;
			    font: bold 12px;
			    color: #40728B;	/* Text color */
			}
			QTabBar::tab:hover {
			    background: #9EC8FF;
			}
			QTabBar::tab:selected {
				background: #53A9C1;
			    border-color: #9B9B9B;
			    margin-top: 4px;
			    color: white;
			}
			QTabWidget::pane {
				background: red;
			}
        """)
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(400, 200, 400, 331)  
        self.setFixedSize(400, 331)
        self.tabWidget = QtWidgets.QTabWidget(self)
        self.tabWidget.setGeometry(QtCore.QRect(0, 2, 402, 331))
        #self.tabWidget.setGeometry(QtCore.QRect(0, 2, 602, 450))
        

        # Tab 1 : Classifier        
        self.tab_Classifier = QtWidgets.QWidget()

        # Components

        self.lbl_Classifier_Examples = QtWidgets.QLabel(self.tab_Classifier)
        self.lbl_Classifier_Examples.setGeometry(QtCore.QRect(10, 10, 150, 13))
        self.lbl_Classifier_Examples.setText("Fichier des ST d'entraînement")

        self.cb_Classifier_SelectCl = QtWidgets.QComboBox(self.tab_Classifier)
        self.cb_Classifier_SelectCl.setGeometry(QtCore.QRect(200, 10, 151, 23))
        self.cb_Classifier_SelectCl.addItem("")
        self.cb_Classifier_SelectCl.setItemText(0, "Charger mon fichier ...")

        self.lbl_Classifier_Distance = QtWidgets.QLabel(self.tab_Classifier)
        self.lbl_Classifier_Distance.setGeometry(QtCore.QRect(10, 40, 150, 16))
        self.lbl_Classifier_Distance.setText("Type du classifieur")
        self.cb_Classifier_Distance = QtWidgets.QComboBox(self.tab_Classifier)
        self.cb_Classifier_Distance.setGeometry(QtCore.QRect(200, 40, 151, 22))
        self.cb_Classifier_Distance.addItem("")
        self.cb_Classifier_Distance.setItemText(0, "1NN-DTW")

        self.lbl_Classifier_Save = QtWidgets.QLabel(self.tab_Classifier)
        self.lbl_Classifier_Save.setGeometry(QtCore.QRect(10, 70, 150, 16))
        self.lbl_Classifier_Save.setText("Construction et sauvegarde")
        self.btn_Classifier_Save = QtWidgets.QPushButton(self.tab_Classifier)
        self.btn_Classifier_Save.setGeometry(QtCore.QRect(200, 70, 151, 23))
        self.btn_Classifier_Save.setText("Sauvegarder ...                     ")
        self.btn_Classifier_Save.clicked.connect(self.saveClassifier) 

        # Add
        self.tabWidget.addTab(self.tab_Classifier, "Classifieur")
       

        # Tab 2 : TS        
        self.tab_TS = QtWidgets.QWidget()

        # Components

        self.btn_TS_Charge = QtWidgets.QPushButton(self.tab_TS)
        self.btn_TS_Charge.setGeometry(QtCore.QRect(10, 20, 75, 23))
        self.btn_TS_Charge.setText("Charger ...")
        self.btn_TS_Charge.clicked.connect(self.openTSImage)  # Action of the button

        self.lbl_TS_Index = QtWidgets.QLabel(self.tab_TS)
        self.lbl_TS_Index.setGeometry(QtCore.QRect(140, 20, 50, 20))
        self.lbl_TS_Index.setText("Indice :")
        self.txt_TS_Index = QtWidgets.QLineEdit(self.tab_TS)
        self.txt_TS_Index.setGeometry(QtCore.QRect(180, 20, 50, 20))
        self.txt_TS_Index.setText("54")

        self.btn_TS_Affiche = QtWidgets.QPushButton(self.tab_TS)
        self.btn_TS_Affiche.setGeometry(QtCore.QRect(290, 20, 75, 23))
        self.btn_TS_Affiche.setText("Afficher")
        #self.btn_TS_Affiche.clicked.connect(self.showTSImage)
        
        self.img_TS_TS = QtWidgets.QLabel(self.tab_TS)
        self.img_TS_TS.setGeometry(QtCore.QRect(60, 80, 256, 192))
        '''self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setGeometry(QtCore.QRect(60, 80, 256, 192))

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.tab_TS.setLayout(layout)'''

        # Add
        self.tabWidget.addTab(self.tab_TS, "ST")


        # Tab 3 : Shapelet
        self.tab_Shapelet = QtWidgets.QWidget()

        # Components

        self.btn_Shapelet_ChargeSh = QtWidgets.QPushButton(self.tab_Shapelet)
        self.btn_Shapelet_ChargeSh.setGeometry(QtCore.QRect(10, 10, 75, 40))
        self.btn_Shapelet_ChargeSh.setText("Charger\nShapelet")
        self.btn_Shapelet_ChargeSh.clicked.connect(self.openImageShapelet) 
        self.img_Shapelet_Shapelet = QtWidgets.QLabel(self.tab_Shapelet) 
        self.img_Shapelet_Shapelet.setGeometry(QtCore.QRect(100, 10, 71, 71))

        self.lbl_Shapelet_IndexSh = QtWidgets.QLabel(self.tab_Shapelet)
        self.lbl_Shapelet_IndexSh.setGeometry(QtCore.QRect(10, 55, 50, 20))
        self.lbl_Shapelet_IndexSh.setText("Indice :")
        self.txt_Shapelet_IndexSh = QtWidgets.QLineEdit(self.tab_Shapelet)
        self.txt_Shapelet_IndexSh.setGeometry(QtCore.QRect(50, 55, 35, 20))
        self.txt_Shapelet_IndexSh.setText("8")

        self.btn_Shapelet_ChargeTS = QtWidgets.QPushButton(self.tab_Shapelet)
        self.btn_Shapelet_ChargeTS.setGeometry(QtCore.QRect(220, 10, 75, 40))
        self.btn_Shapelet_ChargeTS.setText("Charger\nST")
        self.btn_Shapelet_ChargeTS.clicked.connect(self.openImageShapeletTS)
        self.img_Shapelet_TS = QtWidgets.QLabel(self.tab_Shapelet)
        self.img_Shapelet_TS.setGeometry(QtCore.QRect(310, 10, 71, 71))

        self.lbl_Shapelet_IndexTS = QtWidgets.QLabel(self.tab_Shapelet)
        self.lbl_Shapelet_IndexTS.setGeometry(QtCore.QRect(220, 55, 50, 20))
        self.lbl_Shapelet_IndexTS.setText("Indice :")
        self.txt_Shapelet_IndexTS = QtWidgets.QLineEdit(self.tab_Shapelet)
        self.txt_Shapelet_IndexTS.setGeometry(QtCore.QRect(260, 55, 35, 20))
        self.txt_Shapelet_IndexTS.setText("54")

        self.btn_Shapelet_Show = QtWidgets.QPushButton(self.tab_Shapelet)
        self.btn_Shapelet_Show.setGeometry(QtCore.QRect(150, 95, 75, 20))
        self.btn_Shapelet_Show.setText("Afficher")

        self.lbl_Shapelet = QtWidgets.QLabel(self.tab_Shapelet)
        self.lbl_Shapelet.setGeometry(QtCore.QRect(20, 120, 351, 20))
        self.lbl_Shapelet.setText("Visualisation de la distance la plus petite d\'une Shapelet à une ST :")
        self.img_Shapelet_Distance = QtWidgets.QLabel(self.tab_Shapelet)
        self.img_Shapelet_Distance.setGeometry(QtCore.QRect(110, 150, 171, 131))
        fileName = QtCore.QDir.currentPath() + "\\icons\\ShapeletDistance.JPG"
        self.img_Shapelet_Distance.setPixmap(QPixmap(fileName).scaled(171, 131, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

        # Add
        self.tabWidget.addTab(self.tab_Shapelet, "Shapelet")

              
        # Tab 4 : LIME 
        self.tab_LIME = QtWidgets.QWidget()

        # Components

        self.lbl_LIME_Classifier = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_Classifier.setGeometry(QtCore.QRect(10, 10, 150, 13))
        self.lbl_LIME_Classifier.setText("Sélection du classifieur")
        self.btn_LIME_Classifier = QtWidgets.QPushButton(self.tab_LIME)
        self.btn_LIME_Classifier.setGeometry(QtCore.QRect(200, 10, 151, 22))
        self.btn_LIME_Classifier.setText("Charger ...                             ")
        self.btn_LIME_Classifier.clicked.connect(self.openClassifier)

        self.lbl_LIME_SelectTS = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_SelectTS.setGeometry(QtCore.QRect(10, 40, 150, 16))
        self.lbl_LIME_SelectTS.setText("Sélection du fichier des ST")
        self.btn_LIME_SelectTS = QtWidgets.QPushButton(self.tab_LIME)
        self.btn_LIME_SelectTS.setGeometry(QtCore.QRect(200, 40, 151, 23))
        self.btn_LIME_SelectTS.setText("Charger ...                             ")
        self.btn_LIME_SelectTS.clicked.connect(self.openTSExamples) 

        self.lbl_LIME_Index = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_Index.setGeometry(QtCore.QRect(10, 70, 150, 16))
        self.lbl_LIME_Index.setText("Indice de la ST à expliquer")
        self.cb_LIME_Index = QtWidgets.QLineEdit(self.tab_LIME)
        self.cb_LIME_Index.setGeometry(QtCore.QRect(200, 70, 151, 20))
        self.cb_LIME_Index.setText("8")

        self.lbl_LIME_Classes = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_Classes.setGeometry(QtCore.QRect(10, 100, 150, 16))
        self.lbl_LIME_Classes.setText("Classes à expliquer")
        self.cb_LIME_Classes = QtWidgets.QComboBox(self.tab_LIME)
        self.cb_LIME_Classes.setGeometry(QtCore.QRect(200, 100, 151, 22))
        self.cb_LIME_Classes.addItem("")
        self.cb_LIME_Classes.setItemText(0, "Toutes")

        self.lbl_LIME_NbAttributes = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_NbAttributes.setGeometry(QtCore.QRect(10, 130, 171, 16))
        self.lbl_LIME_NbAttributes.setText("Nombre max d\'attributs (-1 = tous)")
        self.txt_LIME_NbAttributes = QtWidgets.QLineEdit(self.tab_LIME)
        self.txt_LIME_NbAttributes.setGeometry(QtCore.QRect(200, 130, 151, 20))
        self.txt_LIME_NbAttributes.setText("3")

        self.lbl_LIME_Distance = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_Distance.setGeometry(QtCore.QRect(10, 160, 150, 16))
        self.lbl_LIME_Distance.setText("Distance")
        self.rb_LIME_Cos = QtWidgets.QRadioButton(self.tab_LIME)
        self.rb_LIME_Cos.setGeometry(QtCore.QRect(100, 160, 82, 17))
        self.rb_LIME_Cos.setChecked(True)
        self.rb_LIME_Cos.setText("Cosinus")
        self.rb_LIME_Eucl = QtWidgets.QRadioButton(self.tab_LIME)
        self.rb_LIME_Eucl.setGeometry(QtCore.QRect(100, 180, 82, 17))
        self.rb_LIME_Eucl.setText("Euclidienne")

        self.btn_LIME_Exec = QtWidgets.QPushButton(self.tab_LIME)
        self.btn_LIME_Exec.setGeometry(QtCore.QRect(290, 200, 61, 23))
        self.btn_LIME_Exec.setText("Exécuter")
        #self.btn_LIME_Exec.clicked.connect(self.execLIME)

        # Add
        self.tabWidget.addTab(self.tab_LIME, "LIME")
        
        # Show the window
        self.show()

 
    # Action of the button 'Charger' of the TS tab
    def openTSImage(self):
        fileName, _  = QFileDialog.getOpenFileName(self,"Ouvrir un fichier",QtCore.QDir.currentPath(),"All Files (*)") # Filtering types : "Fichiers d'image (*.jpg, *.jpeg, *.gif, *.png)"
        if fileName:
            self.img_TS_TS.setPixmap(QPixmap(fileName).scaled(256, 192, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    # Action of the button 'Charger' of the TS tab
    '''def showTSImage(self):
        data = [random.random() for i in range(11)]
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(data, '*-')
        self.canvas.draw()'''

    # Action of the button 'Charger Shapelet' of the Shapelet tab
    def openImageShapelet(self):
        fileName, _  = QFileDialog.getOpenFileName(self,"Ouvrir un fichier",QtCore.QDir.currentPath(),"All Files (*)")
        if fileName:
            self.img_Shapelet_Shapelet.setPixmap(QPixmap(fileName).scaled(71, 71, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    # Action of the button 'Charger ST' of the Shapelet tab
    def openImageShapeletTS(self):
        fileName, _  = QFileDialog.getOpenFileName(self,"Ouvrir un fichier",QtCore.QDir.currentPath(),"All Files (*)")
        if fileName:
            self.img_Shapelet_TS.setPixmap(QPixmap(fileName).scaled(71, 71, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    # Action of the button 'Charger' for the TS examples of the Classifier tab
    def openTSExamples(self):
        fileName, _  = QFileDialog.getOpenFileName(self,"Ouvrir un fichier",QtCore.QDir.currentPath(),"Text files (*.txt)")

        # Action of the button 'Charger' for the TS of the LIME tab
    def openClassifier(self):
        fileName, _  = QFileDialog.getOpenFileName(self,"Ouvrir un fichier",QtCore.QDir.currentPath(),"Saved classifiers (*.sav)")

    # Action of the button 'Sauvegarder' of the Classifier tab
    def saveClassifier(self):
        fileName, _  = QFileDialog.getSaveFileName(self,"Sauvegarder un fichier",QtCore.QDir.currentPath(),"Saved classifiers (*.sav)") 

    # Action of the button 'Executer' of the LIME tab
    #def execLIME(self):
        # Show explanation

    # Random graph
    '''def plot(self):
        data = [random.random() for i in range(10)] # random data
        self.figure.clear()
        ax = self.figure.add_subplot(111) # create an axis
        ax.plot(data, '*-') # plot data
        self.canvas.draw() # refresh canvas'''
    
        
if __name__=='__main__':
    app=QApplication(sys.argv)      
    ex=App()
    sys.exit(app.exec_())