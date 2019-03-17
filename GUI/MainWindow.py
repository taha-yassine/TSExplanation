import sys
sys.path.insert(0, "../Classifier")
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QSizePolicy, QSpinBox
from PyQt5.QtGui import QIcon, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import ExplanationWindow
import importTS
import LearningClassifier

class App(QWidget):

    fileNameTS = ''
    fileNameSh = ''
    X_train = None 
    X_trainSh = None
    fileNameCl = ''
    explanation = ''

    def __init__(self):
        super().__init__()
        self.title='TSExplanation'
        self.initUI()
        self.setStyleSheet(open("style.qss", "r").read())

        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(400, 200, 500, 450)  
        self.setMinimumSize(500, 450)
        self.setWindowIcon(QtGui.QIcon("icons/TSExplanation.ico"))
        self.tabWidget = QtWidgets.QTabWidget(self)
        self.tabWidget.setGeometry(QtCore.QRect(0, 2, 502, 450))
        self.tabWidget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        
        with open("ListeTS.csv","r") as file:
            listeTS = file.readline().split(";")

        #-----------------------------------------------------------------------------------------------
        # Tab 1 : Classifier        
        #-----------------------------------------------------------------------------------------------

        self.tab_Classifier = QtWidgets.QWidget()

        # Components

        self.formLayout_Classifier = QtWidgets.QFormLayout(self.tab_Classifier)
        self.formLayout_Classifier.setContentsMargins(18, 18, 18, 18)

        self.lbl_Classifier_SelectTS = QtWidgets.QLabel(self.tab_Classifier)
        self.lbl_Classifier_SelectTS.setText("Fichier des ST d'entraînement")
        self.formLayout_Classifier.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lbl_Classifier_SelectTS)

        self.cb_Classifier_SelectTS = QtWidgets.QComboBox(self.tab_Classifier)
        self.cb_Classifier_SelectTS.addItem("")
        self.cb_Classifier_SelectTS.setItemText(0, "Charger mon fichier ...")
        for ts in listeTS:
            self.cb_Classifier_SelectTS.addItem(ts)
        self.cb_Classifier_SelectTS.activated.connect(lambda idx="Classifier": self.selectionTSChange("Classifier"))
        self.formLayout_Classifier.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.cb_Classifier_SelectTS)

        self.lbl_Classifier = QtWidgets.QLabel(self.tab_Classifier)
        self.lbl_Classifier.setText("Type du classifieur")
        self.formLayout_Classifier.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.lbl_Classifier)

        self.cb_Classifier = QtWidgets.QComboBox(self.tab_Classifier)
        self.cb_Classifier.addItem("")
        self.cb_Classifier.setItemText(0, "1NN-DTW")
        self.cb_Classifier.addItem("Learning Shapelet")
        self.formLayout_Classifier.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.cb_Classifier)

        self.lbl_Classifier_Save = QtWidgets.QLabel(self.tab_Classifier)
        self.lbl_Classifier_Save.setText("Construction et sauvegarde")
        self.formLayout_Classifier.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.lbl_Classifier_Save)

        self.btn_Classifier_Save = QtWidgets.QPushButton(self.tab_Classifier)
        self.btn_Classifier_Save.setText("Sauvegarder ...                     ")
        self.btn_Classifier_Save.clicked.connect(self.saveClassifier) 
        self.formLayout_Classifier.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.btn_Classifier_Save)


        # Add
        self.tabWidget.addTab(self.tab_Classifier, "Classifieur")
       


        #-----------------------------------------------------------------------------------------------
        # Tab 2 : TS        
        #-----------------------------------------------------------------------------------------------

        self.tab_TS = QtWidgets.QWidget()

        # Components

        self.verticalLayout_TS = QtWidgets.QVBoxLayout(self.tab_TS)
        self.verticalLayout_TS.setContentsMargins(20, 20, 20, 20)
        self.verticalLayout_TS.setSpacing(20)
        self.horizontalLayout_TS = QtWidgets.QHBoxLayout()

        self.cb_TS_SelectTS = QtWidgets.QComboBox(self.tab_TS)
        self.cb_TS_SelectTS.addItem("")
        self.cb_TS_SelectTS.setItemText(0, "Charger mon fichier ...")
        for ts in listeTS:
            self.cb_TS_SelectTS.addItem(ts)
        self.cb_TS_SelectTS.activated.connect(lambda idx="TS": self.selectionTSChange("TS"))
        self.horizontalLayout_TS.addWidget(self.cb_TS_SelectTS)

        self.lbl_TS_Index = QtWidgets.QLabel(self.tab_TS)
        self.lbl_TS_Index.setText("Indice :")
        self.horizontalLayout_TS.addWidget(self.lbl_TS_Index)  

        self.txt_TS_Index = QtWidgets.QSpinBox(self.tab_TS)
        self.txt_TS_Index.setMinimum(1)
        self.txt_TS_Index.setEnabled(False)
        self.txt_TS_Index.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.horizontalLayout_TS.addWidget(self.txt_TS_Index)

        self.btn_TS_Affiche = QtWidgets.QPushButton(self.tab_TS)
        self.btn_TS_Affiche.setText("Afficher")
        self.btn_TS_Affiche.clicked.connect(self.showTSPlot)
        self.horizontalLayout_TS.addWidget(self.btn_TS_Affiche)

        self.verticalLayout_TS.addLayout(self.horizontalLayout_TS)
        self.figure_TS = plt.figure()
        self.canvas_TS = FigureCanvas(self.figure_TS)
        self.verticalLayout_TS.addWidget(self.canvas_TS)


        # Add
        self.tabWidget.addTab(self.tab_TS, "ST")


        #-----------------------------------------------------------------------------------------------
        # Tab 3 : Shapelet
        #-----------------------------------------------------------------------------------------------

        self.tab_Shapelet = QtWidgets.QWidget()

        # Components

        self.verticalLayout_Shapelet = QtWidgets.QVBoxLayout(self.tab_Shapelet)     
        self.verticalLayout_Shapelet.setContentsMargins(18, 18, 18, 18)

        self.horizontalLayout_Shapelet = QtWidgets.QHBoxLayout()

        self.verticalLayout_Shapelet_Sh = QtWidgets.QVBoxLayout()
        self.btn_Shapelet_ChargeSh = QtWidgets.QPushButton(self.tab_Shapelet)
        self.btn_Shapelet_ChargeSh.setText("Charger\nShapelet")
        self.btn_Shapelet_ChargeSh.clicked.connect(self.openShFile)
        
        self.verticalLayout_Shapelet_Sh.addWidget(self.btn_Shapelet_ChargeSh)
        self.formLayout_Shapelet_Sh = QtWidgets.QFormLayout()
        self.lbl_Shapelet_IndexSh = QtWidgets.QLabel(self.tab_Shapelet)
        self.lbl_Shapelet_IndexSh.setText("Indice :")
        self.formLayout_Shapelet_Sh.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lbl_Shapelet_IndexSh)
        self.txt_Shapelet_IndexSh = QtWidgets.QSpinBox(self.tab_Shapelet)
        self.txt_Shapelet_IndexSh.setMinimum(1)
        self.txt_Shapelet_IndexSh.setEnabled(False)
        self.txt_Shapelet_IndexSh.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.formLayout_Shapelet_Sh.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.txt_Shapelet_IndexSh)
        self.verticalLayout_Shapelet_Sh.addLayout(self.formLayout_Shapelet_Sh)
        self.horizontalLayout_Shapelet.addLayout(self.verticalLayout_Shapelet_Sh)

        self.line = QtWidgets.QFrame(self.tab_Shapelet)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.horizontalLayout_Shapelet.addWidget(self.line)

        self.verticalLayout_Shapelet_TS = QtWidgets.QVBoxLayout()

        self.lbl_Shapelet_ChargerTS = QtWidgets.QLabel(self.tab_Shapelet)
        self.lbl_Shapelet_ChargerTS.setText("Charger ST :")
        self.verticalLayout_Shapelet_TS.addWidget(self.lbl_Shapelet_ChargerTS)
        self.cb_Shapelet_SelectTS = QtWidgets.QComboBox(self.tab_Shapelet)
        self.cb_Shapelet_SelectTS.addItem("")
        self.cb_Shapelet_SelectTS.setItemText(0, "Charger mon fichier ...")
        for ts in listeTS:
            self.cb_Shapelet_SelectTS.addItem(ts)
        self.cb_Shapelet_SelectTS.activated.connect(lambda idx="Shapelet": self.selectionTSChange("Shapelet"))
        self.verticalLayout_Shapelet_TS.addWidget(self.cb_Shapelet_SelectTS)

        self.formLayout_Shapelet_TS = QtWidgets.QFormLayout()
        self.lbl_Shapelet_IndexTS = QtWidgets.QLabel(self.tab_Shapelet)
        self.lbl_Shapelet_IndexTS.setText("Indice :")
        self.formLayout_Shapelet_TS.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lbl_Shapelet_IndexTS)
        self.txt_Shapelet_IndexTS = QtWidgets.QSpinBox(self.tab_Shapelet)
        self.txt_Shapelet_IndexTS.setMinimum(1)
        self.txt_Shapelet_IndexTS.setEnabled(False)
        self.txt_Shapelet_IndexTS.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.formLayout_Shapelet_TS.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.txt_Shapelet_IndexTS)
        self.verticalLayout_Shapelet_TS.addLayout(self.formLayout_Shapelet_TS)
        self.horizontalLayout_Shapelet.addLayout(self.verticalLayout_Shapelet_TS)
        self.verticalLayout_Shapelet.addLayout(self.horizontalLayout_Shapelet)

        self.btn_Shapelet_Show = QtWidgets.QPushButton(self.tab_Shapelet)
        self.btn_Shapelet_Show.setText("Afficher")
        self.btn_Shapelet_Show.clicked.connect(self.showPlots)
        self.verticalLayout_Shapelet.addWidget(self.btn_Shapelet_Show)

        self.lbl_Shapelet = QtWidgets.QLabel(self.tab_Shapelet)
        self.lbl_Shapelet.setText("Visualisation de la plus petite distance d\'une Shapelet à une ST :")
        self.verticalLayout_Shapelet.addWidget(self.lbl_Shapelet)

        self.figure_Sh = plt.figure()
        self.canvas_Sh = FigureCanvas(self.figure_Sh)
        self.verticalLayout_Shapelet.addWidget(self.canvas_Sh)


        # Add
        self.tabWidget.addTab(self.tab_Shapelet, "Shapelet")



        #-----------------------------------------------------------------------------------------------              
        # Tab 4 : LIME 
        #-----------------------------------------------------------------------------------------------

        self.tab_LIME = QtWidgets.QWidget()

        # Components

        self.formLayout_LIME = QtWidgets.QFormLayout(self.tab_LIME)
        self.formLayout_LIME.setContentsMargins(18, 18, 18, 18)

        self.lbl_LIME_Classifier = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_Classifier.setText("Sélection du classifieur")
        self.formLayout_LIME.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lbl_LIME_Classifier)
        self.btn_LIME_Classifier = QtWidgets.QPushButton(self.tab_LIME)
        self.btn_LIME_Classifier.setText("Charger ...")
        self.btn_LIME_Classifier.clicked.connect(self.openClassifier)
        self.formLayout_LIME.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.btn_LIME_Classifier)

        self.lbl_LIME_SelectTS = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_SelectTS.setText("Sélection du fichier des ST")
        self.formLayout_LIME.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.lbl_LIME_SelectTS)

        self.cb_LIME_SelectTS = QtWidgets.QComboBox(self.tab_LIME)
        self.cb_LIME_SelectTS.addItem("")
        self.cb_LIME_SelectTS.setItemText(0, "Charger mon fichier ...")
        for ts in listeTS:
            self.cb_LIME_SelectTS.addItem(ts)
        self.cb_LIME_SelectTS.activated.connect(lambda idx="LIME": self.selectionTSChange("LIME"))
        self.formLayout_LIME.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.cb_LIME_SelectTS)

        self.lbl_LIME_Index = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_Index.setText("Indice de la ST à expliquer")
        self.formLayout_LIME.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.lbl_LIME_Index)        
        self.txt_LIME_Index = QtWidgets.QSpinBox(self.tab_LIME)
        self.txt_LIME_Index.setMinimum(1)
        self.txt_LIME_Index.setEnabled(False)
        self.txt_LIME_Index.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.formLayout_LIME.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.txt_LIME_Index)

        self.lbl_LIME_Classes = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_Classes.setText("Classes à expliquer")
        self.formLayout_LIME.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.lbl_LIME_Classes)        
        self.cb_LIME_Classes = QtWidgets.QComboBox(self.tab_LIME)
        self.cb_LIME_Classes.addItem("")
        self.cb_LIME_Classes.setItemText(0, "Toutes")
        self.formLayout_LIME.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.cb_LIME_Classes)

        self.lbl_LIME_NbAttributes = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_NbAttributes.setText("Nombre max d\'attributs (-1 = tous)")
        self.formLayout_LIME.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.lbl_LIME_NbAttributes)        
        self.txt_LIME_NbAttributes = QtWidgets.QLineEdit(self.tab_LIME)
        self.txt_LIME_NbAttributes.setText("3")
        self.formLayout_LIME.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.txt_LIME_NbAttributes)

        self.lbl_LIME_Segm = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_Segm.setText("Algorithme de segmentation")
        self.formLayout_LIME.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.lbl_LIME_Segm) 


        self.horizontalLayout_LIME_Widget = QtWidgets.QWidget(self.tab_LIME)
        self.horizontalLayout_LIME = QtWidgets.QHBoxLayout(self.horizontalLayout_LIME_Widget)

        self.cb_LIME_Segm = QtWidgets.QComboBox(self.tab_LIME)
        self.cb_LIME_Segm.addItem("")
        self.cb_LIME_Segm.setItemText(0, "Uniforme") 
        self.cb_LIME_Segm.addItem("Autre")
        self.cb_LIME_Segm.currentIndexChanged.connect(self.selectionSegmChange)
        self.horizontalLayout_LIME.addWidget(self.cb_LIME_Segm)
        self.txt_LIME_Segm = QtWidgets.QSpinBox(self.tab_LIME)    # (pas toujours nécessaire selon l'option choisie)
        #self.txt_LIME_Segm.setMinimum(1)
        self.txt_LIME_Segm.setMinimumWidth(75)
        self.txt_LIME_Segm.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.horizontalLayout_LIME.addWidget(self.txt_LIME_Segm)
        self.horizontalLayout_LIME.setContentsMargins(0, 0, 0, 0)
        self.formLayout_LIME.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_LIME_Widget)

        self.lbl_LIME_RempSS = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_RempSS.setText("Remplacement des sous-séries")
        self.formLayout_LIME.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.lbl_LIME_RempSS)        
        self.cb_LIME_RempSS = QtWidgets.QComboBox(self.tab_LIME)
        self.cb_LIME_RempSS.addItem("")
        self.cb_LIME_RempSS.setItemText(0, "Segments nuls") # Voir les autres possibilités
        self.formLayout_LIME.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.cb_LIME_RempSS)

        self.lbl_LIME_Distance = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_Distance.setText("Distance")
        self.formLayout_LIME.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.lbl_LIME_Distance)        
        self.verticalLayout_LIME_Widget = QtWidgets.QWidget(self.tab_LIME)
        self.verticalLayout_LIME = QtWidgets.QVBoxLayout(self.verticalLayout_LIME_Widget)
        self.rb_LIME_Cos = QtWidgets.QRadioButton(self.tab_LIME)
        self.rb_LIME_Cos.setChecked(True)
        self.rb_LIME_Cos.setText("Cosinus")
        self.verticalLayout_LIME.addWidget(self.rb_LIME_Cos)
        self.rb_LIME_Eucl = QtWidgets.QRadioButton(self.tab_LIME)
        self.rb_LIME_Eucl.setText("Euclidienne")
        self.verticalLayout_LIME.addWidget(self.rb_LIME_Eucl)
        self.formLayout_LIME.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.verticalLayout_LIME_Widget)

        self.btn_LIME_Exec = QtWidgets.QPushButton(self.tab_LIME)
        self.btn_LIME_Exec.setText("Exécuter")
        self.btn_LIME_Exec.clicked.connect(self.execLIME)
        self.formLayout_LIME.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.btn_LIME_Exec)


        # Add
        self.tabWidget.addTab(self.tab_LIME, "LIME")
        
        # Show the window
        self.show()


    ##################################################################################################


    # Action of the button 'Sauvegarder' of the Classifier tab
    def saveClassifier(self):
        fileName, _  = QFileDialog.getSaveFileName(self,"Sauvegarder un fichier","../Classifier/SaveClassifierFiles","Saved classifiers (*.sav)")
        if self.cb_Classifier_SelectTS.currentText() == "Charger mon fichier ...":
            X_train, Y_train = importTS.fileImportTS(App.fileNameTS)
            if self.cb_Classifier.currentText() == "Learning Shapelet":
                clLS = LearningClassifier.learningShapeletClassifier(X_train, Y_train)
                LearningClassifier.saveClassifierLS(clLS, fileName)
            else:
                cl = LearningClassifier.NN1_DTWClassifier(X_train, Y_train)
                LearningClassifier.saveClassifier1NN(cl,fileName)
        else:
            X_train, Y_train, _, _= importTS.dataImport(self.cb_Classifier_SelectTS.currentText())
            if self.cb_Classifier.currentText() == "Learning Shapelet":
                clLS = LearningClassifier.learningShapeletClassifier(X_train, Y_train)
                LearningClassifier.saveClassifierLS(clLS, fileName)
            else:
                cl = LearningClassifier.NN1_DTWClassifier(X_train, Y_train)
                LearningClassifier.saveClassifier1NN(cl, fileName)


    # Action of the button 'Afficher' of the TS tab
    def showTSPlot(self):
        l = App.X_train[self.txt_TS_Index.value() - 1].ravel().tolist()
        self.figure_TS.clear()
        ax = self.figure_TS.add_subplot(111)
        #ax.plot(l, linestyle='-', marker='.', markerfacecolor='#E20047', markeredgecolor='#E20047', markersize=2)
        ax.plot(l, linestyle='-')
        self.canvas_TS.draw()


    # Action of the button 'Charger Shapelet' of the Shapelet tab
    def openShFile(self):
        fileName, _  = QFileDialog.getOpenFileName(self,"Ouvrir un fichier","../Classifier/TimeSeriesFiles","Text files (*.txt)")
        if fileName:
            App.fileNameSh = fileName
            App.X_trainSh,_ = importTS.fileImportTS(App.fileNameSh)
            self.txt_Shapelet_IndexSh.setMaximum(len(App.X_trainSh))
            self.txt_Shapelet_IndexSh.setEnabled(True)


    # Compute the min distance between a Shapelet and a TS
    def minDistance(self, shapelet, ts):
        ts_cleaned = [t for t in ts if str(t) != 'nan'] # Removing all the NaN values to get the exact length of the ts
        if len(ts_cleaned) >= len(shapelet):
            debut = 0
            distanceMin = sys.maxsize
            for i in range(0, len(ts) - len(shapelet) + 1):
                distance = 0
                for j in range(0, len(shapelet)):
                    distance += abs(ts[i+j] - shapelet[j])
                if distance < distanceMin:
                    distanceMin = distance
                    debut = i
            return debut
        else:
            return -1


    # Action of the button 'Afficher' of the Shapelet tab
    def showPlots(self):
        shapelet = App.X_trainSh[self.txt_Shapelet_IndexSh.value()-1].ravel().tolist()
        ts = App.X_train[self.txt_Shapelet_IndexTS.value()-1].ravel().tolist()
        debut = self.minDistance(shapelet, ts)
        if debut == -1:
            print("Erreur : la Shapelet doit être plus courte que la ST !")
        else:
            # Normalizing the Shapelet length to be able to print it on the same graphic than the TS :
            listeG = []
            for i in range(0,debut):
                listeG.append(0)
            listeD = []
            for i in range(debut + len(shapelet), len(ts)):
                listeD.append(0)
            listeG.extend(shapelet)
            listeG.extend(listeD)
            l2 = listeG       
            # Masking the empty parts of the Shapelet :
            x = np.arange(0.0, len(ts), 1)
            y2 = np.ma.masked_where((x<debut), l2)
            y = np.ma.masked_where((x>(debut + len(shapelet) - 1)), y2)
            # Showing the graphic :
            self.figure_Sh.clear()
            ax = self.figure_Sh.add_subplot(111)
            ax.plot(ts, linestyle='-')
            ax.plot(y, linestyle='-', color='red')
            ax.fill_between(x, ts, y, facecolor='#b7b7b7')
            self.canvas_Sh.draw()


    # Action of the button 'Charger' for the classifier of the LIME tab
    def openClassifier(self):
        fileName, _  = QFileDialog.getOpenFileName(self,"Ouvrir un fichier","../Classifier/SaveClassifierFiles","Saved classifiers (*.sav)")
        if fileName:
            App.fileNameCl = fileName


    # Action of the button 'Executer' of the LIME tab
    def execLIME(self): # Show explanation
        App.explanation = "..."
        self.UIexplanation = ExplanationWindow.UI_Explanation()
        self.UIexplanation.showUI(App.explanation)# + result (classifier.predict(myTS))
    

    # Binding between the size of the tab widget and the size of the window
    def resizeEvent(self, resizeEvent):
        self.tabWidget.setGeometry(0, 2, self.geometry().width() + 2, self.geometry().height())
        QtWidgets.QWidget.resizeEvent(self, resizeEvent)


    # Set the max value of the index
    def setIndex(self, cb, index):
        if(cb.currentText() == "Charger mon fichier ..."):
            fileName, _  = QFileDialog.getOpenFileName(self,"Ouvrir un fichier","../Classifier/TimeSeriesFiles","Text files (*.txt)")
            if fileName:
                App.fileNameTS = fileName
                App.X_train,_ = importTS.fileImportTS(App.fileNameTS)
                index.setMaximum(len(App.X_train))
                index.setEnabled(True)
        else:
            App.X_train, _, _, _= importTS.dataImport(cb.currentText())
            index.setMaximum(len(App.X_train))
            index.setEnabled(True)


    # Action when there is a change in the TS file
    def selectionTSChange(self, tab):
        if (tab == "Classifier") and (self.cb_Classifier_SelectTS.currentText() == "Charger mon fichier ..."):
            App.fileNameTS, _  = QFileDialog.getOpenFileName(self,"Ouvrir un fichier","../Classifier/TimeSeriesFiles","Text files (*.txt)")
        if (tab == "TS"):
            self.setIndex(self.cb_TS_SelectTS, self.txt_TS_Index)
        if (tab == "Shapelet"):
            self.setIndex(self.cb_Shapelet_SelectTS, self.txt_Shapelet_IndexTS)
        if (tab == "LIME"):
            self.setIndex(self.cb_LIME_SelectTS, self.txt_LIME_Index)


    # Action when there is a change in the type of segmentation, in the LIME tab
    def selectionSegmChange(self):
        if self.cb_LIME_Segm.currentText() == "Uniforme":
            self.txt_LIME_Segm.setEnabled(True)
        else:
            self.txt_LIME_Segm.setEnabled(False)


    ###########################################################################################
     

if __name__=='__main__':
    app=QApplication(sys.argv)      
    ex=App()
    sys.exit(app.exec_())