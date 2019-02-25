import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QSizePolicy, QSpinBox
from PyQt5.QtGui import QIcon, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import random
import ExplanationWindow

class App(QWidget):

    fileNameTS = ''
    fileNameSh = ''
    fileNameCl = ''
    explanation = ''

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
                color: #40728B; /* Text color */
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
        self.setGeometry(400, 200, 500, 450)  
        #self.setFixedSize(400, 331)
        self.setMinimumSize(500, 450)
        self.tabWidget = QtWidgets.QTabWidget(self)
        self.tabWidget.setGeometry(QtCore.QRect(0, 2, 502, 450))
        self.tabWidget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

        
        #-----------------------------------------------------------------------------------------------
        # Tab 1 : Classifier        
        #-----------------------------------------------------------------------------------------------

        self.tab_Classifier = QtWidgets.QWidget()

        # Components

        self.formLayout_Classifier = QtWidgets.QFormLayout(self.tab_Classifier)
        self.formLayout_Classifier.setContentsMargins(18, 18, 18, 18)

        self.lbl_Classifier_SelectTS = QtWidgets.QLabel(self.tab_Classifier)
        #self.lbl_Classifier_SelectTS.setGeometry(QtCore.QRect(10, 10, 150, 13))
        self.lbl_Classifier_SelectTS.setText("Fichier des ST d'entraînement")
        self.formLayout_Classifier.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lbl_Classifier_SelectTS)

        self.cb_Classifier_SelectTS = QtWidgets.QComboBox(self.tab_Classifier)
        #self.cb_Classifier_SelectTS.setGeometry(QtCore.QRect(200, 10, 151, 23))
        self.cb_Classifier_SelectTS.addItem("")
        self.cb_Classifier_SelectTS.setItemText(0, "Charger mon fichier ...")
        self.cb_Classifier_SelectTS.activated.connect(self.openTSFile)
        self.formLayout_Classifier.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.cb_Classifier_SelectTS)

        self.lbl_Classifier = QtWidgets.QLabel(self.tab_Classifier)
        #self.lbl_Classifier.setGeometry(QtCore.QRect(10, 40, 150, 16))
        self.lbl_Classifier.setText("Type du classifieur")
        self.formLayout_Classifier.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.lbl_Classifier)

        self.cb_Classifier = QtWidgets.QComboBox(self.tab_Classifier)
        #self.cb_Classifier.setGeometry(QtCore.QRect(200, 40, 151, 22))
        self.cb_Classifier.addItem("")
        self.cb_Classifier.setItemText(0, "1NN-DTW")
        self.cb_Classifier.addItem("Learning Shapelet")
        self.formLayout_Classifier.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.cb_Classifier)

        self.lbl_Classifier_Save = QtWidgets.QLabel(self.tab_Classifier)
        #self.lbl_Classifier_Save.setGeometry(QtCore.QRect(10, 70, 150, 16))
        self.lbl_Classifier_Save.setText("Construction et sauvegarde")
        self.formLayout_Classifier.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.lbl_Classifier_Save)

        self.btn_Classifier_Save = QtWidgets.QPushButton(self.tab_Classifier)
        #self.btn_Classifier_Save.setGeometry(QtCore.QRect(200, 70, 151, 23))
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

        self.btn_TS_Charge = QtWidgets.QPushButton(self.tab_TS)
        self.btn_TS_Charge.setText("Charger ...")
        #self.btn_TS_Charge.clicked.connect(self.openTSFile)
        self.btn_TS_Charge.clicked.connect(lambda idx="TS": self.openTSFile("TS"))
        self.horizontalLayout_TS.addWidget(self.btn_TS_Charge)

        self.lbl_TS_Index = QtWidgets.QLabel(self.tab_TS)
        self.lbl_TS_Index.setText("Indice :")
        self.horizontalLayout_TS.addWidget(self.lbl_TS_Index)  

        #self.txt_TS_Index = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        #self.txt_TS_Index.setGeometry(QtCore.QRect(180, 20, 50, 20))
        #self.txt_TS_Index.setText("54")
        self.txt_TS_Index = QtWidgets.QSpinBox(self.tab_TS)
        self.txt_TS_Index.setMinimum(1)
        self.txt_TS_Index.setEnabled(False)
        self.txt_TS_Index.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.horizontalLayout_TS.addWidget(self.txt_TS_Index)

        self.btn_TS_Affiche = QtWidgets.QPushButton(self.tab_TS)
        #self.btn_TS_Affiche.setGeometry(QtCore.QRect(290, 20, 75, 23))
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
        #self.txt_Shapelet_IndexSh = QtWidgets.QLineEdit(self.tab_Shapelet)
        #self.txt_Shapelet_IndexSh.setText("8")
        self.txt_Shapelet_IndexSh = QtWidgets.QSpinBox(self.tab_Shapelet)
        self.txt_Shapelet_IndexSh.setMinimum(1)
        self.txt_Shapelet_IndexSh.setEnabled(False)
        self.txt_Shapelet_IndexSh.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.formLayout_Shapelet_Sh.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.txt_Shapelet_IndexSh)
        self.verticalLayout_Shapelet_Sh.addLayout(self.formLayout_Shapelet_Sh)
        self.horizontalLayout_Shapelet.addLayout(self.verticalLayout_Shapelet_Sh)

        self.img_Shapelet_Shapelet = QtWidgets.QLabel(self.tab_Shapelet) 
        fileName = QtCore.QDir.currentPath() + "\\icons\\ShapeletDistance.JPG"
        self.img_Shapelet_Shapelet.setPixmap(QPixmap(fileName).scaled(100, 70, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
        self.horizontalLayout_Shapelet.addWidget(self.img_Shapelet_Shapelet)

        self.line = QtWidgets.QFrame(self.tab_Shapelet)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.horizontalLayout_Shapelet.addWidget(self.line)

        self.verticalLayout_Shapelet_TS = QtWidgets.QVBoxLayout()
        self.btn_Shapelet_ChargeTS = QtWidgets.QPushButton(self.tab_Shapelet)
        self.btn_Shapelet_ChargeTS.setText("Charger\nST")
        #self.btn_Shapelet_ChargeTS.clicked.connect(self.openTSFile)
        self.btn_Shapelet_ChargeTS.clicked.connect(lambda idx="Shapelet": self.openTSFile("Shapelet"))
        self.verticalLayout_Shapelet_TS.addWidget(self.btn_Shapelet_ChargeTS)
        self.formLayout_Shapelet_TS = QtWidgets.QFormLayout()
        self.lbl_Shapelet_IndexTS = QtWidgets.QLabel(self.tab_Shapelet)
        self.lbl_Shapelet_IndexTS.setText("Indice :")
        self.formLayout_Shapelet_TS.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lbl_Shapelet_IndexTS)
        #self.txt_Shapelet_IndexTS = QtWidgets.QLineEdit(self.tab_Shapelet)
        #self.txt_Shapelet_IndexTS.setText("54")
        self.txt_Shapelet_IndexTS = QtWidgets.QSpinBox(self.tab_Shapelet)
        self.txt_Shapelet_IndexTS.setMinimum(1)
        self.txt_Shapelet_IndexTS.setEnabled(False)
        self.txt_Shapelet_IndexTS.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


        self.formLayout_Shapelet_TS.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.txt_Shapelet_IndexTS)
        self.verticalLayout_Shapelet_TS.addLayout(self.formLayout_Shapelet_TS)
        self.horizontalLayout_Shapelet.addLayout(self.verticalLayout_Shapelet_TS)

        self.img_Shapelet_TS = QtWidgets.QLabel(self.tab_Shapelet)
        fileName = QtCore.QDir.currentPath() + "\\icons\\ShapeletDistance.JPG"
        self.img_Shapelet_TS.setPixmap(QPixmap(fileName).scaled(100, 70, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
        self.horizontalLayout_Shapelet.addWidget(self.img_Shapelet_TS)

        self.verticalLayout_Shapelet.addLayout(self.horizontalLayout_Shapelet)

        self.btn_Shapelet_Show = QtWidgets.QPushButton(self.tab_Shapelet)
        self.btn_Shapelet_Show.setText("Afficher")
        self.btn_Shapelet_Show.clicked.connect(self.showPlots)
        self.verticalLayout_Shapelet.addWidget(self.btn_Shapelet_Show)

        self.lbl_Shapelet = QtWidgets.QLabel(self.tab_Shapelet)
        self.lbl_Shapelet.setText("Visualisation de la distance la plus petite d\'une Shapelet à une ST :")
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
        self.btn_LIME_SelectTS = QtWidgets.QPushButton(self.tab_LIME)
        self.btn_LIME_SelectTS.setText("Charger ...")
        #self.btn_LIME_SelectTS.clicked.connect(self.openTSFile) 
        self.btn_LIME_SelectTS.clicked.connect(lambda idx="LIME": self.openTSFile("LIME"))
        self.formLayout_LIME.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.btn_LIME_SelectTS)

        self.lbl_LIME_Index = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_Index.setText("Indice de la ST à expliquer")
        self.formLayout_LIME.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.lbl_LIME_Index)        
        #self.txt_LIME_Index = QtWidgets.QLineEdit(self.tab_LIME)
        #self.txt_LIME_Index.setText("8")
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
        self.cb_LIME_Segm = QtWidgets.QComboBox(self.tab_LIME)
        self.cb_LIME_Segm.addItem("")
        self.cb_LIME_Segm.setItemText(0, "??")
        self.formLayout_LIME.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.cb_LIME_Segm)

        self.lbl_LIME_RempSS = QtWidgets.QLabel(self.tab_LIME)
        self.lbl_LIME_RempSS.setText("Remplacement des sous-séries")
        self.formLayout_LIME.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.lbl_LIME_RempSS)        
        self.cb_LIME_RempSS = QtWidgets.QComboBox(self.tab_LIME)
        self.cb_LIME_RempSS.addItem("")
        self.cb_LIME_RempSS.setItemText(0, "Segments nuls")
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


    # Action of the item 'Charger mon fichier' of the Classifier tab
    #           the button 'Charger' of the TS tab
    #           the button 'Charger ST' of the Shapelet tab 
    #           the button 'Charger' for the TS file of the LIME tab
    def openTSFile(self, tab):
        fileName, _  = QFileDialog.getOpenFileName(self,"Ouvrir un fichier",QtCore.QDir.currentPath(),"Text files (*.txt)")
        if fileName:
            App.fileNameTS = fileName
            num_lines = sum(1 for line in open(App.fileNameTS))
            if tab == 'TS':
                self.txt_TS_Index.setMaximum(num_lines)
                self.txt_TS_Index.setEnabled(True)
            if tab == 'Shapelet':
                self.txt_Shapelet_IndexTS.setMaximum(num_lines)
                self.txt_Shapelet_IndexTS.setEnabled(True)
            if tab == 'LIME':
                self.txt_LIME_Index.setMaximum(num_lines)
                self.txt_LIME_Index.setEnabled(True)


    def test(self, tab):
        print(tab)


    # Action of the button 'Sauvegarder' of the Classifier tab
    def saveClassifier(self):
        fileName, _  = QFileDialog.getSaveFileName(self,"Sauvegarder un fichier",QtCore.QDir.currentPath(),"Saved classifiers (*.sav)") 

    # Action of the button 'Afficher' of the TS tab
    #def showTSPlot(self):
    #    data = [random.random() for i in range(11)]
    #    self.figure_TS.clear()
    #    ax = self.figure_TS.add_subplot(111)
    #    ax.plot(data, '*-')
    #    self.canvas_TS.draw()

    # Action of the button 'Afficher' of the TS tab
    def showTSPlot(self):
        #if App.fileNameTS != '' and self.txt_TS_Index.text().isdigit():
        if App.fileNameTS != '':
            with open(App.fileNameTS,'r') as file:
                #for i in range(int(self.txt_TS_Index.text())):                
                for i in range(self.txt_TS_Index.value()):
                    line = file.readline()
                l = line.split(",")
                l = list(map(float, l))
                self.figure_TS.clear()
                ax = self.figure_TS.add_subplot(111)
                ax.plot(l, linestyle='-', marker='.', markerfacecolor='#E20047', markeredgecolor='#E20047', markersize=2)
                self.canvas_TS.draw()
                ############# changer la taille des points, possibilité de zoomer ??, gestion des exc ? 1 par défaut ? Spiner ?

    # Action of the button 'Charger Shapelet' of the Shapelet tab
    def openShFile(self):
        fileName, _  = QFileDialog.getOpenFileName(self,"Ouvrir un fichier",QtCore.QDir.currentPath(),"Text files (*.txt)")
        if fileName:
            App.fileNameSh = fileName
            num_lines = sum(1 for line in open(App.fileNameSh))
            self.txt_Shapelet_IndexSh.setMaximum(num_lines)
            self.txt_Shapelet_IndexSh.setEnabled(True)

    # Action of the button 'Afficher' of the Shapelet tab
    def showPlots(self):
        data = [random.random() for i in range(11)]
        self.figure_Sh.clear()
        ax = self.figure_Sh.add_subplot(111)
        ax.plot(data, '*-')
        self.canvas_Sh.draw()

    # Action of the button 'Charger' for the classifier of the LIME tab
    def openClassifier(self):
        fileName, _  = QFileDialog.getOpenFileName(self,"Ouvrir un fichier",QtCore.QDir.currentPath(),"Saved classifiers (*.sav)")
        if fileName:
            App.fileNameCl = fileName

    # Action of the button 'Executer' of the LIME tab
    def execLIME(self): # Show explanation
        App.explanation = "..."
        self.UIexplanation = ExplanationWindow.UI_Explanation()
        self.UIexplanation.showUI(explanation)# + result (classifier.predict(myTS))
    
    # Binding between the size of the tab widget and the size of the window
    def resizeEvent(self, resizeEvent):
        self.tabWidget.setGeometry(0, 2, self.geometry().width() + 2, self.geometry().height())
        QtWidgets.QWidget.resizeEvent(self, resizeEvent)


    ###########################################################################################
     

if __name__=='__main__':
    app=QApplication(sys.argv)      
    ex=App()
    sys.exit(app.exec_())