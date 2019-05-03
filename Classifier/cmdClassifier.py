import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str, help="TS used to learn the classifier")
parser.add_argument("classifier_type", type=str, choices=['1NN','1NN-DTW','LS'], help="classifier's type")
parser.add_argument("output_name", type=str, help="name of the output file, it will be followed by the <classifier_type>.")

parser.add_argument("-p", "--perso", help="indicate that the TS file <input_file> is on the computer",
                    action="store_false")
# default = pas dans la ligne de commande
# const = dans la ligne mais pas renseigné

args = parser.parse_args()

import importTS
import LearningClassifier

#fileName, _  = QFileDialog.getSaveFileName(self,"Sauvegarder un fichier","../Classifier/SaveClassifierFiles","Saved classifiers (*.sav)")
if args.perso == True:
    X_train, Y_train = importTS.fileImportTS(args.input_file)
else:
    X_train, Y_train, _, _= importTS.dataImport(args.input_file)

if args.classifier_type == "LS":
    classifier = LearningClassifier.learningShapeletClassifier(X_train, Y_train)
    LearningClassifier.saveClassifierLS(classifier, fileName)
elif args.classifier_type == "1NN-DTW":
    classifier = LearningClassifier.NN1_DTWClassifier(X_train, Y_train)
    LearningClassifier.saveClassifier1NN(classifier,fileName)
else:
    classifier = LearningClassifier.NN1_Classifier(X_train, Y_train)
    LearningClassifier.saveClassifier1NN(classifier,fileName)

print("Classifier saved !")