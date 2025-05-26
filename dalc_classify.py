#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
DOMAIN ADAPTATION OF LINEAR CLASSIFIERS (aka DALC)
See: http://arxiv.org/abs/1506.04573

Executable script to use the classifier (to be used after the learning process).

@author: Pascal Germain -- http://researchers.lille.inria.fr/pgermain/
'''
import common
from dalc import *
from dataset import *
from kernel import *

import sys
import pickle
import argparse 
import os
common.print_header('CLASSIFICATION')

# Arguments parser   
parser = argparse.ArgumentParser(description="", formatter_class=common.custom_formatter, epilog="")
parser.add_argument("-b", dest="B_value", type=float, default=1.0, help="Trade-off parameter \"B\" (source joint error modifier). Default: 1.0")
parser.add_argument("-c", dest="C_value", type=float, default=1.0, help="Trade-off parameter \"C\" (target disagreement modifier). Default: 1.0")
parser.add_argument("--nb_restarts", "-n", dest="nb_restarts",  type=int, default=1, help='Number of random restarts of the optimization process. Default: 1')
parser.add_argument("--format", "-f", dest="format",  choices=['matrix', 'svmlight'], default='matrix', help='Datasets format. Default: matrix (each line defines an example, the first column defines the label in {-1, 1}, and the next columns represent the real-valued features)')
parser.add_argument("--model",  "-m", dest="model_file", default='model.bin', help="Model file name. Default: model.bin")
parser.add_argument("--pred",   "-p", dest="prediction_file", default='predictions.out', help="Save predictions into files. Default: predictions.out")
parser.add_argument('--nodalc', action='store_true')
parser.add_argument('--postOptimize', dest="post", action='store_true')
parser.add_argument("source_file", help="Defines the file containing the dataset to train.")
parser.add_argument("test_file", help="Defines the file containing the dataset to classify.")
args = parser.parse_args()

# Main program
###############################################################################
print('... Loading model file ...')
###############################################################################
try:
    with open(args.model_file, 'rb') as model:
        classifier = pickle.load(model)
except:
    print('ERROR: Unable to load model file "' + args.model_file + '".')
    sys.exit(-1)

print('File "' + args.model_file + '" loaded.')

###############################################################################
print('\n... Loading dataset file ...')
print("Loading ", args.test_file, " ...")
###############################################################################
try:
    if args.format == 'matrix':
        source_data = dataset_from_matrix_file(args.source_file)
        test_data = dataset_from_matrix_file(args.test_file)
    elif args.format == 'svmlight':   
        source_data = dataset_from_svmlight_file(args.source_file, classifier.X1_shape[1])  
        test_data = dataset_from_svmlight_file(args.test_file, classifier.X1_shape[1])
except:
    print('ERROR: Unable to load test file "' + args.test_file + '".')
    sys.exit(-1)
 
print(str(test_data.get_nb_examples()) + ' test examples loaded.')

###############################################################################
print('\n... Prediction ...')
###############################################################################
predictions = classifier.predict(test_data.X)
accuracy = classifier.calc_accuracy(Y=test_data.Y, predictions=predictions)
classifier.calc_cost(source_data, test_data)
print("Accuracy: ", accuracy)
try:
    predictions.tofile(os.path.join("/home/wang/Data/android", args.prediction_file), '\n')
    print('File "' + args.prediction_file + '" created.')
except:
    print('ERROR: Unable to write prediction file "' + args.prediction_file + '".')

risk = classifier.calc_risk(test_data.Y, predictions=predictions)


if(args.post):
    if not np.any(classifier.alpha_vector):
        raise ValueError("alpha_vector is all zeros.")
    algo = Dalc(C=args.C_value, B=args.B_value, verbose=True, nb_restarts=args.nb_restarts, nodalc=False, post=True, alpha=classifier.alpha_vector)
    new_classifier = algo.learn(source_data, test_data, classifier.kernel)
    predictions = new_classifier.predict(test_data.X)
    accuracy = new_classifier.calc_accuracy(Y=test_data.Y, predictions=predictions)
    new_classifier.calc_cost(source_data, test_data)
    print("Accuracy: ", accuracy)
    try:
        predictions.tofile(os.path.join("/home/wang/Data/android", f"{args.prediction_file}-post"), '\n')
        print('File "' + f"{args.prediction_file}-post" + '" created.')
    except:
        print('ERROR: Unable to write prediction file "' + f"{args.prediction_file}-post" + '".')

    risk = new_classifier.calc_risk(test_data.Y, predictions=predictions)
    
    print('Test risk = ' + str(risk))

    ###############################################################################
    print('\n... Saving model: "' + f"{args.model_file}-post" + '" ...')
    ###############################################################################
    try:
        with open(f"{args.model_file}-post", 'wb') as model:
            pickle.dump(classifier, model, pickle.HIGHEST_PROTOCOL)
        print('File "' + f"{args.model_file}-post" + '" created.')
    except:
        print('ERROR: Unable to write model file "' + f"{args.model_file}-post" + '".')