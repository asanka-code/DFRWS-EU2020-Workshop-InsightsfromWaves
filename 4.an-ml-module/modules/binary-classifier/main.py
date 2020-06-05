import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from emvincelib import iq, ml, stat
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from scipy.fftpack import fft
from sklearn import preprocessing
from joblib import dump, load
from statistics import mode

# update the module name
module_name = "binary-classifier"

moduleId = 0
emTracePath = ""
resultsDirectory = ""
mlModelPath = ""

def initialize(module_id, em_trace_path, results_directory):
    print("initializing...")

    global moduleId
    global emTracePath
    global resultsDirectory
    global mlModelPath

    moduleId = int(module_id)
    emTracePath = em_trace_path
    # creating a directory to store output results of this module
    Path(results_directory + "/" + str(module_id)).mkdir(parents=True, exist_ok=True)
    resultsDirectory = results_directory  + "/" + str(module_id)
    mlModelPath = "./modules/" + str(module_name) + "/ml-model.joblib"

##############################################################################################

def getResults():
    print("generating results...")

    print("EM trace path: " + emTracePath)
    print("Results directory: " + resultsDirectory)

    clf = load(mlModelPath)
    #clf = load('/home/asanka/Documents/github/EMvidence/EMvidence/modules/binary-classifier/ml-model.joblib')
    iq.sampleRate = 32e3
    sliding_window = 0.1
    feature_vector_size = 50

    file = emTracePath
    duration = iq.getTimeDuration(file, fileType="npy")
    #ml.loadPredictingData(file, iq.sampleRate, feature_vector_size, sliding_window, duration)
    #result = ml.predictClass(clf)
    x = ml.loadPredictingData(file, iq.sampleRate, feature_vector_size, sliding_window, duration)
    result = ml.predictClass(clf, x)
    
    #--------------------------------------------------
    # textual results of the module
    #--------------------------------------------------
    f = open(resultsDirectory + "/results.txt","w+")

    f.write("Classification Result: " + str(result))
    
    f.close() 
    #--------------------------------------------------

    results = "Classification: " + str(result)
    return str(result)