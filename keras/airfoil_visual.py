import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from keras.datasets import mnist

print("Keras Start")
file_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
	#"data/airfoil_self_noise.data"

try:
    """
    1. Frequency, in Hertzs.
    2. Angle of attack, in degrees.
    3. Chord length, in meters.
    4. Free-stream velocity, in meters per second.
    5. Suction side displacement thickness, in meters.

    The only output is:
    6. Scaled sound pressure level, in decibels. 
    """
    names = ['frequency','angle','chord_length','velocity','thickness',     'pressure']

    #Import Dataset
    dataset = pd.read_csv(file_path, names=names, delimiter='\t').to_numpy()

    print(dataset)

    #plt.scatter(dataset[:,-1], dataset[:,1], marker='^', c=(1,0,0))
    #plt.show()
    

    fig = plt.figure()
    for i in range(5):
        plt.subplot(3,2,i+1)
        plt.tight_layout()
        #plt.imshow(x_train[i], interpolation='none')
        plt.scatter(dataset[:,-1], dataset[:,i], marker='^', c=[1,0,0])
        plt.title(names[i])
        #plt.xticks([])
        #plt.yticks([])
    plt.show()
    
    
    fig = plt.figure()
    num_bins=8
    bins = plt.hist(dataset[:,-1], bins=num_bins)
    for i in range(num_bins):
        plt.text(bins[1][i]+1.2,bins[0][i],str(int(bins[0][i])))
    plt.ylabel('Occurances')
    plt.xlabel('Pressure (dB)')
    plt.title('Airfoil Dataset')
    plt.show()
except Exception as e:
    print("Exception:: %s" % e)
