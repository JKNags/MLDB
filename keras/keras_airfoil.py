import os
import numpy as np
import pandas as pd
import time
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Sequential
from keras.utils import to_categorical
from keras import optimizers
from keras import initializers
from keras import backend

print("Keras - Airfoil")
file_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
	#"data/airfoil_self_noise.data"

try:
	start_time = time.time()

	#Import Dataset
	dataset = pd.read_csv(file_path, delimiter='\t')

	# Normalize Dataset
	dataset = dataset.sample(frac=1).to_numpy()

	label_max = dataset[-1].max()
	label_min = dataset[-1].min()

	def denorm_label(label_value):
		return label_value * (label_max - label_min) + label_min

	print("Label Max:%.4f,  Label Min:%.4f" % (label_max,label_min))

	for feature_num in range(0,dataset.shape[1]-1):
		dataset[:, feature_num] = (dataset[:, feature_num] - dataset[:, feature_num].mean()) / dataset[:, feature_num].std()

	# Parse Dataset
	pct_test = 5
	num_train = int(dataset.shape[0] * (100-pct_test) / 100)

	train_features = dataset[:num_train, :-1]
	train_labels = dataset[:num_train, -1]
	test_features = dataset[num_train:, :-1]
	test_labels = dataset[num_train:, -1]

	# Create model
	model_abbrev = "1x8t" # change to affect out file type
	model = Sequential()
	model.add(Dense(8, activation='tanh', input_shape=(5,),
		kernel_initializer=initializers.RandomUniform(-0.05, 0.05), bias_initializer=initializers.Constant(0.0)))
	#model.add(Dense(6, activation='tanh',
	#	kernel_initializer=initializers.RandomUniform(-0.05, 0.05), bias_initializer=initializers.Constant(0.0)))
	#model.add(Dense(8, activation='tanh',
	#	kernel_initializer=initializers.RandomUniform(-0.05, 0.05), bias_initializer=initializers.Constant(0.0)))
	model.add(Dense(1, activation='tanh',
		kernel_initializer=initializers.RandomUniform(-0.05, 0.05), bias_initializer=initializers.Constant(0.0)))

	def mse(y_true, y_pred):
		return backend.mean(backend.square(y_pred - y_true) / 2.0, axis=-1)
	def accuracy(y_true, y_pred):
		return 100.0 - (abs(denorm_label(y_true) - denorm_label(y_pred)) / denorm_label(y_true) * 100.0)

	sgd = optimizers.SGD(lr=0.1)
	model.compile(loss=mse, optimizer=sgd, metrics=[accuracy]) #mean_squared_error
	model.summary()

	# Params
	batch_size = 25
	num_epochs = 25
	shuffle = False
	verbose = 0
	#out_file_name = "%s_E%d_M%s.txt" % (datetime.now().strftime('%m%d%H%M'), num_epochs, model_abbrev)
	#out_file_name = "airfoil_E%d_M%s.txt" % (num_epochs, model_abbrev)
	#out_file = open(os.path.join("outputs", out_file_name), 'w')
	#print("File Name: %s" % out_file_name)

	
	# Test untrained model
	train_eval = model.evaluate(train_features[:], train_labels[:], 
		verbose=0, batch_size=batch_size)
	test_eval = model.evaluate(test_features[:], test_labels[:], 
		verbose=0, batch_size=batch_size)
	print("Pre-Training:	%f	%f	%f	%f" \
		% (train_eval[0], train_eval[1], test_eval[0], test_eval[1]))
	#out_file.write("%f	%f	%f	%f\n" \
	#	% (train_eval[0], train_eval[1], test_eval[0], test_eval[1]))

	# Train model
	history = model.fit(train_features[:], train_labels[:], shuffle=shuffle,
	 					epochs=num_epochs, batch_size=batch_size, verbose=verbose,
	 					validation_data=(test_features[:], test_labels[:]))

	# Print Time
	total_time = time.time() - start_time
	print("TIME: %s" % total_time)

	#print("KEYS: %s" % str(history.history.keys()))

	# Training Data
	#for epoch in range(num_epochs):
		#print("Epoch %d::  Loss: %.4f, Accuracy: %.4f" \
		#	% (epoch+1, history.history['loss'][epoch], history.history['accuracy'][epoch]))
		#out_file.write("%f	%f	%f	%f\n" \
		#	% (history.history['loss'][epoch], history.history['accuracy'][epoch],
		#		history.history['val_loss'][epoch], history.history['val_accuracy'][epoch]))

	print("Final Epoch: Loss:%.4f, Acc:%.4f, Val Loss:%.4f, Val Acc:%.4f" \
		% (history.history['loss'][-1], history.history['accuracy'][-1],
			history.history['val_loss'][-1], history.history['val_accuracy'][-1]))
	
	# Testing Data
	train_eval = model.evaluate(train_features[:], train_labels[:], 
		verbose=0, batch_size=batch_size)
	test_eval = model.evaluate(test_features[:], test_labels[:], 
		verbose=0, batch_size=batch_size)
	#out_file.write("%f	%f	%f	%f" \
	#	% (train_eval[0], train_eval[1], test_eval[0], test_eval[1]))
		
	print("Train Data:  Loss:%.4f, MSE:%.4f" % (train_eval[0],train_eval[1]))
	print("Test Data:   Loss:%.4f, MSE:%.4f" % (test_eval[0],test_eval[1]))

	"""
	for feature, label, pred in zip(test_features, denorm_labels(test_labels), denorm_labels(model.predict(test_features)[:,0])):
		print("F:%s, L:%.4f, P:%.4f, Acc:%.4f" \
			% ([],label, pred, 100.0 - abs(label-pred) / label * 100))
	"""

	#out_file.close()
except Exception as e:
	print("Exception:: %s" % e)
