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

print("Keras - Iris")
file_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	#"data/iris.data"

try:
	start_time = time.time()

	# Import Dataset
	dataset = pd.read_csv(file_path)
	dataset = dataset.sample(frac=1).to_numpy()

	# Normalize Dataset
	for feature_num in range(0,dataset.shape[1]-1):
		dataset[:, feature_num] = (dataset[:, feature_num] - dataset[:, feature_num].mean()) / dataset[:, feature_num].std()

	labels = to_categorical(pd.Categorical(dataset[:,-1]).codes, 3)

	# Parse Dataset
	pct_test = 15
	num_train = int(dataset.shape[0] * (100-pct_test) / 100)

	train_features = dataset[:num_train, :4]
	train_labels = labels[:num_train]
	test_features = dataset[num_train:, :4]
	test_labels = labels[num_train:]

	# Create model
	model_abbrev = "1x8t" # change to affect out file type
	model = Sequential()
	model.add(Dense(8, activation='tanh', input_shape=(train_features.shape[1],),
		kernel_initializer=initializers.RandomUniform(-0.05, 0.05), bias_initializer=initializers.Constant(0.0)))
	#model.add(Dense(8, activation='tanh',
	#	kernel_initializer=initializers.RandomUniform(-0.05, 0.05), bias_initializer=initializers.Constant(0.0)))
	#model.add(Dense(8, activation='tanh',
	#	kernel_initializer=initializers.RandomUniform(-0.05, 0.05), bias_initializer=initializers.Constant(0.0)))
	model.add(Dense(3, activation='softmax',
		kernel_initializer=initializers.RandomUniform(-0.05, 0.05), bias_initializer=initializers.Constant(0.0)))

	sgd = optimizers.SGD(lr=0.1)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	model.summary()

	# Params
	batch_size = 25
	num_epochs = 100
	shuffle = False
	verbose = 0

	# Test untrained model
	train_eval = model.evaluate(train_features[:], train_labels[:], 
		verbose=0, batch_size=batch_size)
	test_eval = model.evaluate(test_features[:], test_labels[:], 
		verbose=0, batch_size=batch_size)
	print("Pre-Training:	%f	%f	%f	%f" \
		% (train_eval[0], train_eval[1], test_eval[0], test_eval[1]))

	# Train model
	history = model.fit(train_features[:], train_labels[:], shuffle=shuffle,
	 					epochs=num_epochs, batch_size=batch_size, verbose=verbose,
	 					validation_data=(test_features[:], test_labels[:]))

	# Print Time
	total_time = time.time() - start_time
	print("TIME: %s" % total_time)

	#out_file_name = "%s_E%d_M%s.txt" % (datetime.now().strftime('%m%d%H%M'), num_epochs, model_abbrev)
	out_file_name = "iris_E%d_M%s.txt" % (num_epochs, model_abbrev)
	out_file = open(os.path.join("outputs", out_file_name), 'w')
	print("File Name: %s" % out_file_name)

	# Training Data
	for epoch in range(num_epochs):
		#print("Epoch %d::  Loss: %.4f, Accuracy: %.4f" \
		#	% (epoch+1, history.history['loss'][epoch], history.history['accuracy'][epoch]))
		out_file.write("%f	%f	%f	%f\n" \
			% (history.history['loss'][epoch], history.history['accuracy'][epoch],
				history.history['val_loss'][epoch], history.history['val_accuracy'][epoch]))

	print("Final Epoch: Loss:%.4f, Acc:%.4f" % (history.history['loss'][-1], history.history['accuracy'][-1]))
	
	# Testing Data
	train_eval = model.evaluate(train_features[:], train_labels[:], 
		verbose=verbose, batch_size=batch_size)
	test_eval = model.evaluate(test_features[:], test_labels[:], 
		verbose=verbose, batch_size=batch_size)
	out_file.write("%f	%f	%f	%f" \
		% (train_eval[0], train_eval[1], test_eval[0], test_eval[1]))
		
	print("Train Data:  Loss:%.4f, Acc:%.4f" % (train_eval[0],train_eval[1]))
	print("Test Data:   Loss:%.4f, Acc:%.4f" % (test_eval[0],test_eval[1]))

	#for feature, label, pred in zip(test_features, test_labels, model.predict(test_features)):
	#	print("F:%s, L:%s, P:%s, result:%s" \
	#		% ([],np.argmax(label), np.argmax(pred), "RIGHT" if np.argmax(label) == np.argmax(pred) else "WRONG"))

	out_file.close()
	
except Exception as e:
	print("Exception:: %s" % e)
