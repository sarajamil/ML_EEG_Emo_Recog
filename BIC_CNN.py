def BIC_CNN_LOO(S,OV,vis,first,name,nFilter,nKernel,nNeuron,nLayer,batchNorm):
	#### Leave One Subject Out
	#### Bicoherence as input to CNN
	import numpy as np
	import math
	import scipy.io as sio
	import tensorflow as tf
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
	from keras.optimizers import SGD, Adam 
	from keras.layers.normalization import BatchNormalization
	import time
	import h5py
	from sklearn import preprocessing

	np.random.seed(7)

	total_time = time.time()
	# Load files
	if first:
		if OV==0:
			filename = "BICdata_S%d_Firstn.mat" % S
		else:
			filename = "BICdata_S%d_OVp5_Firstn.mat" % S
	else:
		if OV==0:
			filename = "BICdata_S%d_Lastn.mat" % S
		else:
			filename = "BICdata_S%d_OVp5_Lastn.mat" % S

	mat_contents = sio.loadmat(filename)

	Xdata = mat_contents['XBICdata']
	Ydata = mat_contents['YBICdata']

	for i in range(0,Xdata.shape[0]):
	    for j in range(0,Xdata.shape[1]):
	        for k in range(0,Xdata.shape[2]):
	            Xdata[i,j,k,:,:] = preprocessing.scale(Xdata[i,j,k,:,:])

	Xdata = Xdata.astype('float32')
	Ydata = Ydata.astype('float32')

	sublist = np.arange(Xdata.shape[0])
	vislist = vis

	batch_sz = 10
	nb_epoch = 50

	TrainAcc = np.zeros([len(sublist),len(vislist),4]) #loss, accuracy, val_loss, and val_acc
	TestAcc  = np.zeros([len(sublist),len(vislist),2]) #loss and accuracy
	# take average of loss, acc, etc of last 5 outputs of training acc

	model_hist = np.zeros([len(sublist),len(vislist),4,nb_epoch]) #save model history - loss, accuracy, val_loss, val_acc

	for nVisit in vislist:
		for nSubj in sublist:
			print("\n--- Visit %d/%d - Subject %d/%d ---" % (nVisit+1,len(vislist),nSubj+1,len(sublist)))
			TestData = Xdata[nSubj,nVisit,:,:,:]
			TestOutput = Ydata[nSubj,nVisit,:]
			tSubs = sublist[sublist!=nSubj]
			TrainData = Xdata[tSubs,nVisit,:,:,:]
			TrainData = TrainData.reshape(TrainData.shape[0]*TrainData.shape[1],TrainData.shape[2],TrainData.shape[3])
			TrainOutput = Ydata[tSubs,nVisit,:]
			TrainOutput = TrainOutput.reshape(TrainOutput.shape[0]*TrainOutput.shape[1])

			# print(TrainData.shape,TrainOutput.shape)
			# print(TestData.shape,TestOutput.shape,)

			input_size = (Xdata.shape[3],Xdata.shape[4])

			# CNN Params
			c1_nf = nFilter
			c1_lf = nKernel
			c1_pl = 2 

			c2_nf = 2*nFilter
			c2_lf = nKernel
			c2_pl = 2

			# MLP Params
			l1_n = nNeuron

			#dp1 = 0.3
			#dp2 = 0.3

			# Create CNN
			model = Sequential()

			model.add(Conv1D(c1_nf,c1_lf,input_shape=input_size,activation='relu'))
			if batchNorm:
				model.add(BatchNormalization())
			model.add(MaxPooling1D(pool_size=c1_pl))
			if nLayer==2:
				model.add(Conv1D(c2_nf,c2_lf,activation='relu'))
				if batchNorm:
					model.add(BatchNormalization())
				model.add(MaxPooling1D(pool_size=c2_pl))
			model.add(Flatten())
			model.add(Dense(l1_n,activation='relu'))
			model.add(Dense(1,activation='sigmoid'))

			model.summary() 

			# Compile model
			#opt = SGD(lr=0.01)
			#opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #Not yet tested
			model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

			# Train model
			start_time = time.time()
			model_out = model.fit(TrainData,TrainOutput,batch_size=batch_sz,epochs=nb_epoch,verbose=0,validation_data=(TestData,TestOutput))
			elapsed_time = time.time() - start_time

			print('\tTrain time = ',np.round(elapsed_time/60,3),'min')

			model_hist[nSubj,nVisit,0,:] = model_out.history['loss']
			model_hist[nSubj,nVisit,1,:] = model_out.history['acc']
			model_hist[nSubj,nVisit,2,:] = model_out.history['val_loss']
			model_hist[nSubj,nVisit,3,:] = model_out.history['val_acc']

			TrainAcc[nSubj,nVisit,0] = model_out.history['loss'][-1]
			TrainAcc[nSubj,nVisit,1] = model_out.history['acc'][-1]
			TrainAcc[nSubj,nVisit,2] = model_out.history['val_loss'][-1]
			TrainAcc[nSubj,nVisit,3] = model_out.history['val_acc'][-1]

			print('\tTrain Acc = ',np.round(100*TrainAcc[nSubj,nVisit,1],3),'%')
			print('\tVal Acc = ',np.round(100*TrainAcc[nSubj,nVisit,3],3),'%')

			# Test model
			test_out = model.evaluate(TestData,TestOutput,verbose=0)
			# Acc = np.ones(TestOutput.shape)
			# for i in range(0,TestOutput.shape[0]):
			#     Ypred = model.predict(TestData[i].reshape(1,TestData.shape[1],TestData.shape[2]))
			#     Ypred = Ypred>=0.5
			#     Acc[i] = Ypred==TestOutput[i]
			#     print(Acc[i])
			# MAcc = np.mean(Acc)
			# SAcc = np.std(Acc)

			print('\tTest Acc = ',np.round(100*test_out[1],3),'%')

			TestAcc[nSubj,nVisit,0] = test_out[0] 	#loss
			TestAcc[nSubj,nVisit,1] = test_out[1]	#acc

	savename = "%s.mat" % name
	sio.savemat(savename,{'TrainAcc':TrainAcc,'TestAcc':TestAcc,'model_hist':model_hist})

	total_elapsed_time = time.time() - total_time
	print('Total Elapsed Time = ',np.round(total_elapsed_time/60,3),'min')

	print('\n----- Visit 1 Results -----\n');
	print('mean Train Acc = ',np.round(100*np.mean(TrainAcc[:,0,1]),2),'%')
	print('mean Val Acc   = ',np.round(100*np.mean(TrainAcc[:,0,3]),2),'%')
	print('mean Test Acc  = ',np.round(100*np.mean(TestAcc[:,0,1]),2),'%')
	# print('\n----- Visit 2 Results -----\n');
	# print('mean Train Acc = ',np.round(100*np.mean(TrainAcc[:,1,1]),2),'%')
	# print('mean Val Acc   = ',np.round(100*np.mean(TrainAcc[:,1,3]),2),'%')
	# print('mean Test Acc  = ',np.round(100*np.mean(TestAcc[:,1,1]),2),'%')







def BIC_CNN_KFold(S,OV,vis,first,name,kfold,nFilter,nKernel,nNeuron,nLayer,batchNorm):
	#### Leave One Subject Out
	#### Bicoherence as input to CNN
	import numpy as np
	import math
	import scipy.io as sio
	import tensorflow as tf
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
	from keras.optimizers import SGD, Adam 
	from keras.layers.normalization import BatchNormalization
	import time
	import h5py
	from sklearn import preprocessing
	from sklearn.model_selection import KFold

	np.random.seed(7)

	total_time = time.time()
	# Load files
	if first:
		if OV==0:
			filename = "BICdata_S%d_Firstn.mat" % S
		else:
			filename = "BICdata_S%d_OVp5_Firstn.mat" % S
	else:
		if OV==0:
			filename = "BICdata_S%d_Lastn.mat" % S
		else:
			filename = "BICdata_S%d_OVp5_Lastn.mat" % S

	mat_contents = sio.loadmat(filename)

	Xdata = mat_contents['XBICdata']
	Ydata = mat_contents['YBICdata']

	for i in range(0,Xdata.shape[0]):
	    for j in range(0,Xdata.shape[1]):
	        for k in range(0,Xdata.shape[2]):
	            Xdata[i,j,k,:,:] = preprocessing.scale(Xdata[i,j,k,:,:])

	Xdata = Xdata.astype('float32')
	Ydata = Ydata.astype('float32')

	sublist = np.arange(Xdata.shape[0])
	vislist = vis

	batch_sz = 10
	nb_epoch = 50

	CVO = KFold(n_splits=kfold, shuffle=True)
	X = np.arange(116)

	TrainAcc = np.zeros([CVO.get_n_splits(X),len(vislist),4]) #loss, accuracy, val_loss, and val_acc
	TestAcc  = np.zeros([CVO.get_n_splits(X),len(vislist),2]) #loss and accuracy
	# take average of loss, acc, etc of last 5 outputs of training acc

	model_hist = np.zeros([len(sublist),len(vislist),4,nb_epoch]) #save model history - loss, accuracy, val_loss, val_acc

	for nVisit in vislist:
		foldnum = 0;
		for train_ind, test_ind in CVO.split(X):
			print("\n--- Visit %d/%d - KFold %d/%d ---" % (nVisit+1,len(vislist),foldnum+1,CVO.get_n_splits(X)))
			TestData = Xdata[test_ind,nVisit,:,:,:]
			TestData = TestData.reshape(TestData.shape[0]*TestData.shape[1],TestData.shape[2],TestData.shape[3])
			TestOutput = Ydata[test_ind,nVisit,:]
			TestOutput = TestOutput.reshape(TestOutput.shape[0]*TestOutput.shape[1])
			TrainData = Xdata[train_ind,nVisit,:,:,:]
			TrainData = TrainData.reshape(TrainData.shape[0]*TrainData.shape[1],TrainData.shape[2],TrainData.shape[3])
			TrainOutput = Ydata[train_ind,nVisit,:]
			TrainOutput = TrainOutput.reshape(TrainOutput.shape[0]*TrainOutput.shape[1])


			input_size = (Xdata.shape[3],Xdata.shape[4])

			# CNN Params
			c1_nf = nFilter
			c1_lf = nKernel
			c1_pl = 2 

			c2_nf = 2*nFilter
			c2_lf = nKernel
			c2_pl = 2

			# MLP Params
			l1_n = nNeuron

			#dp1 = 0.3
			#dp2 = 0.3

			# Create CNN
			model = Sequential()

			model.add(Conv1D(c1_nf,c1_lf,input_shape=input_size,activation='relu'))
			if batchNorm:
				model.add(BatchNormalization())
			model.add(MaxPooling1D(pool_size=c1_pl))
			if nLayer==2:
				model.add(Conv1D(c2_nf,c2_lf,activation='relu'))
				if batchNorm:
					model.add(BatchNormalization())
				model.add(MaxPooling1D(pool_size=c2_pl))
			model.add(Flatten())
			model.add(Dense(l1_n,activation='relu'))
			model.add(Dense(1,activation='sigmoid'))

			model.summary() 

			# Compile model
			#opt = SGD(lr=0.01)
			#opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #Not yet tested
			model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

			# Train model
			start_time = time.time()
			model_out = model.fit(TrainData,TrainOutput,batch_size=batch_sz,epochs=nb_epoch,verbose=0,validation_data=(TestData,TestOutput))
			elapsed_time = time.time() - start_time

			print('\tTrain time = ',np.round(elapsed_time/60,3),'min')

			model_hist[foldnum,nVisit,0,:] = model_out.history['loss']
			model_hist[foldnum,nVisit,1,:] = model_out.history['acc']
			model_hist[foldnum,nVisit,2,:] = model_out.history['val_loss']
			model_hist[foldnum,nVisit,3,:] = model_out.history['val_acc']

			TrainAcc[foldnum,nVisit,0] = model_out.history['loss'][-1]
			TrainAcc[foldnum,nVisit,1] = model_out.history['acc'][-1]
			TrainAcc[foldnum,nVisit,2] = model_out.history['val_loss'][-1]
			TrainAcc[foldnum,nVisit,3] = model_out.history['val_acc'][-1]

			print('\tTrain Acc = ',np.round(100*TrainAcc[foldnum,nVisit,1],3),'%')
			print('\tVal Acc = ',np.round(100*TrainAcc[foldnum,nVisit,3],3),'%')

			# Test model
			test_out = model.evaluate(TestData,TestOutput,verbose=0)
			# Acc = np.ones(TestOutput.shape)
			# for i in range(0,TestOutput.shape[0]):
			#     Ypred = model.predict(TestData[i].reshape(1,TestData.shape[1],TestData.shape[2]))
			#     Ypred = Ypred>=0.5
			#     Acc[i] = Ypred==TestOutput[i]
			#     print(Acc[i])
			# MAcc = np.mean(Acc)
			# SAcc = np.std(Acc)

			print('\tTest Acc = ',np.round(100*test_out[1],3),'%')

			TestAcc[foldnum,nVisit,0] = test_out[0] 	#loss
			TestAcc[foldnum,nVisit,1] = test_out[1]	#acc

			foldnum = foldnum+1

	savename = "%s.mat" % name
	sio.savemat(savename,{'TrainAcc':TrainAcc,'TestAcc':TestAcc,'model_hist':model_hist})

	total_elapsed_time = time.time() - total_time
	print('Total Elapsed Time = ',np.round(total_elapsed_time/60,3),'min')

	print('\n----- Visit 1 Results -----\n');
	print('mean Train Acc = ',np.round(100*np.mean(TrainAcc[:,0,1]),2),'%')
	print('mean Val Acc   = ',np.round(100*np.mean(TrainAcc[:,0,3]),2),'%')
	print('mean Test Acc  = ',np.round(100*np.mean(TestAcc[:,0,1]),2),'%')
	# print('\n----- Visit 2 Results -----\n');
	# print('mean Train Acc = ',np.round(100*np.mean(TrainAcc[:,1,1]),2),'%')
	# print('mean Val Acc   = ',np.round(100*np.mean(TrainAcc[:,1,3]),2),'%')
	# print('mean Test Acc  = ',np.round(100*np.mean(TestAcc[:,1,1]),2),'%')
