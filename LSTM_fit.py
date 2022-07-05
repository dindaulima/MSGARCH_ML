def lstm(Xtrain, ytrain, Xtest, ytest, node_hidden, epoch=10000):
    import numpy as np
    import pandas as pd

    # reshape input to be [samples, time steps, features] = [n, 1, nlag] -> 1 time step, nlag feature
    shapedXtrain = np.reshape(Xtrain, (Xtrain.shape[0],1,Xtrain.shape[1]))
    shapedXtest = np.reshape(Xtest, (Xtest.shape[0],1,Xtest.shape[1]))

    nfeature = Xtrain.shape[1]
    timestep = 1

    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from numpy import array
    from keras import callbacks

    tf.random.set_seed(12345)

    # create and fit the LSTM network
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
    early_stop = keras.callbacks.EarlyStopping(monitor='loss',min_delta = 0.000000000000001, patience=90)

    model = Sequential()
    model.add(LSTM(node_hidden,input_shape=(timestep,nfeature)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss ='mse', optimizer = opt,  metrics=['mae','mse'])
    model.fit(shapedXtrain, ytrain, epochs=epoch, verbose=0, callbacks=[early_stop], validation_data=(shapedXtest, ytest))

    print(model.summary())
    #make predict data train
    trainpredict = model.predict(shapedXtrain)
    print(trainpredict.shape)
    trainresult = pd.DataFrame()
    trainresult['predict'] = np.nan
    trainresult['predict'] = trainpredict[:,0]
    trainresult['actual'] = ytrain
    trainresult['idx'] = np.arange(1,len(ytrain)+1,1)

    #make predict data test
    testpredict = model.predict(shapedXtest)

    testresult = pd.DataFrame()
    testresult['predict'] = np.nan
    testresult['predict'] = testpredict[:,0]
    testresult['actual'] = ytest
    testresult['idx'] = np.arange(1,len(ytest)+1,1)

    result = {} 
    result['train'] = trainresult
    result['test'] = testresult
    result['model'] = model


    return result