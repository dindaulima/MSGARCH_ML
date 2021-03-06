def sliding_window(x, y, window_size):
    import numpy as np

    ytemp , xtemp = [] , []
    for i in range(len(x)-window_size):
        ytemp.append(y[i+window_size])
        xtemp.append(x[i:i+window_size+1,:])
    xnew = np.stack(xtemp)
    ynew = np.array(ytemp)
    return xnew,ynew

def lstmfit(Xtrain, ytrain, Xtest, ytest, node_hidden, epoch, allow_negative=0, window_size=1, filename="model"):
    import numpy as np
    import pandas as pd
  
    nfeature = Xtrain.shape[1]
    timestep = window_size

    if(window_size>1):
        timestep = window_size+1 # karena yang diambil data t, t-1, ..., t-5 = (windows_size + 1) data
        Xtest = np.concatenate((Xtrain[-window_size:], Xtest), axis=0)
        ytest = np.concatenate((ytrain[-window_size:], ytest), axis=0)

        Xtrain, ytrain = sliding_window(Xtrain, ytrain, window_size)
        Xtest, ytest = sliding_window(Xtest, ytest, window_size)

    # reshape input to be [samples, time steps, features] = [n, 1, nlag] -> 1 time step, nlag feature
    shapedXtrain = np.reshape(Xtrain, (Xtrain.shape[0],timestep,nfeature))
    shapedXtest = np.reshape(Xtest, (Xtest.shape[0],timestep,nfeature))
    print(shapedXtrain.shape)
    print(shapedXtest.shape)

    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.models import save_model

    tf.random.set_seed(12345)

    # create and fit the LSTM network
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)
    early_stop = keras.callbacks.EarlyStopping(monitor='loss',min_delta = 0.000000000000001, patience=90)

    model = Sequential()
    model.add(LSTM(node_hidden,input_shape=(timestep,nfeature)))
    if(allow_negative):
        model.add(Dense(1))
    else :
        model.add(Dense(1, activation="sigmoid"))
    model.compile(loss ='mse', optimizer = opt,  metrics=['mae','mse'])
    model.fit(shapedXtrain, ytrain, epochs=epoch, verbose=0, callbacks=[early_stop], validation_data=(shapedXtest, ytest))

    print(model.summary())
    #make predict data train
    trainpredict = model.predict(shapedXtrain)
    trainresult = pd.DataFrame()
    trainresult['predict'] = trainpredict[:,0]
    trainresult['actual'] = ytrain
    trainresult['idx'] = np.arange(1,len(ytrain)+1,1)

    #make predict data test
    testpredict = model.predict(shapedXtest)
    testresult = pd.DataFrame()
    testresult['predict'] = testpredict[:,0]
    testresult['actual'] = ytest
    testresult['idx'] = np.arange(1,len(ytest)+1,1)

    result = {} 
    result['train'] = trainresult
    result['test'] = testresult
    result['model'] = model

    save_model(model, filename)

    return result

def lstmpred(filename, X):
    import numpy as np
    import pandas as pd
    from tensorflow.keras.models import load_model
    model = load_model(filename)

    # reshape input to be [samples, time steps, features] = [n, 1, nlag] -> 1 time step, nlag feature
    shapedX = np.reshape(X, (X.shape[0],1,X.shape[1]))


    import tensorflow as tf

    tf.random.set_seed(12345)

    #make forecast
    forecasted = model.predict(shapedX)


    return forecasted