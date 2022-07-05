def lstmforecast(lstmmodel, X):
    import numpy as np
    import pandas as pd

    # reshape input to be [samples, time steps, features] = [n, 1, nlag] -> 1 time step, nlag feature
    shapedX = np.reshape(X, (X.shape[0],1,X.shape[1]))


    import tensorflow as tf

    tf.random.set_seed(12345)

    #make forecast
    forecasted = lstmmodel.predict(shapedX)


    return forecasted