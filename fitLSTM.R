setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML/")
source("getDataLuxemburg.R")
source("allfunction.R")
source_python('LSTM_fit.py')

# source_python('LSTM_forecast.py')

##### set environtment python in R #####
# ini dijalankan sekali saja
# library(usethis)
# edit_r_environ()
# print("copy and paste C:/Users/User/AppData/Local/Programs/Python/Python39/python.exe to file .Renviron")

# repl_python()
# Sys.setenv(RETICULATE_PYTHON = "C:/Users/User/AppData/Local/Programs/Python/Python39/python.exe")
# path_to_python <- "/Users/User/AppData/Local/Programs/Python/Python39""
# use_python(path_to_python)
# use_condaenv("env_name",required=T)
# use_python(Sys.which("python"))
# py_module_available("tensorflow'")
# reticulate::py_discover_config("tensorflow")
# conda_list()
##### end of set environtment python in R #####

#inisialisasi
epoch = as.integer(100000)
node_hidden = c(1:20)
lossfunction = getlossfunction()
len.loss = length(lossfunction)
losstrain.LSTM = matrix(nrow=9, ncol=len.loss)
losstest.LSTM = matrix(nrow=9,ncol=len.loss)
colnames(losstrain.LSTM) = lossfunction
colnames(losstest.LSTM) = lossfunction
model.LSTM = vector()


############################
# 1. Model ARMA-LSTM
############################
idx.lstm=1
model.LSTM[idx.lstm] = "ARMA-LSTM"
ylabel = "return"
xlabel = "t"
base.data = data.frame(time=mydata$date,y=mydata$return)
head(base.data)

##### Model AR #####
#get data AR(p)
data.LSTM.AR = makeData(data = base.data, datalag = mydata$return, numlag = optARMAlag$ARlag, lagtype = "rt")
data.LSTM.AR = na.omit(data.LSTM.AR)

# fit LSTM model
title = "AR-LSTM"
source("allfunction.R")
source_python('LSTM_fit.py')
data = data.LSTM.AR
head(data)
result.LSTM.AR = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=1, window_size = 1, title=title)

# get best result
result = list()
result = result.LSTM.AR
data = data.LSTM.AR
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]

n.neuron = length(node_hidden)
loss = matrix(nrow=n.neuron, ncol=2)
colnames(loss) = c("MSEtrain","MSEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(testactual, testpred, method = "MSE")
}
loss
opt_idx = which.min(loss[,2]);opt_idx
bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = result[[opt_idx]]$train
bestresult$test$actual = testactual
bestresult$test$predict = result[[opt_idx]]$test

bestresult.LSTM.AR = bestresult

# plotting the prediction result
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.AR
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
##### end of Model AR #####

##### Model ARMA #####
dataall = mydata$return
base.data = data.LSTM.AR
head(base.data)

# get resi AR
LSTMbestresult = list()
resitrain = resitest = resi = vector()
LSTMbestresult = bestresult.LSTM.AR
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)

# get data only optimal lag
data.LSTM.ARMA = makeData(data = base.data, datalag = resi, numlag = optARMAlag$MAlag, lagtype = "et")
data.LSTM.ARMA = na.omit(data.LSTM.ARMA)

# fit LSTM model
title = "ARMA-LSTM"
data = data.LSTM.ARMA
head(data)
result.LSTM.ARMA = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=1, window_size = 1, title=title)

# get best result
result = list()
result = result.LSTM.ARMA
data = data.LSTM.ARMA
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]

loss = matrix(nrow=n.neuron, ncol=2)
colnames(loss) = c("MSEtrain","MSEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(testactual, testpred, method = "MSE")
}
loss
opt_idx = which.min(loss[,2]);opt_idx
bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = result[[opt_idx]]$train
bestresult$test$actual = testactual
bestresult$test$predict = result[[opt_idx]]$test

bestresult.LSTM.ARMA = bestresult

# plot the prediction result
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}
##### end of Model ARMA #####


##### UJI LAGRANGE MULTIPLIER #####
source("allfunction.R")
LSTMbestresult = list()
resitrain = resitest = resi = vector()
LSTMbestresult = bestresult.LSTM.ARMA
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)
LMtest(resi)
##### end of UJI LAGRANGE MULTIPLIER #####


############################
# 2. Model GARCH-LSTM
############################
idx.lstm=2
model.LSTM[idx.lstm] = "GARCH-LSTM"
ylabel = "volatilitas"
xlabel = "t" 

rt2 = mydata$return^2

#get lag signifikan
par(mfrow=c(1,2))
acf.resikuadrat = acf(rt2, lag.max = maxlag, type = "correlation")
acf.resikuadrat <- acf.resikuadrat$acf[2:(maxlag+1)]
pacf.resikuadrat = pacf(rt2, lag.max = maxlag)
pacf.resikuadrat <- pacf.resikuadrat$acf[1:maxlag]
batas.rt2 = 1.96/sqrt(length(rt2)-1)
optlag = getLagSignifikan(rt2, maxlag = maxlag, batas = batas.rt2, alpha = alpha, na=FALSE)


##### UJI Linearitas GARCH #####
chisq.linear = terasvirta.test(ts(rt2), lag=min(optlag$PACFlag), type = "Chisq");chisq.linear
if(chisq.linear$p.value<alpha){
  cat("Dengan Statistik uji Chisquare, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji Chisquare, Gagal Tolak H0, data linear")
}
F.linear = terasvirta.test(ts(rt2), lag=min(optlag$PACFlag), type = "F");F.linear
if(F.linear$p.value<alpha){
  cat("Dengan Statistik uji F, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji F, Gagal Tolak H0, data linear")
}
##### end of UJI Linearitas GARCH #####

##### Model ARCH #####
# get data ARCH
time = mydata$date
base.data = data.frame(time,y=rt2)
head(base.data)
data.LSTM.ARCH = makeData(data = base.data, datalag = rt2, numlag = optlag$PACFlag, lagtype = "rt2")
data.LSTM.ARCH = na.omit(data.LSTM.ARCH)

# fit LSTM model
title = "ARCH-LSTM"
data = data.LSTM.ARCH
head(data)
result.LSTM.ARCH = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=0, window_size=1, title=title)

# get best result
result = list()
trainactual = vector()
testactual = vector()

result = result.LSTM.ARCH
data = data.LSTM.ARCH
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]

loss = matrix(nrow=n.neuron, ncol=2)
colnames(loss) = c("MSEtrain","MSEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(testactual, testpred, method = "MSE")
}
loss
opt_idx = which.min(loss[,2]);opt_idx
bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = result[[opt_idx]]$train
bestresult$test$actual = testactual
bestresult$test$predict = result[[opt_idx]]$test

bestresult.LSTM.ARCH = bestresult


# plot the ptrdiction result
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARCH
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)
##### end of Model ARCH #####

# get resi ARCH
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARCH
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)
ut2 = resi^2

# get data GARCH
data.LSTM.GARCH = makeData(data = data.LSTM.ARCH, datalag = ut2, numlag=optlag$ACFlag, lagtype = "ut2")
data.LSTM.GARCH = na.omit(data.LSTM.GARCH)

# fit LSTM model
title = "GARCH-LSTM"
data = data.LSTM.GARCH
head(data)
result.LSTM.GARCH = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=0, window_size=1, title=title)

# get best result
result = list()
trainactual = vector()
testactual = vector()

result = result.LSTM.GARCH
data = data.LSTM.GARCH
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]

loss = matrix(nrow=n.neuron, ncol=2)
colnames(loss) = c("MSEtrain","MSEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(testactual, testpred, method = "MSE")
}
loss
opt_idx = which.min(loss[,2]);opt_idx
bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = result[[opt_idx]]$train
bestresult$test$actual = testactual
bestresult$test$predict = result[[opt_idx]]$test

bestresult.LSTM.GARCH = bestresult

# plot the ptrdiction result
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.GARCH
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}

############################
# 3. Model ARMA-GARCH-LSTM
############################
idx.lstm=3
model.LSTM[idx.lstm] = "ARMA-GARCH-LSTM"
ylabel = "volatilitas"
xlabel = "t" 

# get resi ARMA
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)
at = resi
at2 = resi^2

#get lag signifikan
par(mfrow=c(1,2))
acf.resikuadrat = acf(at2, lag.max = maxlag, type = "correlation")
acf.resikuadrat <- acf.resikuadrat$acf[2:(maxlag+1)]
pacf.resikuadrat = pacf(at2, lag.max = maxlag)
pacf.resikuadrat <- pacf.resikuadrat$acf[1:maxlag]
batas.at2 = 1.96/sqrt(length(at2)-1)
optlag = getLagSignifikan(at2, maxlag = maxlag, batas = batas.at2, alpha = alpha, na=FALSE)


##### UJI Linearitas GARCH #####
chisq.linear = terasvirta.test(ts(at2), lag=min(optlag$PACFlag), type = "Chisq");chisq.linear
if(chisq.linear$p.value<alpha){
  cat("Dengan Statistik uji Chisquare, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji Chisquare, Gagal Tolak H0, data linear")
}
F.linear = terasvirta.test(ts(at2), lag=min(optlag$PACFlag), type = "F");F.linear
if(F.linear$p.value<alpha){
  cat("Dengan Statistik uji F, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji F, Gagal Tolak H0, data linear")
}
##### end of UJI Linearitas GARCH #####

##### Model ARMA-ARCH #####
# get data ARMA-ARCH
time = data.LSTM.ARMA$time
base.data = data.frame(time,y=at2)
head(base.data)
data.LSTM.ARMA.ARCH = makeData(data = base.data, datalag = at2, numlag = optlag$PACFlag, lagtype = "at2")
data.LSTM.ARMA.ARCH = na.omit(data.LSTM.ARMA.ARCH)

# fit LSTM model
title = "ARMA-ARCH-LSTM"
data = data.LSTM.ARMA.ARCH
head(data)
result.LSTM.ARMA.ARCH = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=0, window_size=1, title=title)

# get best result
result = list()
trainactual = vector()
testactual = vector()
result = result.LSTM.ARMA.ARCH
data = data.LSTM.ARMA

max.lag.sig = max(optlag$PACFlag)
t.all = nrow(data)
trainactual = (data$y[(max.lag.sig+1):(t.all-nfore)])^2
testactual = (data$y[(t.all-nfore+1):t.all])^2
rt.hat.train = bestresult.LSTM.ARMA$train$predict[-c(1:max.lag.sig)]
rt.hat.test = bestresult.LSTM.ARMA$test$predict

length(rt.hat.train)
length(trainactual)
length(result.LSTM.ARMA.ARCH$`Neuron 1`$train)

loss = matrix(nrow=n.neuron, ncol=2)
colnames(loss) = c("MSEtrain","MSEtest")
for(i in 1:n.neuron){
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(testactual, testpred, method = "MSE")
}
loss
opt_idx = which.min(loss[,2]);opt_idx
bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = (rt.hat.train + sqrt(result[[opt_idx]]$train))^2
bestresult$test$actual = testactual
bestresult$test$predict = (rt.hat.test + sqrt(result[[opt_idx]]$test))^2

bestresult.LSTM.ARMA.ARCH = bestresult

# plot the ptrdiction result
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.ARCH
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)
##### end of Model ARMA-ARCH #####

##### Model ARMA-GARCH #####
# get resi ARMA-ARCH
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.ARCH
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)
ut2 = resi^2

# get data ARMA-GARCH
data.LSTM.ARMA.GARCH = makeData(data = data.LSTM.ARMA.ARCH, datalag = ut2, numlag=optlag$ACFlag, lagtype = "ut2")
data.LSTM.ARMA.GARCH = na.omit(data.LSTM.ARMA.GARCH)

# fit LSTM model
title = "ARMA-GARCH-LSTM"
data = data.LSTM.ARMA.GARCH
head(data)
result.LSTM.ARMA.GARCH = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=0, window_size=1, title=title)


# get best result
result = list()
trainactual = testactual = vector()
result = result.LSTM.ARMA.GARCH
data = data.LSTM.ARMA

max.lag.sig = max(optlag$PACFlag)+max(optlag$ACFlag)
t.all = nrow(data)
trainactual = (data$y[(max.lag.sig+1):(t.all-nfore)])^2
testactual = (data$y[(t.all-nfore+1):t.all])^2
rt.hat.train = bestresult.LSTM.ARMA$train$predict[-c(1:max.lag.sig)]
rt.hat.test = bestresult.LSTM.ARMA$test$predict

length(trainactual)
length(result.LSTM.ARMA.GARCH$`Neuron 1`$train)
length(rt.hat.train)

loss = matrix(nrow=n.neuron, ncol=2)
colnames(loss) = c("MSEtrain","MSEtest")
for(i in 1:n.neuron){
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(testactual, testpred, method = "MSE")
}
loss
opt_idx = which.min(loss[,2]);opt_idx
bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = (rt.hat.train + sqrt(result[[opt_idx]]$train))^2
bestresult$test$actual = testactual
bestresult$test$predict = (rt.hat.test + sqrt(result[[opt_idx]]$test))^2

bestresult.LSTM.ARMA.GARCH = bestresult

# plot the ptrdiction result
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.GARCH
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}

############################
# UJI PERUBAHAN STRUKTUR
############################
source("allfunction.R")
head(data.LSTM.ARMA.GARCH)
dim(data.LSTM.ARMA.GARCH)
chowtest = ujiperubahanstruktur(data.LSTM.ARMA.GARCH, startTrain, endTrain, endTest, alpha)


############################
# 4. MSGARCH -> sGARCH, norm
# i = (4, 5) harus running berurutan, 
# karena proses ambil veriabel dan datanya nyambung
############################
idx.lstm=4
model.LSTM[idx.lstm] = "MSGARCH"
result = list()
ylabel = "volatilitas"
xlabel="t"

# fit msgarch model
result.LSTM.MSGARCH = fitMSGARCH(data = dataTrain$return, TrainActual = dataTrain$return^2, TestActual=dataTest$return^2, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

# get best result
result = list()
trainactual = testactual = vector()
result = result.LSTM.MSGARCH
data = mydata
t.all = nrow(data)
trainactual = dataTrain$return^2
testactual = dataTest$return^2

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = result.LSTM.MSGARCH$train
bestresult$test$actual = testactual
bestresult$test$predict = result.LSTM.MSGARCH$test

bestresult.LSTM.MSGARCH = bestresult


# plotting the prediction result
title = model.LSTM[idx.lstm]
LSTMbestresult = list()            
LSTMbestresult = bestresult.LSTM.MSGARCH
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"),xlabel=xlabel, ylabel=ylabel)

# calculate the predition error
for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}


source("allfunction.R")
############################
# 5. MSGARCH-based LSTM -> input rt"
############################
idx.lstm=5
model.LSTM[idx.lstm] = "rt MSGARCH-LSTM"
msgarch.model = result.LSTM.MSGARCH

##### Essential section for MSGARCH-NN process clean code #####
SR.fit <- ExtractStateFit(msgarch.model$modelfit)

K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=dim(dataTrain)[1], ncol=K)
voltest = matrix(nrow=dim(dataTest)[1], ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = dataTrain$return, TrainActual = dataTrain$return^2, 
                               TestActual=dataTest$return^2, nfore, nstate=2)
  
  voltrain[,k] = msgarch.SR[[k]]$train
  voltest[,k] = msgarch.SR[[k]]$test
}

Ptrain = State(object = msgarch.model$modelfit)
predProb.train = Ptrain$PredProb[-1,1,]
vtrain.pit = predProb.train * voltrain
plot(dataTrain$return^2, type="l")
lines(rowSums(vtrain.pit), type="l", col="blue")

Ptest = State(object = msgarch.model$modelspec, par = msgarch.model$modelfit$par, data = dataTest$return)
predProb.test = Ptest$PredProb[-1,1,]
vtest.pit = predProb.test * voltest
plot(dataTest$rv, type="l")
lines(rowSums(vtest.pit), type="l", col="blue")
##### end of Essential section for MSGARCH-NN process clean code #####

# get variabel input ML
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1p1t","v2p2t")
par(mfrow=c(1,1))
plot(mydata$rv, type="l")
lines(rowSums(v), col="red")
lines(c(bestresult.LSTM.MSGARCH$train$predict,bestresult.LSTM.MSGARCH$test$predict),col="green")

# form the msgarch data
time = mydata$date
rt2 = mydata$rv
base.data = data.frame(time,y=rt2,v)
data.LSTM.MSGARCH.LSTM = na.omit(base.data)

# fit LSTM model
source("allfunction.R")
source_python('LSTM_fit.py')
title = "MSGARCH-LSTM"
data = data.LSTM.MSGARCH.LSTM
head(data)
result.LSTM.MSGARCH.LSTM = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=0, window_size=1, title=title)

# get best result
result = list()
trainactual = vector()
testactual = vector()

result = result.LSTM.MSGARCH.LSTM
data = data.LSTM.MSGARCH.LSTM
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]

loss = matrix(nrow=n.neuron, ncol=2)
colnames(loss) = c("MSEtrain","MSEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(testactual, testpred, method = "MSE")
}
loss
opt_idx = which.min(loss[,2]);opt_idx
bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = result[[opt_idx]]$train
bestresult$test$actual = testactual
bestresult$test$predict = result[[opt_idx]]$test

bestresult.LSTM.MSGARCH.LSTM = bestresult

# plotting the prediction result
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.MSGARCH.LSTM
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}



############################
# 6. MSGARCH -> input at
# i = (6, 7) harus running berurutan, 
# karena proses ambil veriabel dan datanya nyambung
############################
idx.lstm=6
model.LSTM[idx.lstm] = "ARMA-MSGARCH"
ylabel = "volatilitas"
xlabel="t"

LSTMbestresult = bestresult.LSTM.ARMA
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)

# fit msgarch model
result.LSTM.ARMA.MSGARCH = fitMSGARCH(data = resitrain, TrainActual = resitrain^2, TestActual=resitest^2, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

# get best result
result = list()
trainactual = testactual = vector()
result = result.LSTM.ARMA.MSGARCH
data = data.LSTM.ARMA

t.all = nrow(data)
trainactual = bestresult.LSTM.ARMA$train$actual^2
testactual = bestresult.LSTM.ARMA$test$actual^2
rt.hat.train = bestresult.LSTM.ARMA$train$predict
rt.hat.test = bestresult.LSTM.ARMA$test$predict

length(rt.hat.train)
length(resitrain)
length(result.LSTM.ARMA.MSGARCH$train)

trainpred = (rt.hat.train + sqrt(result$train))^2
testpred = (rt.hat.test + sqrt(result$test))^2

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = trainpred
bestresult$test$actual = testactual
bestresult$test$predict = testpred

bestresult.LSTM.ARMA.MSGARCH = bestresult

# plotting the prediction result
title = model.LSTM[idx.lstm]
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.MSGARCH
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"),xlabel=xlabel, ylabel=ylabel)

#calculate the prediction error
for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}



source("allfunction.R")
############################
# 7. MSGARCH-based LSTM -> input at"
############################
idx.lstm=7
model.LSTM[idx.lstm] = "ARMA-MSGARCH-LSTM"
msgarch.model = result.LSTM.ARMA.MSGARCH

##### Essential section for MSGARCH-NN process clean code #####
SR.fit <- ExtractStateFit(msgarch.model$modelfit)
K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=length(resitrain), ncol=K)
voltest = matrix(nrow=length(resitest), ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = resitrain, TrainActual = resitrain^2, 
                               TestActual=resitest^2, nfore=nfore, nstate=2)
  voltrain[,k] = msgarch.SR[[k]]$train
  voltest[,k] = msgarch.SR[[k]]$test
}

Ptrain = State(object = msgarch.model$modelfit)
predProb.train = Ptrain$PredProb[-1,1,] 
vtrain.pit = predProb.train * voltrain
plot(resitrain^2, type="l")
lines(rowSums(vtrain.pit), type="l", col="blue")


Ptest = State(object = msgarch.model$modelspec, par = msgarch.model$modelfit$par, data = resitest)
predProb.test = Ptest$PredProb[-1,1,] 
vtest.pit = predProb.test * voltest
plot(resitest^2, type="l")
lines(rowSums(vtest.pit), type="l", col="blue")
##### end of Essential section for MSGARCH-NN process clean code #####
#get variabel input ML
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1p1t","v2p2t")
par(mfrow=c(1,1))
plot(resi^2, type="l")
lines(rowSums(v), col="red")
# lines(c(bestresult.LSTM.ARMA.MSGARCH$train$predict,bestresult.LSTM.ARMA.MSGARCH$test$predict),col="green")

# form the msgarch data
time = data.LSTM.ARMA$time
at2 = resi^2

base.data = data.frame(time,y=at2,v)
data.LSTM.ARMA.MSGARCH.LSTM = na.omit(base.data)

# fit LSTM model
title = "ARMA-MSGARCH-LSTM"
data = data.LSTM.ARMA.MSGARCH.LSTM
head(data)
result.LSTM.ARMA.MSGARCH.LSTM  = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=0, window_size=1, title=title)

# get best result
result = list()
result = result.LSTM.ARMA.MSGARCH.LSTM

trainactual = testactual = rt.hat.train = rt.hat.test = vector()
trainactual = bestresult.LSTM.ARMA$train$actual^2
testactual = bestresult.LSTM.ARMA$test$actual^2
rt.hat.train = bestresult.LSTM.ARMA$train$predict
rt.hat.test = bestresult.LSTM.ARMA$test$predict

loss = matrix(nrow=n.neuron, ncol=2)
colnames(loss) = c("MSEtrain","MSEtest")
for(i in 1:n.neuron){
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2
  class(rt.hat.train)
  class(attestpred)
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(testactual, testpred, method = "MSE")
}
loss
opt_idx = which.min(loss[,2]);opt_idx
bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = (rt.hat.train + sqrt(result[[opt_idx]]$train))^2
bestresult$test$actual = testactual
bestresult$test$predict = (rt.hat.test + sqrt(result[[opt_idx]]$test))^2

bestresult.LSTM.ARMA.MSGARCH.LSTM = bestresult

# plotting the prediction result
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.MSGARCH.LSTM
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}

source("allfunction.R")
############################
# 8. MSGARCH-based LSTM -> input rt sliding window 5"
############################
idx.lstm=8
model.LSTM[idx.lstm] = "MSGARCH-LSTM window5"

# fit LSTM model
source("allfunction.R")
source_python('LSTM_fit.py')
title = "MSGARCH-LSTM"
data = data.LSTM.MSGARCH
head(data)
result.LSTM.MSGARCH.LSTM.window5 = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=0, window_size=5, title=title)

# get best result
result = list()
trainactual = vector()
testactual = vector()
window_size=5

result = result.LSTM.MSGARCH.LSTM.window5
data = data.LSTM.MSGARCH
t.all = nrow(data)
trainactual = data$y[(window_size+1):(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]

length(trainactual)
length(result.LSTM.MSGARCH.LSTM.window5$`Neuron 1`$train)
loss = matrix(nrow=n.neuron, ncol=2)
colnames(loss) = c("MSEtrain","MSEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(testactual, testpred, method = "MSE")
}
loss
opt_idx = which.min(loss[,2]);opt_idx
bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = result[[opt_idx]]$train
bestresult$test$actual = testactual
bestresult$test$predict = result[[opt_idx]]$test

bestresult.LSTM.MSGARCH.LSTM.window5 = bestresult

# plotting the prediction result
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.MSGARCH.LSTM.window5
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}


source("allfunction.R")
############################
# 9. MSGARCH-based LSTM -> input at window 5"
############################
idx.lstm=9
model.LSTM[idx.lstm] = "ARMA-MSGARCH-LSTM window 5"

# fit LSTM model
title = "ARMA-MSGARCH-LSTM window5"
data = data.LSTM.ARMA.MSGARCH
head(data)
result.LSTM.ARMA.MSGARCH.LSTM.window5  = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=0, window_size=5, title=title)

# get best result
result = list()
result = result.LSTM.ARMA.MSGARCH.LSTM.window5

trainactual = testactual = rt.hat.train = rt.hat.test = vector()
trainactual = (bestresult.LSTM.ARMA$train$actual[-c(1:window_size)])^2
testactual = (bestresult.LSTM.ARMA$test$actual)^2
rt.hat.train = bestresult.LSTM.ARMA$train$predict[-c(1:window_size)]
rt.hat.test = bestresult.LSTM.ARMA$test$predict

length(trainactual)
length(rt.hat.train)
length(result.LSTM.ARMA.MSGARCH.LSTM.window5$`Neuron 1`$train)

loss = matrix(nrow=n.neuron, ncol=2)
colnames(loss) = c("MSEtrain","MSEtest")
for(i in 1:n.neuron){
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2
  class(rt.hat.train)
  class(attestpred)
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(testactual, testpred, method = "MSE")
}
loss
opt_idx = which.min(loss[,2]);opt_idx
bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = (rt.hat.train + sqrt(result[[opt_idx]]$train))^2
bestresult$test$actual = testactual
bestresult$test$predict = (rt.hat.test + sqrt(result[[opt_idx]]$test))^2

bestresult.LSTM.ARMA.MSGARCH.LSTM.window5 = bestresult

# plotting the prediction result
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.MSGARCH.LSTM.window5
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}
############################
# PERBANDINGAN AKURASI
############################
rownames(losstrain.LSTM) = model.LSTM
rownames(losstest.LSTM) = model.LSTM
colnames(losstrain.LSTM) = lossfunction
colnames(losstest.LSTM) = lossfunction

which.min(rowSums(losstrain.LSTM))
ranktrain = data.frame(losstrain.LSTM,sum = rowSums(losstrain.LSTM), rank = rank(rowSums(losstrain.LSTM)))
ranktest = data.frame(losstest.LSTM,sum = rowSums(losstest.LSTM), rank = rank(rowSums(losstest.LSTM)))

cat("min loss in data training is",model.LSTM[which.min(ranktrain$sum)])
cat("min loss in data testing is",model.LSTM[which.min(ranktest$sum)])
ranktrain
ranktest

############################
# Save all data and result
############################
# save(data.LSTM.AR.p, data.LSTM.ARMA.pq, data.LSTM.ARCH, data.LSTM.GARCH, data.LSTM.ARMA.ARCH, data.LSTM.ARMA.GARCH,
#       data.LSTM.MSGARCH.rt,data.LSTM.MSGARCH.at, file = "data/Datauji_LSTM_window5.RData")
# save(result.LSTM.AR.p, result.LSTM.ARMA.pq, result.LSTM.ARCH, result.LSTM.GARCH, result.LSTM.ARMA.ARCH, result.LSTM.ARMA.GARCH,
#       result.LSTM.MSGARCH.rt, result.LSTM.MSGARCH.at, file="data/result_LSTM_window5.RData")
# save(bestresult.LSTM.AR.p, bestresult.LSTM.ARMA.pq, bestresult.LSTM.ARCH, bestresult.LSTM.GARCH, bestresult.LSTM.ARMA.ARCH, bestresult.LSTM.ARMA.GARCH,
#       bestresult.LSTM.MSGARCH.rt, bestresult.LSTM.MSGARCH.at, file="data/bestresult_LSTM_window5.RData")
# save(losstrain.LSTM, losstest.LSTM, file="data/loss_LSTM_window5.RData")

