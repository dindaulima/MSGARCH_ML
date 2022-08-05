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
node_hidden = c(1:5)
lossfunction = getlossfunction()
len.loss = length(lossfunction)
losstrain.LSTM = matrix(nrow=7, ncol=len.loss)
losstest.LSTM = matrix(nrow=7,ncol=len.loss)
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
base.data = data.frame(mydata$date,mydata$return)
colnames(base.data) = c("time","rt")
head(base.data)

##### Model AR #####
#get data AR(p)
data.LSTM.AR.p = makeData(data = base.data, datalag = mydata$return, numlag = optARMAlag$ARlag, lagtype = "rt")
data.LSTM.AR.p = na.omit(data.LSTM.AR.p)

# fit LSTM model
title = "AR-LSTM"
source("allfunction.R")
source_python('LSTM_fit.py')
data = data.LSTM.AR.p
head(data)
result.LSTM.AR.p = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=1, window_size = 1, title=title)
bestresult.LSTM.AR.p = result.LSTM.AR.p[[result.LSTM.AR.p$opt_idx]]

# plotting the prediction result
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.AR.p
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
##### end of Model AR #####

##### Model ARMA #####
dataall = mydata$return
base.data = data.LSTM.AR.p
head(base.data)

# get resi AR
LSTMbestresult = list()
resitrain = resitest = resi = vector()
LSTMbestresult = bestresult.LSTM.AR.p
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)

# get data only optimal lag
data.LSTM.ARMA.pq = makeData(data = base.data, datalag = resi, numlag = optARMAlag$MAlag, lagtype = "at")
data.LSTM.ARMA.pq = na.omit(data.LSTM.ARMA.pq)

# fit LSTM model
title = "ARMA-LSTM"
data = data.LSTM.ARMA.pq
head(data)
result.LSTM.ARMA.pq = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=1, window_size = 1, title=title)
bestresult.LSTM.ARMA.pq = result.LSTM.ARMA.pq[[result.LSTM.ARMA.pq$opt_idx]]

# plot the prediction result
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.pq
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
LSTMbestresult = bestresult.LSTM.ARMA.pq
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
base.data = data.frame(time,rt2)
head(base.data)
data.LSTM.ARCH = makeData(data = base.data, datalag = rt2, numlag = optlag$PACFlag, lagtype = "rt2")
data.LSTM.ARCH = na.omit(data.LSTM.ARCH)

# fit LSTM model
title = "ARCH-LSTM"
data = data.LSTM.ARCH
head(data)
result.LSTM.ARCH = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=0, window_size=1, title=title)
bestresult.LSTM.ARCH = result.LSTM.ARCH[[result.LSTM.ARCH$opt_idx]]


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
head(data.LSTM.GARCH)

# fit LSTM model
title = "GARCH-LSTM"
data = data.LSTM.GARCH
head(data)
result.LSTM.GARCH = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=0, window_size=1, title=title)
bestresult.LSTM.GARCH = result.LSTM.GARCH[[result.LSTM.GARCH$opt_idx]]

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
LSTMbestresult = bestresult.LSTM.ARMA.pq
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
time = data.LSTM.ARMA.pq$time
base.data = data.frame(time,at2)
head(base.data)
data.LSTM.ARMA.ARCH = makeData(data = base.data, datalag = at2, numlag = optlag$PACFlag, lagtype = "at2")
data.LSTM.ARMA.ARCH = na.omit(data.LSTM.ARMA.ARCH)

# fit LSTM model
title = "ARMA-ARCH-LSTM"
data = data.LSTM.ARMA.ARCH
head(data)
result.LSTM.ARMA.ARCH = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=0, window_size=1, title=title)
bestresult.LSTM.ARMA.ARCH = result.LSTM.ARMA.ARCH[[result.LSTM.ARMA.ARCH$opt_idx]]

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
bestresult.LSTM.ARMA.GARCH = result.LSTM.ARMA.GARCH[[result.LSTM.ARMA.GARCH$opt_idx]]

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
model.LSTM[idx.lstm] = "MSGARCH input rt"
result = list()
ylabel = "volatilitas"
xlabel="t"

# fit msgarch model
msgarch.LSTM.rt = fitMSGARCH(data = dataTrain$return, TrainActual = dataTrain$return^2, TestActual=dataTest$return^2, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

# plotting the prediction result
title = model.LSTM[idx.lstm]
LSTMbestresult = list()            
LSTMbestresult = msgarch.LSTM.rt
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
garch.model = msgarch.LSTM.rt

##### Essential section for MSGARCH-NN process clean code #####
SR.fit <- ExtractStateFit(msgarch.model$modelfit)

K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=dim(dataTrain)[1], ncol=K)
voltest = matrix(nrow=dim(dataTest)[1], ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = dataTrain$return, TrainActual = dataTrain$return^2, 
                               TestActual=dataTest$return^2, nfore, nstate=2)
  
  voltrain[,k] = msgarch.SR[[k]]$train$predict
  voltest[,k] = msgarch.SR[[k]]$test$predict
}

Ptrain = State(object = msgarch.model$modelfit)
predProb.train = Ptrain$PredProb[-1,1,]
vtrain.pit = predProb.train * voltrain
plot(dataTrain$rv, type="l")
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
lines(c(msgarch.model$train$predict,msgarch.model$test$predict),col="green")

# form the msgarch data
time = mydata$date
rt2 = mydata$rv
base.data = data.frame(time,rt2,v)
data.LSTM.MSGARCH.rt = na.omit(base.data)

# fit LSTM model
source("allfunction.R")
source_python('LSTM_fit.py')
title = "rt MSGARCH-LSTM"
data = data.LSTM.MSGARCH.rt
head(data)
result.LSTM.MSGARCH.rt = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=0, window_size=5, title=title)
bestresult.LSTM.MSGARCH.rt = result.LSTM.MSGARCH.rt[[result.LSTM.MSGARCH.rt$opt_idx]]

# plotting the prediction result
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.MSGARCH.rt
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
model.LSTM[idx.lstm] = "MSGARCH input at"
LSTMresult = list()
ylabel = "volatilitas"
xlabel="t"

LSTMbestresult = bestresult.LSTM.ARMA.pq
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)

# fit msgarch model
msgarch.at.LSTM = fitMSGARCH(data = resitrain, TrainActual = resitrain^2, TestActual=resitest^2, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

# plotting the prediction result
title = model.LSTM[idx.lstm]
LSTMbestresult = list()
LSTMbestresult = msgarch.at.LSTM
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"),xlabel=xlabel, ylabel=ylabel)

#calculate the prediction error
for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}

##### Essential section for MSGARCH-NN process clean code #####
msgarch.model = msgarch.at.LSTM
msgarch.at.LSTM$modelfit
SR.fit <- ExtractStateFit(msgarch.model$modelfit)

K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=length(resitrain), ncol=K)
voltest = matrix(nrow=length(resitest), ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = resitrain, TrainActual = resitrain^2, 
                               TestActual=resitest^2, nfore=nfore, nstate=2)
  voltrain[,k] = msgarch.SR[[k]]$train$predict
  voltest[,k] = msgarch.SR[[k]]$test$predict
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


source("allfunction.R")
############################
# 7. MSGARCH-based LSTM -> input at"
############################
idx.lstm=7
model.LSTM[idx.lstm] = "at MSGARCH-LSTM"
garch.model = msgarch.at.LSTM

#get variabel input ML
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1p1t","v2p2t")
par(mfrow=c(1,1))
plot(resi^2, type="l")
lines(rowSums(v), col="red")
# lines(c(msgarch.model$train$predict,msgarch.model$test$predict),col="green")

# form the msgarch data
time = data.LSTM.ARMA.pq$time
at2 = resi^2
base.data = data.frame(time,at2,v)
data.LSTM.MSGARCH.at = na.omit(base.data)

# fit LSTM model
title = "at MSGARCH-LSTM"
data = data.LSTM.MSGARCH.at
head(data)
result.LSTM.MSGARCH.at  = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=0, window_size=5, title=title)
bestresult.LSTM.MSGARCH.at = result.LSTM.MSGARCH.at[[result.LSTM.MSGARCH.at$opt_idx]]

# plotting the prediction result
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.MSGARCH.at
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
save(data.LSTM.AR.p, data.LSTM.ARMA.pq, data.LSTM.ARCH, data.LSTM.GARCH, data.LSTM.ARMA.ARCH, data.LSTM.ARMA.GARCH,
      data.LSTM.MSGARCH.rt,data.LSTM.MSGARCH.at, file = "data/Datauji_LSTM_window5.RData")
save(result.LSTM.AR.p, result.LSTM.ARMA.pq, result.LSTM.ARCH, result.LSTM.GARCH, result.LSTM.ARMA.ARCH, result.LSTM.ARMA.GARCH,
      result.LSTM.MSGARCH.rt, result.LSTM.MSGARCH.at, file="data/result_LSTM_window5.RData")
save(bestresult.LSTM.AR.p, bestresult.LSTM.ARMA.pq, bestresult.LSTM.ARCH, bestresult.LSTM.GARCH, bestresult.LSTM.ARMA.ARCH, bestresult.LSTM.ARMA.GARCH,
      bestresult.LSTM.MSGARCH.rt, bestresult.LSTM.MSGARCH.at, file="data/bestresult_LSTM_window5.RData")
save(losstrain.LSTM, losstest.LSTM, file="data/loss_LSTM_window5.RData")

