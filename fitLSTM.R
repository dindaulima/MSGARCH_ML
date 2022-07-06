setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML/")
source("getDataLuxemburg.R")
source("allfunction.R")
source_python('LSTM_fit.py')
source_python('LSTM_forecast.py')

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

#inisialisasi
epoch = as.integer(100000)
node_hidden = c(1:5)
lossfunction = getlossfunction()
len.loss = length(lossfunction)
losstrain.LSTM = matrix(nrow=6, ncol=len.loss)
losstest.LSTM = matrix(nrow=6,ncol=len.loss)
colnames(losstrain.LSTM) = lossfunction
colnames(losstest.LSTM) = lossfunction
model.LSTM = vector()


############################
# 1. Model ARMA-based LSTM
############################
idx.lstm=1
model.LSTM[idx.lstm] = "ARMA-based LSTM"
ylabel = "return"
xlabel = "t"
base.data = data.frame(mydata$date,mydata$return)
colnames(base.data) = c("time","rt")
head(base.data)

#get data AR(p)
data.LSTM.AR.p = makeData(data = base.data, datalag = mydata$return, numlag = optARMAlag$ARlag, lagtype = "rt")
data.LSTM.AR.p = na.omit(data.LSTM.AR.p)

#template
source("allfunction.R")
data = data.LSTM.AR.p
head(data)
LSTMresult = list()
resitrain = resitest = resi = vector()
LSTMresult = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch)

LSTMbestresult = LSTMresult[[LSTMresult$opt_idx]]
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(model.LSTM[idx.lstm],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(model.LSTM[idx.lstm],"Test"), xlabel = xlabel, ylabel=ylabel)


result.LSTM.AR.p = LSTMresult
bestreslut.LSTM.AR.p = LSTMbestresult
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)
head(resi)

##### Model ARMA(p,q) #####

dataall = mydata$return
base.data = data.LSTM.AR.p
head(base.data)
#get data only optimal lag
data.LSTM.ARMA.pq = makeData(data = base.data, datalag = resi, numlag = optARMAlag$MAlag, lagtype = "at")
data.LSTM.ARMA.pq = na.omit(data.LSTM.ARMA.pq)

#template
data = data.LSTM.ARMA.pq
head(data)
LSTMresult = list()
resitrain = resitest = resi = vector()
LSTMresult = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch)

LSTMbestresult = LSTMresult[[LSTMresult$opt_idx]]
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(model.LSTM[idx.lstm],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(model.LSTM[idx.lstm],"Test"), xlabel = xlabel, ylabel=ylabel)

for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}

result.LSTM.ARMA.pq = LSTMresult
bestresult.LSTM.ARMA.pq = LSTMbestresult

############################
# UJI LAGRANGE MULTIPLIER
############################
source("allfunction.R")
LSTMbestresult = bestresult.LSTM.ARMA.pq
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)
LMtest(resi)

############################
# UJI Linearitas GARCH
############################
LSTMbestresult = bestresult.LSTM.ARMA.pq
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)
rt2 = data.LSTM.ARMA.pq$rt^2
at2 = resi^2
datalinear = data.frame(rt2,at2,lag(at2))
colnames(datalinear) = c("rt2","at2","at2lag1")
head(datalinear)
datalinear = na.omit(datalinear)
chisq.linear = terasvirta.test(datalinear$rt2, datalinear$at2lag1, type = "Chisq")
if(chisq.linear$p.value<alpha){
  cat("Dengan Statistik uji Chisquare, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji Chisquare, Gagal Tolak H0, data linear")
}
F.linear = terasvirta.test(datalinear$rt2, datalinear$at2lag1, type = "F")
if(F.linear$p.value<alpha){
  cat("Dengan Statistik uji F, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji F, Gagal Tolak H0, data linear")
}

############################
# 2. Model ARMA(p,q)-GARCH-LSTM
############################
idx.lstm=2
model.LSTM[idx.lstm] = "ARMA(p,q)-GARCH-based LSTM"
ylabel = "return kuadrat"
xlabel = "t" 
LSTMbestresult = bestresult.LSTM.ARMA.pq
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)
rt2 = data.LSTM.ARMA.pq$rt^2
at = resi
at2 = resi^2
time = data.LSTM.ARMA.pq[,1]
base.data = data.frame(time,rt2)
head(base.data)

#get lag signifikan
par(mfrow=c(1,2))
acf.resikuadrat = acf(resitrain^2, lag.max = maxlag, type = "correlation")
acf.resikuadrat <- acf.resikuadrat$acf[2:(maxlag+1)]
pacf.resikuadrat = pacf(resitrain^2, lag.max = maxlag)
pacf.resikuadrat <- pacf.resikuadrat$acf[1:maxlag]
batas.at2 = 1.96/sqrt(length(resitrain)-1)

optlag = getLagSignifikan(resitrain^2, maxlag = maxlag, batas = batas.at2, alpha = alpha, na=FALSE)
data.LSTM.ARCH = makeData(data = base.data, datalag = at2, numlag = optlag$ARlag, lagtype = "at2")
data.LSTM.GARCH = makeData(data = data.LSTM.ARCH, datalag = rt2, numlag=optlag$MAlag, lagtype = "rt2")
data.LSTM.GARCH = na.omit(data.LSTM.GARCH)

#template
data = data.LSTM.GARCH
head(data)
LSTMresult = list()
resitrain = resitest = resi = vector()
LSTMresult = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch)

LSTMbestresult = LSTMresult[[LSTMresult$opt_idx]]
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(model.LSTM[idx.lstm],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(model.LSTM[idx.lstm],"Test"), xlabel=xlabel, ylabel=ylabel)

for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}
result.LSTM.GARCH = LSTMresult
bestresult.LSTM.GARCH = LSTMbestresult

############################
# UJI PERUBAHAN STRUKTUR
############################
source("allfunction.R")
head(data.LSTM.GARCH)
dim(data.LSTM.GARCH)
chowtest = ujiperubahanstruktur(data.LSTM.GARCH, startTrain, endTrain, endTest, alpha)

# i = (3, 4) harus running berurutan, 
# karena proses ambil veriabel dan datanya nyambung
############################
# 3. MSGARCH -> sGARCH, norm
############################
idx.lstm=3
model.LSTM[idx.lstm] = "MSGARCH input rt"
result = list()
ylabel = "return kuadrat"
xlabel="t"

msgarch = fitMSGARCH(data = dataTrain$return, TrainActual = dataTrain$rv, TestActual=dataTest$rv, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

makeplot(msgarch$train$actual, msgarch$train$predict, paste(model.LSTM[idx.lstm],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(msgarch$test$actual, msgarch$test$predict, paste(model.LSTM[idx.lstm],"Test"),xlabel=xlabel, ylabel=ylabel)

#akurasi
LSTMbestresult = msgarch
for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}

msgarch.rt = msgarch

##### Essential section for MSGARCH-NN process clean code #####
msgarch.model = msgarch.rt
SR.fit <- ExtractStateFit(msgarch.model$modelfit)

K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=dim(dataTrain)[1], ncol=K)
voltest = matrix(nrow=dim(dataTest)[1], ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = dataTrain$return, TrainActual = dataTrain$rv, 
                               TestActual=dataTest$rv, nfore, nstate=2)
  
  voltrain[,k] = msgarch.SR[[k]]$train$predict
  voltest[,k] = msgarch.SR[[k]]$test$predict
}

Ptrain = State(object = msgarch.model$modelfit)
predProb.train = Ptrain$PredProb
vtrain.pit = predProb.train[-1,1,] * voltrain
plot(dataTrain$rv, type="l")
lines(rowSums(vtrain.pit), type="l", col="blue")

Ptest = State(object = msgarch.model$modelspec, par = msgarch.model$modelfit$par, data = dataTest$return)
predProb.test = Ptest$PredProb
vtest.pit = predProb.test[-1,1,] * voltest
plot(dataTest$rv, type="l")
lines(rowSums(vtest.pit), type="l", col="blue")
##### end of Essential section for MSGARCH-NN process clean code #####

source("allfunction.R")
############################
# 4. MSGARCH-based LSTM -> input rt"
############################
idx.lstm=4
model.LSTM[idx.lstm] = "rt MSGARCH-LSTM"
garch.model = msgarch.rt

#get variabel input ML
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
data.LSTM.MSGARCH.rt.pit = na.omit(base.data)

#template
data = data.LSTM.MSGARCH.rt.pit
head(data)
LSTMresult = list()
resitrain = resitest = resi = vector()
LSTMresult = fitLSTM(data, startTrain, endTrain, endTest, node_hidden, epoch)

LSTMbestresult = LSTMresult[[LSTMresult$opt_idx]]
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(model.LSTM[idx.lstm],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(model.LSTM[idx.lstm],"Test"), xlabel=xlabel, ylabel=ylabel)

for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}

result.LSTM.MSGARCH.rt.pit = LSTMresult
bestresult.LSTM.MSGARCHrt.pit = LSTMbestresult

# i = (5, 6) harus running berurutan, 
# karena proses ambil veriabel dan datanya nyambung
############################
# 5. MSGARCH -> input at
############################
idx.lstm=5
model.LSTM[idx.lstm] = "MSGARCH input at"
LSTMresult = list()
ylabel = "return kuadrat"
xlabel="t"

LSTMbestresult = bestresult.LSTM.ARMA.pq
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)

msgarch = fitMSGARCH(data = resitrain, TrainActual = LSTMbestresult$train$actual^2, TestActual=LSTMbestresult$test$actual^2, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

makeplot(msgarch$train$actual, msgarch$train$predict, paste(model.LSTM[idx.lstm],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(msgarch$test$actual, msgarch$test$predict, paste(model.LSTM[idx.lstm],"Test"),xlabel=xlabel, ylabel=ylabel)


#akurasi
LSTMbestresult = msgarch
for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}


msgarch.at = msgarch
##### Essential section for MSGARCH-NN process clean code #####
msgarch.model = msgarch.at
SR.fit <- ExtractStateFit(msgarch.model$modelfit)

K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=length(resitrain), ncol=K)
voltest = matrix(nrow=length(resitest), ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = resitrain, TrainActual = LSTMbestresult$train$actual^2, 
                               TestActual=LSTMbestresult$test$actual^2, nfore=nfore, nstate=2)
  voltrain[,k] = msgarch.SR[[k]]$train$predict
  voltest[,k] = msgarch.SR[[k]]$test$predict
}

Ptrain = State(object = msgarch.model$modelfit)
predProb.train = Ptrain$PredProb
vtrain.pit = predProb.train[-1,1,] * voltrain
plot(dataTrain$rv, type="l")
lines(rowSums(vtrain.pit), type="l", col="blue")


Ptest = State(object = msgarch.model$modelspec, par = msgarch.model$modelfit$par, data = dataTest$return)
predProb.test = Ptest$PredProb
vtest.pit = predProb.test[-1,1,] * voltest
plot(dataTest$rv, type="l")
lines(rowSums(vtest.pit), type="l", col="blue")
##### end of Essential section for MSGARCH-NN process clean code #####


source("allfunction.R")
############################
# 6. MSGARCH-based LSTM -> input at"
############################
idx.lstm=6
model.LSTM[idx.lstm] = "at MSGARCH-LSTM"
garch.model = msgarch.at

#get variabel input ML
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1pit","v2pit")
par(mfrow=c(1,1))
plot(mydata$rv, type="l")
lines(rowSums(v), col="red")
lines(c(msgarch.model$train$predict,msgarch.model$test$predict),col="green")

# form the msgarch data
time = data.LSTM.ARMA.pq$time
rt2 = data.LSTM.ARMA.pq$rt^2
base.data = data.frame(time,rt2,v)
data.LSTM.MSGARCH.at.pit = na.omit(base.data)

#template
data = data.LSTM.MSGARCH.at.pit
head(data)
LSTMresult = list()
LSTMresult = fitLSTM(data, startTrain, endTrain, endTest, layer_hidden, node_hidden, epoch)

LSTMbestresult = LSTMresult[[LSTMresult$opt_idx]]
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(model.LSTM[idx.lstm],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(model.LSTM[idx.lstm],"Test"), xlabel = xlabel, ylabel=ylabel)

for(j in 1:len.loss){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}
result.LSTM.MSGARCH.at.pit = LSTMresult
bestresult.LSTM.MSGARCH.at.pit = LSTMbestresult

############################
# PERBANDINGAN AKURASI
############################
msename
MSEtrain = MSEtrain.LSTM
MSEtest = MSEtest.LSTM
allMSE = data.frame(MSEtrain,MSEtest)
ranktrain = rank(allMSE$MSEtrain)
ranktest = rank(allMSE$MSEtest)
allMSE = data.frame(MSEtrain, ranktrain, MSEtest, ranktest)
rownames(allMSE) = msename
allMSE

which.min(allMSE$MSEtrain)
which.min(allMSE$MSEtest)

allMSE

MSErank = cbind(msename,rank(allMSE$MSEtrain))
MSErank

a = MSEtrain[3]
b = MSEtrain[4]
pct = (a-b)/a
pct

##### result #####
