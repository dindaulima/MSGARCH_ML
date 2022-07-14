setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML/")
source("getDataLuxemburg.R")
source("allfunction.R")

#inisialisasi
lossfunction = getlossfunction()
len.loss=length(lossfunction)
losstrain.SVR = matrix(nrow=8, ncol=len.loss)
losstest.SVR = matrix(nrow=8,ncol=len.loss)
colnames(losstrain.SVR) = lossfunction
colnames(losstest.SVR) = lossfunction
model.SVR = vector()

source("allfunction.R")
############################
# 1. Model ARMA-based SVR
############################
idx.svr=1
model.SVR[idx.svr] = "ARMA-based SVR"
ylabel = "return"
xlabel = "time index"
base.data = data.frame(mydata$date,mydata$return)
colnames(base.data) = c("time","rt")
head(base.data)

##### Model AR #####
# get data AR(p)
data.SVR.AR.p = makeData(data = base.data, datalag = base.data$rt, numlag = optARMAlag$ARlag, lagtype = "rt")
data.SVR.AR.p = na.omit(data.SVR.AR.p)

# fit SVR model
data = data.SVR.AR.p
head(data)
result.SVR.AR.p = fitSVR(data, startTrain, endTrain, endTest)

# plotting the prediction result
SVRresult = list()
SVRresult = result.SVR.AR.p
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste("AR-SVR Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste("AR-SVR Test"), xlabel = xlabel, ylabel=ylabel)

##### Model ARMA #####
SVRresult = list()
resitrain = resitest = resi = vector()
base.data = data.frame()

dataall = mydata$return
base.data = data.SVR.AR.p
SVRresult = result.SVR.AR.p
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)

#get data only significant lag
data.SVR.ARMA.pq = makeData(data = base.data, datalag = resi, numlag = optARMAlag$MAlag, lagtype = "at")
data.SVR.ARMA.pq = na.omit(data.SVR.ARMA.pq)
head(data.SVR.ARMA.pq)

# fit SVR model
source("allfunction.R")
data = data.SVR.ARMA.pq
head(data)
result.SVR.ARMA.pq = fitSVR(data, startTrain, endTrain, endTest)

# plot the prediction result
SVRresult = list()
SVRresult = result.SVR.ARMA.pq
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(model.SVR[idx.svr],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(model.SVR[idx.svr],"Test"), xlabel = xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}

############################
# UJI LAGRANGE MULTIPLIER
############################
source("allfunction.R")
SVRresult = list()
resitrain = resitest = resi = vector()
SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)
LMtest(resi)

source("allfunction.R")
############################
# 2. Model ARMA-GARCH-SVR
############################
idx.svr=2
model.SVR[idx.svr] = "ARMA-GARCH-based SVR"
ylabel = "return kuadrat"
xlabel = "t" 

SVRresult = list()
base.data = data.frame()
resitrain = resitest = resi = vector()

SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
at = c(resitrain,resitest)
at2 = at^2
rt2 = data.SVR.ARMA.pq$rt^2
time = data.SVR.ARMA.pq$time
base.data = data.frame(time,rt2)
head(base.data)
par(mfrow=c(1,2))

#get lag signifikan
acf.resikuadrat = acf(at2, lag.max = maxlag, type = "correlation")
acf.resikuadrat <- acf.resikuadrat$acf[2:(maxlag+1)]
pacf.resikuadrat = pacf(at2, lag.max = maxlag)
pacf.resikuadrat <- pacf.resikuadrat$acf[1:maxlag]
batas.at2 = 1.96/sqrt(length(at2)-1)

optlag = getLagSignifikan(at2, maxlag = maxlag, batas = batas.at2, alpha = alpha, na=FALSE)
data.SVR.ARCH = makeData(data = base.data, datalag = at2, numlag = optlag$PACFlag, lagtype = "at2")
data.SVR.GARCH = makeData(data = data.SVR.ARCH, datalag = rt2, numlag=optlag$ACFlag, lagtype = "rt2")
data.SVR.GARCH = na.omit(data.SVR.GARCH)
head(data.SVR.GARCH)


##### UJI Linearitas GARCH #####
SVRresult = list()
resitrain = resitest = resi = vector()

SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)
rt2 = data.SVR.ARMA.pq$rt^2
at2 = resi^2

chisq.linear = terasvirta.test(ts(rt2), lag = min(optlag$ACFlag), type = "Chisq");chisq.linear
if(chisq.linear$p.value<alpha){
  cat("Dengan Statistik uji Chisquare, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji Chisquare, Gagal Tolak H0, data linear")
}
F.linear = terasvirta.test(ts(rt2), lag = min(optlag$ACFlag), type = "F");F.linear
if(F.linear$p.value<alpha){
  cat("Dengan Statistik uji F, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji F, Gagal Tolak H0, data linear")
}
##### end of UJI Linearitas GARCH #####

# fit SVR model
data = data.SVR.GARCH
head(data)
result.SVR.GARCH = fitSVR(data, startTrain, endTrain, endTest)

# plotting the prediction result
SVRresult = list()
SVRresult = result.SVR.GARCH
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(model.SVR[idx.svr],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(model.SVR[idx.svr],"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}

############################
# UJI PERUBAHAN STRUKTUR
############################
source("allfunction.R")
head(data.SVR.GARCH)
chowtest = ujiperubahanstruktur(data.SVR.GARCH, startTrain, endTrain, endTest, alpha)


############################
# 3. MSGARCH -> sGARCH, norm
# idx = (3, 4) harus running berurutan, 
# karena proses ambil veriabel dan datanya nyambung
############################
idx.svr=3
model.SVR[idx.svr] = "MSGARCH input rt"
ylabel = "return kuadrat"
xlabel="t"

msgarch.SVR.rt = fitMSGARCH(data = dataTrain$return, TrainActual = dataTrain$rv, TestActual=dataTest$rv, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

# plotting the prediction result
SVRresult = list()
SVRresult = msgarch.SVR.rt
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(model.SVR,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(model.SVR,"Test"),xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}

##### Essential section for MSGARCH-NN process clean code #####
msgarch.model = msgarch.SVR.rt
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

# ##### get probability one step ahead #####
# tmp = matrix(ncol=2, nrow=nfore)
# tmp.prob = matrix(ncol=2, nrow=nfore)
# datatmp = dataTrain$return
# for(i in 1:nfore){
#   datatmp = c(datatmp, dataTest$return[i])
#   Ptest = State(object = msgarch.model$modelspec, par = msgarch.model$modelfit$par, data = datatmp)
#   predProb.temp = Ptest$PredProb[-1,1,]
#   tmp[i,] = predProb.temp[length(datatmp),] * voltest[i,]
#   tmp.prob[i, ] = predProb.temp[length(datatmp),]
# }
# ##### end of get probability one step ahead #####

Ptest = State(object = msgarch.model$modelspec, par = msgarch.model$modelfit$par, data = dataTest$return)
predProb.test = Ptest$PredProb[-1,1,]
vtest.pit = predProb.test * voltest
par(mfrow=c(1,1))
plot(dataTest$rv, type="l")
lines(rowSums(vtest.pit), type="l", col="blue")
# lines(rowSums(tmp), type="l", col="red")

par(mfrow=c(2,1))
plot(Ptest, type="pred")
# plot(tmp.prob[,1], type="line")
# plot(tmp.prob[,2], type="line")
##### end of Essential section for MSGARCH-NN process clean code #####

source("allfunction.R")
############################
# 4. MSGARCH-based SVR -> input rt"
############################
idx.svr=4
model.SVR[idx.svr] = "rt MSGARCH-SVR"
msgarch.model = msgarch.SVR.rt

# get variabel input ML
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1p1t","v2p2t")
par(mfrow=c(1,1))
plot(mydata$rv, type="l")
lines(rowSums(v), col="red")
# lines(c(msgarch.model$train$predict,msgarch.model$test$predict),col="green")

# form the msgarch data
time = mydata$date
rt2 = mydata$rv
base.data = data.frame(time,rt2,v)
data.SVR.MSGARCH.rt = na.omit(base.data)

# fit SVR model
data = data.SVR.MSGARCH.rt
head(data)
result.SVR.MSGARCH.rt = fitSVR(data, startTrain, endTrain, endTest)

# plotting the prediction result
SVRresult = list()
SVRresult = result.SVR.MSGARCH.rt
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(model.SVR[idx.svr],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(model.SVR[idx.svr],"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}


############################
# 5. MSGARCH -> input at
# idx = (5, 6) harus running berurutan, 
# karena proses ambil veriabel dan datanya nyambung
############################
idx.svr=5
model.SVR[idx.svr] = "MSGARCH input at"
ylabel = "return kuadrat"
xlabel="t"

SVRresult = list()
resitrain = resitest = resi = vector()

SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)

msgarch.SVR.at = fitMSGARCH(data = resitrain, TrainActual = SVRresult$train$actual^2, TestActual=SVRresult$test$actual^2, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

# plotting the prediction result
SVRresult = list()
SVRresult =  msgarch.SVR.at
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(model.SVR[idx.svr],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(model.SVR[idx.svr],"Test"),xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}

##### Essential section for MSGARCH-NN process clean code #####
msgarch.model = msgarch.SVR.at
SR.fit <- ExtractStateFit(msgarch.model$modelfit)

K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=length(resitrain), ncol=K)
voltest = matrix(nrow=length(resitest), ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = resitrain, TrainActual = SVRresult$train$actual, 
                               TestActual=SVRresult$test$actual, nfore=nfore, nstate=2)
  voltrain[,k] = msgarch.SR[[k]]$train$predict
  voltest[,k] = msgarch.SR[[k]]$test$predict
}

Ptrain = State(object = msgarch.model$modelfit)
predProb.train = Ptrain$PredProb
vtrain.pit = predProb.train[-1,1,] * voltrain
plot(dataTrain$rv, type="l")
lines(rowSums(vtrain.pit), type="l", col="blue")

Ptest = State(object = msgarch.model$modelspec, par = msgarch.model$modelfit$par, data = resitest)
predProb.test = Ptest$PredProb
vtest.pit = predProb.test[-1,1,] * voltest
plot(dataTest$rv, type="l")
lines(rowSums(vtest.pit), type="l", col="blue")
##### end of Essential section for MSGARCH-NN process clean code #####


source("allfunction.R")
############################
# 6. MSGARCH-based SVR -> input at"
############################
idx.svr=6
model.SVR[idx.svr] = "at MSGARCH-SVR"
msgarch.model = msgarch.SVR.at

# get variabel input
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1p1t","v2p2t")
par(mfrow=c(1,1))
plot(mydata$rv, type="l")
lines(rowSums(v), col="red")
lines(c(msgarch.model$train$predict,msgarch.model$test$predict),col="green")

# form the msgarch data
time = data.SVR.ARMA.pq$time
rt2 = data.SVR.ARMA.pq$rt^2
base.data = data.frame(time,rt2,v)
data.SVR.MSGARCH.at = na.omit(base.data)

# fit SVR model
data = data.SVR.MSGARCH.at
head(data)
result.SVR.MSGARCH.at = fitSVR(data, startTrain, endTrain, endTest)

# plotting the prediction result
SVRresult = list()
SVRresult = result.SVR.MSGARCH.at
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(model.SVR[idx.svr],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(model.SVR[idx.svr],"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}

##### model tambahan dengan sliding window #####
source("allfunction.R")
############################
# 7. MSGARCH-based SVR -> input rt with sliding window"
############################
idx.svr=7
model.SVR[idx.svr] = "rt MSGARCH-SVR with sliding window"
msgarch.model = msgarch.SVR.rt

#get volatility and probability each regime
SR.fit <- ExtractStateFit(msgarch.model$modelfit)
K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=length(dataTrain$return), ncol=K)
voltest = matrix(nrow=length(dataTest$return), ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = dataTrain$return, TrainActual = dataTrain$rv, 
                               TestActual=dataTest$rv, nfore=nfore, nstate=2)
  voltrain[,k] = msgarch.SR[[k]]$train$predict
  voltest[,k] = msgarch.SR[[k]]$test$predict
}

Ptrain = State(object = msgarch.model$modelfit)
predProb.train = Ptrain$PredProb[-1,1,]
vtrain.pit = predProb.train * voltrain
Ptest = State(object = msgarch.model$modelspec, par = msgarch.model$modelfit$par, data = dataTest$return)
predProb.test = Ptest$PredProb[-1,1,]
vtest.pit = predProb.test * voltest

# get variabel input
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1p1t","v2p2t")

# form the msgarch data
time = mydata$date
rt2 = mydata$rv

# get data sliding window
window_size = 5
window.data = sliding_window(x=v, y=rt2, window_size = window_size)
time = time[(window_size+1):nrow(v)]
rt2 = window.data$y
v = window.data$x
length(rt2)
dim(v)

# final msgarch data
base.data = data.frame(time,rt2,v)
data.SVR.MSGARCH.rt.window5 = na.omit(base.data)

# fit SVR model
data = data.SVR.MSGARCH.rt.window5
head(data)
result.SVR.MSGARCH.rt.window5 = fitSVR(data, startTrain, endTrain, endTest)

# plotting the prediction result
SVRresult = list()
SVRresult = result.SVR.MSGARCH.rt.window5
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(model.SVR[idx.svr],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(model.SVR[idx.svr],"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}

############################
# 8. MSGARCH-based SVR -> input at with sliding window"
############################
idx.svr=8
model.SVR[idx.svr] = "at MSGARCH-SVR with sliding window"
msgarch.model = msgarch.SVR.at

# get at
SVRresult = list()
resitrain = resitest = resi = vector()

SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)

#get volatility and probability each regime
SR.fit <- ExtractStateFit(msgarch.model$modelfit)
K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=length(resitrain), ncol=K)
voltest = matrix(nrow=length(resitest), ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = resitrain, TrainActual = SVRresult$train$actual^2, 
                               TestActual=SVRresult$test$actual^2, nfore=nfore, nstate=2)
  voltrain[,k] = msgarch.SR[[k]]$train$predict
  voltest[,k] = msgarch.SR[[k]]$test$predict
}

Ptrain = State(object = msgarch.model$modelfit)
predProb.train = Ptrain$PredProb[-1,1,]
vtrain.pit = predProb.train * voltrain
Ptest = State(object = msgarch.model$modelspec, par = msgarch.model$modelfit$par, data = resitest)
predProb.test = Ptest$PredProb[-1,1,]
vtest.pit = predProb.test * voltest

# get variabel input
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1p1t","v2p2t")

# form the msgarch data
time = data.SVR.ARMA.pq$time
rt2 = data.SVR.ARMA.pq$rt^2

# get data sliding window
window_size = 5
window.data = sliding_window(x=v, y=rt2, window_size = window_size)
time = time[(window_size+1):nrow(v)]
rt2 = window.data$y
v = window.data$x
length(rt2)
dim(v)

# final msgarch data
base.data = data.frame(time,rt2,v)
data.SVR.MSGARCH.at.window5 = na.omit(base.data)

# fit SVR model
data = data.SVR.MSGARCH.at.window5
head(data)
result.SVR.MSGARCH.at.window5 = fitSVR(data, startTrain, endTrain, endTest)

# plotting the prediction result
SVRresult = list()
SVRresult = result.SVR.MSGARCH.at.window5
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(model.SVR[idx.svr],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(model.SVR[idx.svr],"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}


############################
# PERBANDINGAN AKURASI
############################
rownames(losstrain.SVR) = model.SVR
rownames(losstest.SVR) = model.SVR

which.min(rowSums(losstrain.SVR))
ranktrain = data.frame(losstrain.SVR,sum = rowSums(losstrain.SVR), rank = rank(rowSums(losstrain.SVR)))
ranktest = data.frame(losstest.SVR,sum = rowSums(losstest.SVR), rank = rank(rowSums(losstest.SVR)))

cat("min loss in data training is",model.SVR[which.min(ranktrain$sum)])
cat("min loss in data testing is",model.SVR[which.min(ranktest$sum)])
ranktrain
ranktest


############################
# Save all data and result
############################
save(data.SVR.AR.p, data.SVR.ARMA.pq, data.SVR.ARCH, data.SVR.GARCH,data.SVR.MSGARCH.rt,data.SVR.MSGARCH.at,
     data.SVR.MSGARCH.rt.window5,data.SVR.MSGARCH.at.window5, file = "data/Datauji_SVR_tune_cge_min.RData")
save(result.SVR.AR.p, result.SVR.ARMA.pq, result.SVR.GARCH, result.SVR.MSGARCH.rt, result.SVR.MSGARCH.at, 
     result.SVR.MSGARCH.rt.window5, result.SVR.MSGARCH.at.window5, file="data/result_SVR_tune_cge_min.RData")
save(losstrain.SVR, losstest.SVR, file="data/loss_SVR_tune_cge_min.RData")
