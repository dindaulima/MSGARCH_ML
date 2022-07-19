#https://towardsdatascience.com/radial-basis-function-rbf-kernel-the-go-to-kernel-acf0d22c798a#:~:text=The%20kernel%20equation%20can%20be,same%2C%20i.e.%20X%E2%82%81%20%3D%20X%E2%82%82.

setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML/")
source("getDataLuxemburg.R")
source("allfunction.R")

#inisialisasi
lossfunction = getlossfunction()
len.loss=length(lossfunction)
losstrain.SVR = matrix(nrow=9, ncol=len.loss)
losstest.SVR = matrix(nrow=9,ncol=len.loss)
colnames(losstrain.SVR) = lossfunction
colnames(losstest.SVR) = lossfunction
model.SVR = vector()


source("allfunction.R")
############################
# 1. Model ARMA-SVR
############################
idx.svr=1
model.SVR[idx.svr] = "ARMA-SVR"
ylabel = "return"
xlabel = "t"
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
result.SVR.AR.p = fitSVR(data, startTrain, endTrain, endTest, is.vol=FALSE)

# plotting the prediction result
title = "AR-SVR"
SVRresult = list()
SVRresult = result.SVR.AR.p
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
##### end of Model AR #####

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

# get data only significant lag
data.SVR.ARMA.pq = makeData(data = base.data, datalag = resi, numlag = optARMAlag$MAlag, lagtype = "at")
data.SVR.ARMA.pq = na.omit(data.SVR.ARMA.pq)
head(data.SVR.ARMA.pq)

# fit SVR model
source("allfunction.R")
data = data.SVR.ARMA.pq
head(data)
result.SVR.ARMA.pq = fitSVR(data, startTrain, endTrain, endTest, is.vol=FALSE)

# plotting the prediction result
title = model.SVR[idx.svr]
SVRresult = list()
SVRresult = result.SVR.ARMA.pq
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}
##### end of Model ARMA #####


##### UJI LAGRANGE MULTIPLIER #####
source("allfunction.R")
SVRresult = list()
resitrain = resitest = resi = vector()
SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)
LMtest(resi)
##### END OF UJI LAGRANGE MULTIPLIER #####



source("allfunction.R")
############################
# 2. Model ARMA-GARCH-SVR
############################
idx.svr=2
model.SVR[idx.svr] = "GARCH-SVR"
ylabel = "volatilitas"
xlabel = "t" 

rt2 = mydata$return^2

#get lag signifikan
par(mfrow=c(1,2))
acf.resikuadrat = acf(rt2, lag.max = maxlag, type = "correlation")
acf.resikuadrat <- acf.resikuadrat$acf[2:(maxlag+1)]
pacf.resikuadrat = pacf(rt2, lag.max = maxlag)
pacf.resikuadrat <- pacf.resikuadrat$acf[1:maxlag]
batas.rt2 = 1.96/sqrt(length(at2)-1)
optlag = getLagSignifikan(rt2, maxlag = maxlag, batas = batas.rt2, alpha = alpha, na=FALSE)


##### UJI Linearitas GARCH #####
chisq.linear = terasvirta.test(ts(rt2), lag = min(optlag$PACFlag), type = "Chisq");chisq.linear
if(chisq.linear$p.value<alpha){
  cat("Dengan Statistik uji Chisquare, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji Chisquare, Gagal Tolak H0, data linear")
}
F.linear = terasvirta.test(ts(rt2), lag = min(optlag$PACFlag), type = "F");F.linear
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
data.SVR.ARCH = makeData(data = base.data, datalag = rt2, numlag = optlag$PACFlag, lagtype = "rt2")
data.SVR.ARCH = na.omit(data.SVR.ARCH)

# fit SVR model
source("allfunction.R")
data = data.SVR.ARCH
head(data)
result.SVR.ARCH = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# plotting the prediction result
title = "ARCH-SVR"
SVRresult = list()
SVRresult = result.SVR.ARCH
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)
##### end of Model ARCH #####


##### Model GARCH #####
SVRresult = list()
base.data = data.frame()
resitrain = resitest = resi = vector()

SVRresult = result.SVR.ARCH
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)
ut2 = resi^2

# get data GARCH
data.SVR.GARCH = makeData(data = data.SVR.ARCH, datalag = ut2, numlag=optlag$ACFlag, lagtype = "ut2")
data.SVR.GARCH = na.omit(data.SVR.GARCH)
head(data.SVR.GARCH)

# fit SVR model
source("allfunction.R")
data = data.SVR.GARCH
head(data)
result.SVR.GARCH = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# plotting the prediction result
title = model.SVR[idx.svr]
SVRresult = list()
SVRresult = result.SVR.GARCH
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}




source("allfunction.R")
############################
# 3. Model ARMA-GARCH-SVR
############################
idx.svr=3
model.SVR[idx.svr] = "ARMA-GARCH-SVR"
ylabel = "volatilitas"
xlabel = "t" 

SVRresult = list()
base.data = data.frame()
resitrain = resitest = resi = vector()

SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
at = c(resitrain,resitest)
at2 = at^2


#get lag signifikan
par(mfrow=c(1,2))
acf.resikuadrat = acf(at2, lag.max = maxlag, type = "correlation")
acf.resikuadrat <- acf.resikuadrat$acf[2:(maxlag+1)]
pacf.resikuadrat = pacf(at2, lag.max = maxlag)
pacf.resikuadrat <- pacf.resikuadrat$acf[1:maxlag]
batas.at2 = 1.96/sqrt(length(at2)-1)
optlag = getLagSignifikan(at2, maxlag = maxlag, batas = batas.at2, alpha = alpha, na=FALSE)


##### UJI Linearitas GARCH #####
chisq.linear = terasvirta.test(ts(at2), lag = min(optlag$PACFlag), type = "Chisq");chisq.linear
if(chisq.linear$p.value<alpha){
  cat("Dengan Statistik uji Chisquare, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji Chisquare, Gagal Tolak H0, data linear")
}
F.linear = terasvirta.test(ts(at2), lag = min(optlag$PACFlag), type = "F");F.linear
if(F.linear$p.value<alpha){
  cat("Dengan Statistik uji F, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji F, Gagal Tolak H0, data linear")
}
##### end of UJI Linearitas GARCH #####


##### Model ARCH #####
# get data ARCH
time = data.SVR.ARMA.pq$time
base.data = data.frame(time,at2)
head(base.data)
data.SVR.ARMA.ARCH = makeData(data = base.data, datalag = at2, numlag = optlag$PACFlag, lagtype = "at2")
data.SVR.ARMA.ARCH = na.omit(data.SVR.ARMA.ARCH)

# fit SVR model
data = data.SVR.ARMA.ARCH
head(data)
result.SVR.ARMA.ARCH = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# plotting the prediction result
title = "ARMA-ARCH-SVR"
SVRresult = list()
SVRresult = result.SVR.ARMA.ARCH
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)
##### end of Model ARCH #####

##### Model GARCH #####
SVRresult = list()
base.data = data.frame()
resitrain = resitest = resi = vector()

SVRresult = result.SVR.ARMA.ARCH
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)
ut2 = resi^2

# get data GARCH
data.SVR.ARMA.GARCH = makeData(data = data.SVR.ARMA.ARCH, datalag = ut2, numlag=optlag$ACFlag, lagtype = "ut2")
data.SVR.ARMA.GARCH = na.omit(data.SVR.ARMA.GARCH)
head(data.SVR.ARMA.GARCH)

# fit SVR model
data = data.SVR.ARMA.GARCH
head(data)
result.SVR.ARMA.GARCH = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# plotting the prediction result
title = model.SVR[idx.svr]
SVRresult = list()
SVRresult = result.SVR.ARMA.GARCH
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}


##### UJI PERUBAHAN STRUKTUR #####
source("allfunction.R")
head(data.SVR.ARMA.GARCH)
chowtest = ujiperubahanstruktur(data.SVR.ARMA.GARCH, startTrain, endTrain, endTest, alpha)
##### end of UJI PERUBAHAN STRUKTUR #####


############################
# 4. MSGARCH -> sGARCH, norm
# idx = (4, 5) harus running berurutan, 
# karena proses ambil veriabel dan datanya nyambung
############################
idx.svr=4
model.SVR[idx.svr] = "MSGARCH input rt"
ylabel = "volatilitas"
xlabel="t"

msgarch.SVR.rt = fitMSGARCH(data = dataTrain$return, TrainActual = dataTrain$rv, TestActual=dataTest$rv, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

# plotting the prediction result
title = model.SVR[idx.svr]
SVRresult = list()
SVRresult = msgarch.SVR.rt
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(title,"Test"),xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}


source("allfunction.R")
############################
# 5. MSGARCH-based SVR -> input rt"
############################
idx.svr=5
model.SVR[idx.svr] = "rt MSGARCH-SVR"
msgarch.model = msgarch.SVR.rt

##### Essential section for MSGARCH-SVR process #####
msgarch.model = msgarch.SVR.rt
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
# plot(dataTrain$return^2, type="l")
# lines(rowSums(vtrain.pit), type="l", col="blue")

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
# plot(dataTest$return^2, type="l")
# lines(rowSums(vtest.pit), type="l", col="blue")
# lines(rowSums(tmp), type="l", col="red")

# par(mfrow=c(2,1))
# plot(Ptest, type="pred")
# plot(tmp.prob[,1], type="line")
# plot(tmp.prob[,2], type="line")
##### end of Essential section for MSGARCH-SVR process clean code #####

# get variabel input ML
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1p1t","v2p2t")
# par(mfrow=c(1,1))
# plot(mydata$return^2, type="l")
# lines(rowSums(v), col="red")
# lines(c(msgarch.model$train$predict,msgarch.model$test$predict),col="green")

# form the msgarch data
time = mydata$date
rt2 = mydata$return^2
base.data = data.frame(time,rt2,v)
data.SVR.MSGARCH.rt = na.omit(base.data)

# fit SVR model
data = data.SVR.MSGARCH.rt
head(data)
result.SVR.MSGARCH.rt = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# plotting the prediction result
title = model.SVR[idx.svr]
SVRresult = list()
SVRresult = result.SVR.MSGARCH.rt
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}


############################
# 6. MSGARCH -> input at
# idx = (6, 7) harus running berurutan, 
# karena proses ambil variabel dan datanya nyambung
############################
idx.svr=6
model.SVR[idx.svr] = "MSGARCH input at"
ylabel = "volatilitas"
xlabel="t"

# get data at
SVRresult = list()
resitrain = resitest = resi = vector()

SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)

msgarch.SVR.at = fitMSGARCH(data = resitrain, TrainActual = resitrain^2, TestActual=resitest^2, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

# plotting the prediction result
title = model.SVR[idx.svr]
SVRresult = list()
SVRresult = msgarch.SVR.at
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(title,"Test"),xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}


source("allfunction.R")
############################
# 7. MSGARCH-based SVR -> input at"
############################
idx.svr=7
model.SVR[idx.svr] = "at MSGARCH-SVR"
msgarch.model = msgarch.SVR.at

##### Essential section for MSGARCH-NN process clean code #####
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


# get variabel input
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1p1t","v2p2t")
par(mfrow=c(1,1))
plot(resi^2, type="l")
lines(rowSums(v), col="red")
# lines(c(msgarch.model$train$predict,msgarch.model$test$predict),col="green")

# form the msgarch data
# get data at
SVRresult = list()
resitrain = resitest = resi = vector()

SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)
at2 = resi^2
plot(at2, type="l")

time = data.SVR.ARMA.pq$time
base.data = data.frame(time,at2,v)
data.SVR.MSGARCH.at = na.omit(base.data)

# fit SVR model
data = data.SVR.MSGARCH.at
head(data)
result.SVR.MSGARCH.at = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# plotting the prediction result
title = model.SVR[idx.svr]
SVRresult = list()
SVRresult = result.SVR.MSGARCH.at
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}


##### model tambahan dengan sliding window #####
source("allfunction.R")
############################
# 8. MSGARCH-based SVR -> input rt with sliding window 5"
############################
idx.svr=8
model.SVR[idx.svr] = "rt MSGARCH-SVR with sliding window 5"
msgarch.model = msgarch.SVR.rt

#get volatility and probability each regime
SR.fit <- ExtractStateFit(msgarch.model$modelfit)
K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=length(dataTrain$return), ncol=K)
voltest = matrix(nrow=length(dataTest$return), ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = dataTrain$return, TrainActual = dataTrain$return^2, 
                               TestActual=dataTest$return^2, nfore=nfore, nstate=2)
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
time = mydata$date
rt2 = mydata$return^2

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
result.SVR.MSGARCH.rt.window5 = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# plotting the prediction result
title = model.SVR[idx.svr]
SVRresult = list()
SVRresult = result.SVR.MSGARCH.rt.window5
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}

############################
# 9. MSGARCH-based SVR -> input at with sliding window 5"
############################
idx.svr=9
model.SVR[idx.svr] = "at MSGARCH-SVR with sliding window 5"
msgarch.model = msgarch.SVR.at

# get at
SVRresult = list()
resitrain = resitest = resi = vector()

SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)
at2=resi^2

#get volatility and probability each regime
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
Ptest = State(object = msgarch.model$modelspec, par = msgarch.model$modelfit$par, data = resitest)
predProb.test = Ptest$PredProb[-1,1,]
vtest.pit = predProb.test * voltest

# get variabel input
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1p1t","v2p2t")

# form the msgarch data
time = data.SVR.ARMA.pq$time
at2 = resi^2

# get data sliding window
window_size = 5
window.data = sliding_window(x=v, y=at2, window_size = window_size)
time = time[(window_size+1):nrow(v)]
at2 = window.data$y
v = window.data$x
length(at2)
dim(v)

# final msgarch data
base.data = data.frame(time,at2,v)
data.SVR.MSGARCH.at.window5 = na.omit(base.data)

# fit SVR model
data = data.SVR.MSGARCH.at.window5
head(data)
result.SVR.MSGARCH.at.window5 = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# plotting the prediction result
title = model.SVR[idx.svr]
SVRresult = list()
SVRresult = result.SVR.MSGARCH.at.window5
makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

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
save(data.SVR.AR.p, data.SVR.ARMA.pq, data.SVR.ARCH, data.SVR.GARCH, data.SVR.ARMA.ARCH, data.SVR.ARMA.GARCH, data.SVR.MSGARCH.rt, 
     data.SVR.MSGARCH.at, data.SVR.MSGARCH.rt.window5,data.SVR.MSGARCH.at.window5, file = "data/Datauji_SVR_tune_cge_min_sq.RData")
save(result.SVR.AR.p, result.SVR.ARMA.pq, result.SVR.ARCH, result.SVR.GARCH, result.SVR.ARMA.ARCH, result.SVR.ARMA.GARCH, 
     result.SVR.MSGARCH.rt, result.SVR.MSGARCH.at, result.SVR.MSGARCH.rt.window5, result.SVR.MSGARCH.at.window5, 
     file="data/result_SVR_tune_cge_min_sq.RData")
save(losstrain.SVR, losstest.SVR, file="data/loss_SVR_tune_cge_min_sq.RData")
