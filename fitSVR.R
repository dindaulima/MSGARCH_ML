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
base.data = data.frame(time=mydata$date,y=mydata$return)
head(base.data)

##### Model AR #####
# get data AR
data.SVR.AR = makeData(data = base.data, datalag = base.data$y, numlag = optARMAlag$ARlag, lagtype = "rt")
data.SVR.AR = na.omit(data.SVR.AR)

# fit SVR model
data = data.SVR.AR
head(data)
result.SVR.AR = fitSVR(data, startTrain, endTrain, endTest, is.vol=FALSE)

# get best result
SVRresult = list()
SVRresult = result.SVR.AR
data = data.SVR.AR
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = SVRresult$train
bestresult$test$actual = testactual
bestresult$test$predict = SVRresult$test

bestresult.SVR.AR = bestresult

par(mfrow=c(1,1))
# plotting the prediction result
title = "AR-SVR"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.AR
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
##### end of Model AR #####

##### Model ARMA #####
SVRresult = list()
resitrain = resitest = resi = vector()
base.data = data.frame()

dataall = mydata$return
base.data = data.SVR.AR
SVRbestresult = bestresult.SVR.AR
resitrain = SVRbestresult$train$actual - SVRbestresult$train$predict
resitest = SVRbestresult$test$actual - SVRbestresult$test$predict
resi = c(resitrain,resitest)

# get data only significant lag
data.SVR.ARMA = makeData(data = base.data, datalag = resi, numlag = optARMAlag$MAlag, lagtype = "et")
data.SVR.ARMA = na.omit(data.SVR.ARMA)

# fit SVR model
source("allfunction.R")
data = data.SVR.ARMA
head(data)
result.SVR.ARMA = fitSVR(data, startTrain, endTrain, endTest, is.vol=FALSE)

# get best result
SVRresult = list()
SVRresult = result.SVR.ARMA
data = data.SVR.ARMA
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = SVRresult$train
bestresult$test$actual = testactual
bestresult$test$predict = SVRresult$test

bestresult.SVR.ARMA = bestresult

# plotting the prediction result
title = model.SVR[idx.svr]
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}
losstrain.SVR
losstest.SVR
##### end of Model ARMA #####


##### UJI LAGRANGE MULTIPLIER #####
source("allfunction.R")
SVRbestresult = list()
resitrain = resitest = resi = vector()

SVRbestresult = bestresult.SVR.ARMA
resitrain = SVRbestresult$train$actual - SVRbestresult$train$predict
resitest = SVRbestresult$test$actual - SVRbestresult$test$predict
resi = c(resitrain,resitest)
LMtest(resi)
##### END OF UJI LAGRANGE MULTIPLIER #####



source("allfunction.R")
############################
# 2. Model GARCH-SVR
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
batas.rt2 = 1.96/sqrt(length(rt2)-1)
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
base.data = data.frame(time,y=rt2)
head(base.data)
data.SVR.ARCH = makeData(data = base.data, datalag = rt2, numlag = optlag$PACFlag, lagtype = "rt2")
data.SVR.ARCH = na.omit(data.SVR.ARCH)

# fit SVR model
source("allfunction.R")
data = data.SVR.ARCH
head(data)
result.SVR.ARCH = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# get best result
SVRresult = list()
SVRresult = result.SVR.ARCH
data = data.SVR.ARCH
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = SVRresult$train
bestresult$test$actual = testactual
bestresult$test$predict = SVRresult$test

bestresult.SVR.ARCH = bestresult

# plotting the prediction result
par(mfrow=c(1,1))
title = "ARCH-SVR"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARCH
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)
##### end of Model ARCH #####


##### Model GARCH #####
SVRbestresult = list()
base.data = data.frame()
resitrain = resitest = resi = vector()

SVRbestresult = bestresult.SVR.ARCH
resitrain = SVRbestresult$train$actual - SVRbestresult$train$predict
resitest = SVRbestresult$test$actual - SVRbestresult$test$predict
resi = c(resitrain,resitest)
ut2 = resi^2

# get data GARCH
data.SVR.GARCH = makeData(data = data.SVR.ARCH, datalag = ut2, numlag=optlag$ACFlag, lagtype = "ut2")
data.SVR.GARCH = na.omit(data.SVR.GARCH)

# fit SVR model
source("allfunction.R")
data = data.SVR.GARCH
head(data)
result.SVR.GARCH = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# get best result
SVRresult = list()
SVRresult = result.SVR.GARCH
data = data.SVR.GARCH
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = SVRresult$train
bestresult$test$actual = testactual
bestresult$test$predict = SVRresult$test

bestresult.SVR.GARCH = bestresult

length(trainactual)
length(SVRresult$train)

# plotting the prediction result
title = model.SVR[idx.svr]
SVRbestresult = list()
SVRbestresult = bestresult.SVR.GARCH
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}
losstrain.SVR
losstest.SVR

source("allfunction.R")
############################
# 3. Model ARMA-GARCH-SVR
############################
idx.svr=3
model.SVR[idx.svr] = "ARMA-GARCH-SVR"
ylabel = "volatilitas"
xlabel = "t" 

SVRbestresult = list()
base.data = data.frame()
resitrain = resitest = resi = vector()

SVRbestresult = bestresult.SVR.ARMA
resitrain = SVRbestresult$train$actual - SVRbestresult$train$predict
resitest = SVRbestresult$test$actual - SVRbestresult$test$predict
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

if(length(optlag$PACFlag)==0 && length(optlag$ACFlag)==0) { #jika tidak ada lag signifikan, gunakan GARCH(1,1)
  optlag$PACFlag = c(1)
  optlag$ACFlag = c(1)
}
optlag

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
time = data.SVR.ARMA$time
base.data = data.frame(time,y=at2)
head(base.data)
data.SVR.ARMA.ARCH = makeData(data = base.data, datalag = at2, numlag = optlag$PACFlag, lagtype = "at2")
data.SVR.ARMA.ARCH = na.omit(data.SVR.ARMA.ARCH)

# fit SVR model
data = data.SVR.ARMA.ARCH
head(data)
result.SVR.ARMA.ARCH = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# get best result
SVRresult = list()
trainactual = vector()
testactual = vector()
SVRresult = result.SVR.ARMA.ARCH
data = data.SVR.ARMA

max.lag.sig = max(optlag$PACFlag)
t.all = nrow(data)
trainactual = (data$y[(max.lag.sig+1):(t.all-nfore)])^2
testactual = (data$y[(t.all-nfore+1):t.all])^2
rt.hat.train = bestresult.SVR.ARMA$train$predict[-c(1:max.lag.sig)]
rt.hat.test = bestresult.SVR.ARMA$test$predict

length(rt.hat.train)
length(trainactual)
length(result.SVR.ARMA.ARCH$train)

trainpred = (rt.hat.train + sqrt(SVRresult$train))^2
testpred = (rt.hat.test + sqrt(SVRresult$test))^2

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = trainpred
bestresult$test$actual = testactual
bestresult$test$predict = testpred

bestresult.SVR.ARMA.ARCH = bestresult

par(mfrow=c(1,1))
# plotting the prediction result
title = "ARMA-ARCH-SVR"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA.ARCH
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)
##### end of Model ARCH #####

##### Model GARCH #####
SVRbestresult = list()
base.data = data.frame()
resitrain = resitest = resi = vector()

SVRbestresult = bestresult.SVR.ARMA.ARCH
resitrain = SVRbestresult$train$actual - SVRbestresult$train$predict
resitest = SVRbestresult$test$actual - SVRbestresult$test$predict
resi = c(resitrain,resitest)
ut2 = resi^2

# get data GARCH
data.SVR.ARMA.GARCH = makeData(data = data.SVR.ARMA.ARCH, datalag = ut2, numlag=optlag$ACFlag, lagtype = "ut2")
data.SVR.ARMA.GARCH = na.omit(data.SVR.ARMA.GARCH)

# fit SVR model
data = data.SVR.ARMA.GARCH
head(data)
result.SVR.ARMA.GARCH = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# get best result
SVRresult = list()
trainactual = vector()
testactual = vector()
SVRresult = result.SVR.ARMA.GARCH
data = data.SVR.ARMA

max.lag.sig = max(optlag$PACFlag)+max(optlag$ACFlag)
t.all = nrow(data)
trainactual = (data$y[(max.lag.sig+1):(t.all-nfore)])^2
testactual = (data$y[(t.all-nfore+1):t.all])^2
rt.hat.train = bestresult.SVR.ARMA$train$predict[-c(1:max.lag.sig)]
rt.hat.test = bestresult.SVR.ARMA$test$predict

length(rt.hat.train)
length(trainactual)
length(result.SVR.ARMA.GARCH$train)

trainpred = (rt.hat.train + sqrt(SVRresult$train))^2
testpred = (rt.hat.test + sqrt(SVRresult$test))^2

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = trainpred
bestresult$test$actual = testactual
bestresult$test$predict = testpred

bestresult.SVR.ARMA.GARCH = bestresult

# plotting the prediction result
title = model.SVR[idx.svr]
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA.GARCH
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}
losstrain.SVR
losstest.SVR

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
model.SVR[idx.svr] = "MSGARCH"
ylabel = "volatilitas"
xlabel="t"

result.SVR.MSGARCH = fitMSGARCH(data = dataTrain$return, TrainActual = dataTrain$return^2, TestActual=dataTest$return^2, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

# get best result
SVRresult = list()
trainactual = testactual = vector()
SVRresult = result.SVR.MSGARCH
data = mydata
t.all = nrow(data)
trainactual = dataTrain$return^2
testactual = dataTest$return^2

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = result.SVR.MSGARCH$train
bestresult$test$actual = testactual
bestresult$test$predict = result.SVR.MSGARCH$test

bestresult.SVR.MSGARCH = bestresult

# plotting the prediction result
par(mfrow=c(1,1))
title = model.SVR[idx.svr]
SVRbestresult = list()
SVRbestresult = bestresult.SVR.MSGARCH
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"),xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}
losstrain.SVR
losstest.SVR

source("allfunction.R")
############################
# 5. MSGARCH-based SVR -> input rt"
############################
idx.svr=5
model.SVR[idx.svr] = "MSGARCH-SVR"
msgarch.model = result.SVR.MSGARCH

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
base.data = data.frame(time,y=rt2,v)
data.SVR.MSGARCH.SVR = na.omit(base.data)
source("allfunction.R")

# fit SVR model
data = data.SVR.MSGARCH.SVR
head(data)
result.SVR.MSGARCH.SVR = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# get best result
SVRresult = list()
SVRresult = result.SVR.MSGARCH.SVR
data = data.SVR.MSGARCH.SVR
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]

length(trainactual)
length(result.SVR.MSGARCH.SVR$train)
# max(trainactual)
# max(result.SVR.MSGARCH.SVR$train)
bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = SVRresult$train
bestresult$test$actual = testactual
bestresult$test$predict = SVRresult$test

bestresult.SVR.MSGARCH.SVR = bestresult

# plotting the prediction result
title = model.SVR[idx.svr]
SVRbestresult = list()
SVRbestresult = bestresult.SVR.MSGARCH.SVR
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}
losstrain.SVR
losstest.SVR

############################
# 6. MSGARCH -> input at
# idx = (6, 7) harus running berurutan, 
# karena proses ambil variabel dan datanya nyambung
############################
idx.svr=6
model.SVR[idx.svr] = "ARMA-MSGARCH"
ylabel = "volatilitas"
xlabel="t"

# get data at
SVRbestresult = list()
resitrain = resitest = resi = vector()

SVRbestresult = bestresult.SVR.ARMA
resitrain = SVRbestresult$train$actual - SVRbestresult$train$predict
resitest = SVRbestresult$test$actual - SVRbestresult$test$predict
resi = c(resitrain,resitest)

result.SVR.ARMA.MSGARCH = fitMSGARCH(data = resitrain, TrainActual = resitrain^2, TestActual=resitest^2, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

# get best result
SVRresult = list()
trainactual = vector()
testactual = vector()
SVRresult = result.SVR.ARMA.MSGARCH
data = data.SVR.ARMA

t.all = nrow(data)
trainactual = bestresult.SVR.ARMA$train$actual^2
testactual = bestresult.SVR.ARMA$test$actual^2
rt.hat.train = bestresult.SVR.ARMA$train$predict
rt.hat.test = bestresult.SVR.ARMA$test$predict

length(rt.hat.train)
length(trainactual)
length(SVRresult$train)

trainpred = (rt.hat.train + sqrt(SVRresult$train))^2
testpred = (rt.hat.test + sqrt(SVRresult$test))^2

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = trainpred
bestresult$test$actual = testactual
bestresult$test$predict = testpred

bestresult.SVR.ARMA.MSGARCH = bestresult

# plotting the prediction result
title = model.SVR[idx.svr]
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA.MSGARCH
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"),xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}
losstrain.SVR
losstest.SVR

source("allfunction.R")
############################
# 7. MSGARCH-based SVR -> input at"
############################
idx.svr=7
model.SVR[idx.svr] = "ARMA-MSGARCH-SVR"
msgarch.model = result.SVR.ARMA.MSGARCH

# get data at
SVRbestresult = list()
resitrain = resitest = resi = vector()

SVRbestresult = bestresult.SVR.ARMA
resitrain = SVRbestresult$train$actual - SVRbestresult$train$predict
resitest = SVRbestresult$test$actual - SVRbestresult$test$predict
resi = c(resitrain,resitest)

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


# get variabel input
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1p1t","v2p2t")
par(mfrow=c(1,1))
plot(resi^2, type="l")
lines(rowSums(v), col="red")
# lines(c(msgarch.model$train$predict,msgarch.model$test$predict),col="green")

# form the msgarch data
# get data at
SVRbestresult = list()
resitrain = resitest = resi = vector()

SVRbestresult = bestresult.SVR.ARMA
resitrain = SVRbestresult$train$actual - SVRbestresult$train$predict
resitest = SVRbestresult$test$actual - SVRbestresult$test$predict
resi = c(resitrain,resitest)
# plot(at2, type="l")
time = data.SVR.ARMA$time
at2 = resi^2
trainactual = testactual = rt.hat.train = rt.hat.test = vector()

trainactual = testactual = vector()
trainactual = bestresult.SVR.ARMA$train$actual^2
testactual = bestresult.SVR.ARMA$test$actual^2
rt.hat.train = bestresult.SVR.ARMA$train$predict
rt.hat.test = bestresult.SVR.ARMA$test$predict

base.data = data.frame(time,y=at2,v)
data.SVR.ARMA.MSGARCH.SVR = na.omit(base.data)

# fit SVR model
data = data.SVR.ARMA.MSGARCH.SVR
head(data)
result.SVR.ARMA.MSGARCH.SVR = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# get best result
SVRresult = list()

SVRresult = result.SVR.ARMA.MSGARCH.SVR
data = data.SVR.ARMA

length(rt.hat.train)
length(trainactual)
length(result.SVR.ARMA.MSGARCH.SVR$train)

trainpred = (rt.hat.train + sqrt(SVRresult$train))^2
testpred = (rt.hat.test + sqrt(SVRresult$test))^2

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = trainpred
bestresult$test$actual = testactual
bestresult$test$predict = testpred

bestresult.SVR.ARMA.MSGARCH.SVR = bestresult

# plotting the prediction result
title = model.SVR[idx.svr]
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA.MSGARCH.SVR
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}
losstrain.SVR
losstest.SVR

##### model tambahan dengan sliding window #####
source("allfunction.R")
############################
# 8. MSGARCH-based SVR -> input rt with sliding window 5"
############################
idx.svr=8
model.SVR[idx.svr] = "MSGARCH-SVR window 5"
msgarch.model = result.SVR.MSGARCH

#get volatility and probability each regime
SR.fit <- ExtractStateFit(msgarch.model$modelfit)
K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=length(dataTrain$return), ncol=K)
voltest = matrix(nrow=length(dataTest$return), ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = dataTrain$return, TrainActual = dataTrain$return^2, 
                               TestActual=dataTest$return^2, nfore=nfore, nstate=2)
  voltrain[,k] = msgarch.SR[[k]]$train
  voltest[,k] = msgarch.SR[[k]]$test
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
base.data = data.frame(time,y=rt2,v)
data.SVR.MSGARCH.SVR.window5 = na.omit(base.data)

# fit SVR model
data = data.SVR.MSGARCH.SVR.window5
head(data)
result.SVR.MSGARCH.SVR.window5 = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# get best result
SVRresult = list()
SVRresult = result.SVR.MSGARCH.SVR.window5
data = data.SVR.MSGARCH.SVR.window5
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]
length(trainactual)
length(result.SVR.MSGARCH.SVR.window5$train)

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = SVRresult$train
bestresult$test$actual = testactual
bestresult$test$predict = SVRresult$test

bestresult.SVR.MSGARCH.SVR.window5 = bestresult

# plotting the prediction result
title = model.SVR[idx.svr]
SVRbestresult = list()
SVRbestresult = bestresult.SVR.MSGARCH.SVR.window5
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}
losstrain.SVR
losstest.SVR

############################
# 9. MSGARCH-based SVR -> input at with sliding window 5"
############################
idx.svr=9
model.SVR[idx.svr] = "ARMA-MSGARCH-SVR window 5"
msgarch.model = result.SVR.ARMA.MSGARCH

# get at
SVRbestresult = list()
resitrain = resitest = resi = vector()

SVRbestresult = bestresult.SVR.ARMA
resitrain = SVRbestresult$train$actual - SVRbestresult$train$predict
resitest = SVRbestresult$test$actual - SVRbestresult$test$predict
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
  voltrain[,k] = msgarch.SR[[k]]$train
  voltest[,k] = msgarch.SR[[k]]$test
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
time = data.SVR.ARMA$time
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
base.data = data.frame(time,y=at2,v)
data.SVR.ARMA.MSGARCH.SVR.window5 = na.omit(base.data)

# fit SVR model
data = data.SVR.ARMA.MSGARCH.SVR.window5
head(data)
result.SVR.ARMA.MSGARCH.SVR.window5 = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# get best result
SVRresult = list()
trainactual = vector()
testactual = vector()
SVRresult = result.SVR.ARMA.MSGARCH.SVR.window5
data = data.SVR.ARMA

trainactual = (bestresult.SVR.ARMA$train$actual[-c(1:window_size)])^2
testactual = (bestresult.SVR.ARMA$test$actual)^2
rt.hat.train = bestresult.SVR.ARMA$train$predict[-c(1:window_size)]
rt.hat.test = bestresult.SVR.ARMA$test$predict

trainpred = (rt.hat.train + sqrt(SVRresult$train))^2
testpred = (rt.hat.test + sqrt(SVRresult$test))^2

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = trainpred
bestresult$test$actual = testactual
bestresult$test$predict = testpred

bestresult.SVR.ARMA.MSGARCH.SVR.window5 = bestresult

# plotting the prediction result
title = model.SVR[idx.svr]
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA.MSGARCH.SVR.window5
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}
losstrain.SVR
losstest.SVR

############################
# PERBANDINGAN AKURASI
############################
rownames(losstrain.SVR) = model.SVR
rownames(losstest.SVR) = model.SVR

which.min(rowSums(losstrain.SVR))
ranktrain = data.frame(losstrain.SVR, rank = rank(losstrain.SVR[,1]))
ranktest = data.frame(losstest.SVR, rank = rank(losstest.SVR[,1]))

cat("min loss in data training is",model.SVR[which.min(ranktrain$MSE)])
cat("min loss in data testing is",model.SVR[which.min(ranktest$MSE)])
ranktrain
ranktest

############################
# Save all data and result
############################
# save(data.SVR.AR, data.SVR.ARMA, data.SVR.ARCH, data.SVR.GARCH, data.SVR.ARMA.ARCH, data.SVR.ARMA.GARCH, data.SVR.MSGARCH.SVR,
#      data.SVR.MSGARCH.SVR.window5, data.SVR.ARMA.MSGARCH.SVR, data.SVR.ARMA.MSGARCH.SVR.window5,
#      file="final result/datauji_SVR_sq.RData")
# save(result.SVR.AR, result.SVR.ARMA, result.SVR.ARCH, result.SVR.GARCH, result.SVR.ARMA.ARCH, result.SVR.ARMA.GARCH,
#      result.SVR.MSGARCH, result.SVR.MSGARCH.SVR, result.SVR.MSGARCH.SVR.window5, result.SVR.ARMA.MSGARCH, result.SVR.ARMA.MSGARCH.SVR,
#      result.SVR.ARMA.MSGARCH.SVR.window5, file="final result/result_SVR_sq.RData")
# save(bestresult.SVR.AR, bestresult.SVR.ARMA, bestresult.SVR.ARCH, bestresult.SVR.GARCH, bestresult.SVR.ARMA.ARCH, 
#      bestresult.SVR.ARMA.GARCH, bestresult.SVR.MSGARCH, bestresult.SVR.MSGARCH.SVR, bestresult.SVR.MSGARCH.SVR.window5, 
#      bestresult.SVR.ARMA.MSGARCH, bestresult.SVR.ARMA.MSGARCH.SVR, bestresult.SVR.ARMA.MSGARCH.SVR.window5, 
#      file="final result/bestresult_SVR_sq.RData")
# save(losstrain.SVR, losstest.SVR, file="final result/loss_SVR_sq.RData")


