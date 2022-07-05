setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML/")
source("getDataLuxemburg.R")
source("allfunction.R")

#inisialisasi
lossfunction = getlossfunction()
len.loss=length(lossfunction)
losstrain.SVR = matrix(nrow=6, ncol=len.loss)
losstest.SVR = matrix(nrow=6,ncol=len.loss)
colnames(losstrain.SVR) = lossfunction
colnames(losstest.SVR) = lossfunction
model.SVR = vector()

source("allfunction.R")
############################
# 1. Model ARMA-based SVR
############################
idx.svr=1
model.SVR[idx.svr] = "ARMA(p,q)-based SVR"
ylabel = "return"
xlabel = "time index"
base.data = data.frame(mydata$date,mydata$return)
colnames(base.data) = c("time","rt")
head(base.data)

#get data AR(p)
data.SVR.AR.p = makeData(data = base.data, datalag = base.data$rt, numlag = optARMAlag$ARlag, lagtype = "rt")
data.SVR.AR.p = na.omit(data.SVR.AR.p)

#template
data = data.SVR.AR.p
head(data)
SVRresult = list()
resitrain = resitest = resi = vector()
SVRresult = fitSVR(data, startTrain, endTrain, endTest)

makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(model.SVR[idx.svr],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(model.SVR[idx.svr],"Test"), xlabel = xlabel, ylabel=ylabel)

result.SVR.AR.p = SVRresult

##### Model ARMA(p,q) #####
SVRresult = list()
resitrain = resitest = resi = vector()
base.data = data.frame()

dataall = mydata$return
base.data = data.SVR.AR.p
SVRresult = result.SVR.AR.p
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)
head(resi)

#get data only significant lag
data.SVR.ARMA.pq = makeData(data = base.data, datalag = resi, numlag = optlag$MAlag, lagtype = "at")
data.SVR.ARMA.pq = na.omit(data.SVR.ARMA.pq)
head(data.SVR.ARMA.pq)

#template
data = data.SVR.ARMA.pq
head(data)
SVRresult = list()
resitrain = resitest = resi = vector()
SVRresult = fitSVR(data, startTrain, endTrain, endTest)

makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(model.SVR[idx.svr],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(model.SVR[idx.svr],"Test"), xlabel = xlabel, ylabel=ylabel)

for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}

resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)

ujiLM(resi)

result.SVR.ARMA.pq = SVRresult

############################
# UJI Linearitas GARCH
############################
SVRresult = list()
resitrain = resitest = resi = vector()

SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)
rt2 = data.SVR.ARMA.pq$rt^2
at2 = resi^2

chisq.linear = terasvirta.test(ts(rt2), lag = 1, type = "Chisq")
if(chisq.linear$p.value<alpha){
  cat("Dengan Statistik uji Chisquare, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji Chisquare, Gagal Tolak H0, data linear")
}
F.linear = terasvirta.test(ts(rt2), lag = 1, type = "F")
if(F.linear$p.value<alpha){
  cat("Dengan Statistik uji F, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji F, Gagal Tolak H0, data linear")
}

source("allfunction.R")
############################
# 2. Model ARMA-GARCH-SVR
############################
idx.svr=2
msemodel.SVRname.SVR[idx.svr] = "ARMA-GARCH-based SVR"
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
par(mfrow=c(2,1))

#get lag signifikan
acf.resikuadrat = acf(resitrain^2, lag.max = maxlag, type = "correlation")
acf.resikuadrat <- acf.resikuadrat$acf[2:(maxlag+1)]
pacf.resikuadrat = pacf(resitrain^2, lag.max = maxlag)
pacf.resikuadrat <- pacf.resikuadrat$acf[1:maxlag]
batas.at2 = 1.96/sqrt(length(resitrain)-1)

optlag = getLagSignifikan(at2, maxlag = maxlag, batas = batas.at2, alpha = alpha, na=FALSE)
data.SVR.ARCH = makeData(data = base.data, datalag = at2, numlag = optlag$ARlag, lagtype = "at2")
data.SVR.GARCH = makeData(data = data.SVR.ARCH, datalag = rt2, numlag=optlag$MAlag, lagtype = "rt2")
data.SVR.GARCH.ori = data.SVR.GARCH
data.SVR.GARCH = na.omit(data.SVR.GARCH)
head(data.SVR.GARCH)

#template
data = data.SVR.GARCH
head(data)
SVRresult = list()
resitrain = resitest = resi = vector()
SVRresult = fitSVR(data, startTrain, endTrain, endTest)

makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(model.SVR[idx.svr],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(model.SVR[idx.svr],"Test"), xlabel=xlabel, ylabel=ylabel)

for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}
result.SVR.GARCH = SVRresult

############################
# UJI PERUBAHAN STRUKTUR
############################
source("allfunction.R")
head(data.SVR.GARCH)
dim(data.SVR.GARCH)
chowtest = ujiperubahanstruktur(data.SVR.GARCH, startTrain, endTrain, endTest, alpha)

# i = (3, 4) harus running berurutan, 
# karena proses ambil veriabel dan datanya nyambung
############################
# 3. MSGARCH -> sGARCH, norm
############################
idx.svr=3
model.SVR[idx.svr] = "MSGARCH input rt"
SVRresult = list()
ylabel = "return kuadrat"
xlabel="t"

msgarch = fitMSGARCH(data = dataTrain$return, TrainActual = dataTrain$rv, TestActual=dataTest$rv, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

makeplot(msgarch$train$actual, msgarch$train$predict, paste(model.SVR,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(msgarch$test$actual, msgarch$test$predict, paste(model.SVR,"Test"),xlabel=xlabel, ylabel=ylabel)

#akurasi
SVRresult = msgarch

for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
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
# 4. MSGARCH-based SVR -> input rt"
############################
m=4
model.SVR[idx.svr] = "rt MSGARCH-SVR"
msgarch.model = msgarch.rt

#get variabel input ML
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1pit","v2pit")
par(mfrow=c(1,1))
plot(mydata$rv, type="l")
lines(rowSums(v), col="red")
lines(c(msgarch.model$train$predict,msgarch.model$test$predict),col="green")

# form the msgarch data
time = mydata$date
rt2 = mydata$rv
base.data = data.frame(time,rt2,v)
head(base.data)
data.SVR.MSGARCH.rt = na.omit(base.data)

#template
data = dataMSGARCH.rt
head(data)
SVRresult = list()
SVRresult = fitSVR(data, startTrain, endTrain, endTest)

makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(model.SVR[idx.svr],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(model.SVR[idx.svr],"Test"), xlabel=xlabel, ylabel=ylabel)

for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}
result.SVR.MSGARCH.rt.pit = SVRresult

# i = (5, 6) harus running berurutan, 
# karena proses ambil veriabel dan datanya nyambung
############################
# 5. MSGARCH -> input at
############################
m=5
model.SVR[idx.svr] = "MSGARCH input at"
ylabel = "return kuadrat"
xlabel="t"

SVRresult = list()
resitrain = resitest = resi = vector()

SVRresult = SVRresult.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)

msgarch = fitMSGARCH(data = resitrain, TrainActual = SVRresult$train$actual^2, TestActual=SVRresult$test$actual^2, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

makeplot(msgarch$train$actual, msgarch$train$predict, paste(model.SVR,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(msgarch$test$actual, msgarch$test$predict, paste(model.SVR,"Test"),xlabel=xlabel, ylabel=ylabel)

#akurasi
SVRresult = list()
SVRresult = msgarch

for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
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
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = resitrain, TrainActual = SVRresult$train$actual^2, 
                               TestActual=SVRresult$test$actual^2, nfore=nfore, nstate=2)
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
# 6. MSGARCH-based SVR -> input at"
############################
m=6
model.SVR[idx.svr] = "at MSGARCH-SVR"
msgarch.model = msgarch.at
#get variabel input
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1pit","v2pit")
par(mfrow=c(1,1))
plot(mydata$rv, type="l")
lines(rowSums(v), col="red")
lines(c(msgarch.model$train$predict,msgarch.model$test$predict),col="green")


# form the msgarch data
time = data.SVR.ARMA.pq$time
rt2 = data.SVR.ARMA.pq$rt^2
base.data = data.frame(time,rt2,v)
data.SVR.MSGARCH.at.pit = na.omit(base.data)

#template
data = data.SVR.MSGARCH.at.pit
head(data)
SVRresult = list()
resitrain = resitest = resi = vector()
SVRresult = fitSVR(data, startTrain, endTrain, endTest)

makeplot(SVRresult$train$actual, SVRresult$train$predict, paste(model.SVR[idx.svr],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRresult$test$actual, SVRresult$test$predict, paste(model.SVR[idx.svr],"Test"), xlabel=xlabel, ylabel=ylabel)

for(j in 1:len.loss){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRresult$train$actual, SVRresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRresult$test$actual, SVRresult$test$predict, method = lossfunction[j])
}

result.SVR.MSGARCH.at.pit = SVRresult

############################
# PERBANDINGAN AKURASI
############################
msename = msename.SVR
MSEtrain = MSEtrain.SVR
MSEtesting = MSEtest.SVR

allMSE = data.frame(MSEtrain,MSEtest)
ranktrain = rank(allMSE$MSEtrain)
ranktest = rank(allMSE$MSEtest)
allMSE = data.frame(MSEtrain, ranktrain, MSEtest, ranktest)
rownames(allMSE) = msename
allMSE

besttraining = msename[which.min(allMSE$MSEtrain)]
besttesting = msename[which.min(allMSE$MSEtest)]

cat("model terbaik berdasarkan data training :",besttraining)
cat("model terbaik berdasarkan data testing :",besttesting)

allMSE

MSErank = cbind(msename,rank(allMSE$MSEtrain))
MSErank

a = MSEtrain[3]
b = MSEtrain[4]
pct = (a-b)/a
pct

a = MSEtest[3]
b = MSEtest[8]
pct = (a-b)/a
pct
