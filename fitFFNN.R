setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML/")
source("getDataLuxemburg.R")
source("allfunction.R")

optARMAlag


#inisialisasi
act.fnc = "logistic"
neuron = c(1:20)

# scalling dilakukan untuk pemodelan varians karena jika untuk model mean hasilnya tidak konvergen untuk fungsi aktivasi linear
# jika pemodelan varians tanpa scalling hasil peramalan flat dan jauh diatas realisasi volatilitas di sekitar 10^-3
# scalling pada permodelan varians menunjukkan hasil peramalan tidak flat which is good dan dekat dengan nilai aktual 
# predtest at dg nilai return -> tidak flat
# pred test at dg nilai resitest => flat
# tapi akurasinya masih dibawah model lain
# scalling dengan min max hasilnya mendekti 0
use_sliding_window = TRUE
window_size = 5

model.NN = vector()
lossfunction = getlossfunction()
len.loss = length(lossfunction)
losstrain.NN = matrix(nrow=9, ncol=len.loss)
losstest.NN = matrix(nrow=9, ncol=len.loss)
colnames(losstrain.NN) = lossfunction
colnames(losstest.NN) = lossfunction

#split data train & data test 
dataTrain = mydata%>%filter(date >= as.Date(startTrain) & date <= as.Date(endTrain) )
dataTest = mydata%>%filter(date > as.Date(endTrain) & date <= as.Date(endTest) )
nfore = dim(dataTest)[1];nfore

############################
# 1. Model ARMA-FFNN
############################
idx.ffnn=1
model.NN[idx.ffnn] = "ARMA-FFNN"
ylabel = "return"
xlabel = "t"
base.data = data.frame(time=mydata$date,y=mydata$return)
head(base.data)

##### Model AR #####
#get data AR
data.NN.AR = makeData(data = base.data, datalag = base.data$y, numlag = optARMAlag$ARlag, lagtype = "rt")
data.NN.AR = na.omit(data.NN.AR)

# fit NN model
source("allfunction.R")
data = data.NN.AR
head(data)
result.NN.AR = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=TRUE, scale=FALSE)

# get best result
result = list()
result = result.NN.AR
data = data.NN.AR
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]

n.neuron = length(neuron)
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

bestresult.NN.AR = bestresult

# plot the prediction result
title = "AR-FFNN"
NNbestresult = list()
NNbestresult = bestresult.NN.AR
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)

##### Model ARMA #####
NNbestresult = list()
resitrain = resitest = resi = vector()
base.data = data.frame()

dataall = mydata$return
base.data = data.NN.AR
NNbestresult = bestresult.NN.AR
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)

#get data only significant lag
data.NN.ARMA = makeData(data = base.data, datalag = resi, numlag = optARMAlag$MAlag, lagtype = "et")
data.NN.ARMA = na.omit(data.NN.ARMA)

# fit NN model
data = data.NN.ARMA
head(data)
result.NN.ARMA = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=TRUE, scale=FALSE)

# get best result
result = list()
result = result.NN.ARMA
data = data.NN.ARMA
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

bestresult.NN.ARMA = bestresult

# plot the prediction result
title = model.NN[idx.ffnn]
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}


############################
# UJI LAGRANGE MULTIPLIER
############################
source("allfunction.R")
NNbestresult = list()
resitrain = resitest = resi = vector()

NNbestresult = bestresult.NN.ARMA
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)
LMtest(resi)

############################
# 2. Model GARCH-FFNN
############################
idx.ffnn=2
model.NN[idx.ffnn] = "GARCH-FFNN"
ylabel = "volatilitas"
xlabel = "t" 

# get rt2
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


#### Model ARCH #####
time = mydata$date
base.data = data.frame(time,y=rt2)
head(base.data)
data.NN.ARCH = makeData(data = base.data, datalag = rt2, numlag = optlag$PACFlag, lagtype = "rt2")
data.NN.ARCH = na.omit(data.NN.ARCH)

# fit NN model
source("allfunction.R")
data = data.NN.ARCH
head(data)
result.NN.ARCH = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=FALSE, scale=TRUE)

# get best result
result = list()
trainactual = testactual = vector()

result = result.NN.ARCH
data = data.NN.ARCH
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

bestresult.NN.ARCH = bestresult

# plot the prediction result
title = "ARCH-FFNN"
NNbestresult = list()
NNbestresult = bestresult.NN.ARCH
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
##### end of Model ARCH #####


##### Model GARCH #####
# get resi ARCH ut, di buku mba shindi tidak dikuadratkan
NNbestresult = list()
resitrain = resitest = resi = vector()

NNbestresult = bestresult.NN.ARCH
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)
ut2 = resi^2

# get data GARCH
data.NN.GARCH = makeData(data = data.NN.ARCH, datalag = ut2, numlag=optlag$ACFlag, lagtype = "ut2")
data.NN.GARCH = na.omit(data.NN.GARCH)


# fit NN model
source("allfunction.R")
data = data.NN.GARCH
head(data)
result.NN.GARCH = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=FALSE, scale=TRUE)

# get best result
result = list()
trainactual = testactual = vector()

result = result.NN.GARCH
data = data.NN.GARCH
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

bestresult.NN.GARCH = bestresult

# plot the prediction result
title = model.NN[idx.ffnn]
NNbestresult = list()
NNbestresult = bestresult.NN.GARCH
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in seq_along(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}


############################
# UJI PERUBAHAN STRUKTUR
############################
source("allfunction.R")
head(data.NN.GARCH)
dim(data.NN.GARCH)
chowtest = ujiperubahanstruktur(data.NN.GARCH, startTrain, endTrain, endTest, alpha)


############################
# 3. Model ARMA-GARCH-FFNN
############################
idx.ffnn=3
model.NN[idx.ffnn] = "ARMA-GARCH-FFNN"
ylabel = "volatilitas"
xlabel = "t" 

# get resi ARMA at
NNbestresult = list()
resitrain = resitest = resi = vector()

NNbestresult = bestresult.NN.ARMA
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
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


#### Model ARCH #####
# get data ARCH
time = data.NN.ARMA$time
base.data = data.frame(time,y=at2)
head(base.data)
data.NN.ARMA.ARCH = makeData(data = base.data, datalag = at2, numlag = optlag$PACFlag, lagtype = "at2")
data.NN.ARMA.ARCH = na.omit(data.NN.ARMA.ARCH)

# fit NN model
source("allfunction.R")
data = data.NN.ARMA.ARCH
head(data)
result.NN.ARMA.ARCH = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=FALSE, scale=TRUE)


# get best result
result = list()
trainactual = vector()
testactual = vector()
result = result.NN.ARMA.ARCH
data = data.NN.ARMA

max.lag.sig = max(optlag$PACFlag)
t.all = nrow(data)
trainactual = (data$y[(max.lag.sig+1):(t.all-nfore)])^2
testactual = (data$y[(t.all-nfore+1):t.all])^2
rt.hat.train = bestresult.NN.ARMA$train$predict[-c(1:max.lag.sig)]
rt.hat.test = bestresult.NN.ARMA$test$predict

length(rt.hat.train)
length(trainactual)
length(result.NN.ARMA.ARCH$`Neuron 1`$train)

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

bestresult.NN.ARMA.ARCH = bestresult

# plot the prediction result
title = "ARMA-ARCH-FFNN"
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA.ARCH
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
##### end of Model ARCH #####


##### Model GARCH #####
# get resi ARCH ut, di buku mba shindi tidak dikuadratkan
NNbestresult = list()
resitrain = resitest = resi = vector()

NNbestresult = bestresult.NN.ARMA.ARCH
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)
ut2 = resi 
data.NN.ARMA.GARCH = makeData(data = data.NN.ARMA.ARCH, datalag = ut2, numlag=optlag$ACFlag, lagtype = "ut2")
data.NN.ARMA.GARCH = na.omit(data.NN.ARMA.GARCH)


# fit NN model
source("allfunction.R")
data = data.NN.ARMA.GARCH
head(data)
result.NN.ARMA.GARCH = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=FALSE, scale=TRUE)


# get best result
result = list()
trainactual = testactual = vector()
result = result.NN.ARMA.GARCH
data = data.NN.ARMA

max.lag.sig = max(optlag$PACFlag)+max(optlag$ACFlag)
t.all = nrow(data)
trainactual = (data$y[(max.lag.sig+1):(t.all-nfore)])^2
testactual = (data$y[(t.all-nfore+1):t.all])^2
rt.hat.train = bestresult.NN.ARMA$train$predict[-c(1:max.lag.sig)]
rt.hat.test = bestresult.NN.ARMA$test$predict

length(trainactual)
length(result.NN.ARMA.GARCH$`Neuron 1`$train)
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

bestresult.NN.ARMA.GARCH = bestresult

# plot the prediction result
title = model.NN[idx.ffnn]
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA.GARCH
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in seq_along(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}


############################
# UJI PERUBAHAN STRUKTUR
############################
source("allfunction.R")
head(data.NN.ARMA.GARCH)
dim(data.NN.ARMA.GARCH)
chowtest = ujiperubahanstruktur(data.NN.ARMA.GARCH, startTrain, endTrain, endTest, alpha)


############################
# 4. MSGARCH -> sGARCH, norm
# i = (4, 5) harus running berurutan, 
# karena proses ambil veriabel dan datanya nyambung
############################
idx.ffnn=4
model.NN[idx.ffnn] = "MSGARCH"
ylabel = "volatilitas"
xlabel="t"

result.NN.MSGARCH = fitMSGARCH(data = dataTrain$return, TrainActual = dataTrain$return^2, TestActual=dataTest$return^2, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

# get best result
result = list()
trainactual = testactual = vector()
result = result.NN.MSGARCH
data = mydata
t.all = nrow(data)
trainactual = dataTrain$return^2
testactual = dataTest$return^2

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = result.NN.MSGARCH$train
bestresult$test$actual = testactual
bestresult$test$predict = result.NN.MSGARCH$test

bestresult.NN.MSGARCH = bestresult

# plotting the prediction result
title = "MSGARCH"
NNbestresult = list()
NNbestresult = bestresult.NN.MSGARCH
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"),xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

source("allfunction.R")
############################
# 5. MSGARCH-based FFNN -> input rt "
############################
idx.ffnn=5
model.NN[idx.ffnn] = "MSGARCH-FFNN"
msgarch.model = result.NN.MSGARCH

##### Essential section for MSGARCH-NN process clean code #####
SR.fit <- ExtractStateFit(msgarch.model$modelfit)
K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=dim(dataTrain)[1], ncol=K)
voltest = matrix(nrow=dim(dataTest)[1], ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = dataTrain$return^2, TrainActual = dataTrain$return^2, 
                               TestActual=dataTest$rv, nfore, nstate=2)
  
  voltrain[,k] = msgarch.SR[[k]]$train
  voltest[,k] = msgarch.SR[[k]]$test
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


#get variabel input ML
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1p1t","v2p2t")
par(mfrow=c(1,1))
plot(mydata$rv, type="l")
lines(rowSums(v), col="red")
lines(c(bestresult.NN.MSGARCH$train$predict,bestresult.NN.MSGARCH$test$predict),col="green")

# form the msgarch data
time = mydata$date
rt2 = mydata$return^2
base.data = data.frame(time,y=rt2,v)
dim(base.data)
data.NN.MSGARCH.NN= na.omit(base.data)

# fit NN model
data = data.NN.MSGARCH.NN
head(data)
result.NN.MSGARCH.NN = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=FALSE, scale=TRUE)

# get best result
result = list()
trainactual = testactual = vector()

result = result.NN.MSGARCH.NN
data = data.NN.MSGARCH.NN
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

bestresult.NN.MSGARCH.NN = bestresult


# plotting the prediction result
title = "MSGARCH-FFNN"
NNbestresult = list()
NNbestresult = bestresult.NN.MSGARCH.NN
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}


source("allfunction.R")
############################
# 6. MSGARCH -> input at
# i = (6,7) harus running berurutan, 
# karena proses ambil variabel dan datanya nyambung
############################
idx.ffnn=6
model.NN[idx.ffnn] = "ARMA-MSGARCH"
ylabel = "volatilitas"
xlabel="t"
NNbestresult = list()
resitrain = resitest = resi = vector()

NNbestresult = bestresult.NN.ARMA
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)

result.NN.ARMA.MSGARCH = fitMSGARCH(data = resitrain, TrainActual = resitrain^2, TestActual=resitest^2, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

# get best result
result = list()
trainactual = testactual = vector()
result = result.NN.ARMA.MSGARCH
data = data.NN.ARMA

t.all = nrow(data)
trainactual = bestresult.NN.ARMA$train$actual^2
testactual = bestresult.NN.ARMA$test$actual^2
rt.hat.train = bestresult.NN.ARMA$train$predict
rt.hat.test = bestresult.NN.ARMA$test$predict

length(rt.hat.train)
length(resitrain)
length(result.NN.ARMA.MSGARCH$train)

trainpred = (rt.hat.train + sqrt(result$train))^2
testpred = (rt.hat.test + sqrt(result$test))^2

bestresult = list()
bestresult$train$actual = trainactual
bestresult$train$predict = trainpred
bestresult$test$actual = testactual
bestresult$test$predict = testpred

bestresult.NN.ARMA.MSGARCH = bestresult

# plotting the prediction result
title = model.NN[idx.ffnn]
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA.MSGARCH
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"),xlabel=xlabel, ylabel=ylabel)


# calculate the prediction error
for(j in 1:length(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

source("allfunction.R")
############################
# 7. MSGARCH-based FFNN -> input at"
############################
idx.ffnn=7
model.NN[idx.ffnn] = "ARMA-MSGARCH-FFNN"
msgarch.model = result.NN.ARMA.MSGARCH

NNbestresult = list()
resitrain = resitest = resi = vector()

NNbestresult = bestresult.NN.ARMA
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)

##### Essential section for MSGARCH-FFNN process clean code #####
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
lines(c(bestresult.NN.ARMA.MSGARCH$train$predict,bestresult.NN.ARMA.MSGARCH$test$predict),col="green")

# form the msgarch data
time = data.NN.ARMA$time
at2 = resi^2
trainactual = testactual = rt.hat.train = rt.hat.test = vector()

trainactual = bestresult.NN.ARMA$train$actual^2
testactual = bestresult.NN.ARMA$test$actual^2
rt.hat.train = bestresult.NN.ARMA$train$predict
rt.hat.test = bestresult.NN.ARMA$test$predict


base.data = data.frame(time,y=at2,v)
data.NN.ARMA.MSGARCH.NN = na.omit(base.data)

# fit NN model
data = data.NN.ARMA.MSGARCH.NN
head(data)
result.NN.ARMA.MSGARCH.NN = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=FALSE, scale=TRUE)

# get best result
result = list()
result = result.NN.ARMA.MSGARCH.NN

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

bestresult.NN.ARMA.MSGARCH.NN = bestresult

# plotting the prediction result
title = model.NN[idx.ffnn]
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA.MSGARCH.NN
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:length(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}


source("allfunction.R")
############################
# 8. MSGARCH-based FFNN -> input rt sliding window 5"
############################
idx.ffnn=8
model.NN[idx.ffnn] = "MSGARCH-FFNN window5"
msgarch.model = result.NN.MSGARCH

##### Essential section for MSGARCH-NN process clean code #####
SR.fit <- ExtractStateFit(msgarch.model$modelfit)
K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=dim(dataTrain)[1], ncol=K)
voltest = matrix(nrow=dim(dataTest)[1], ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = dataTrain$return^2, TrainActual = dataTrain$return^2, 
                               TestActual=dataTest$rv, nfore, nstate=2)
  
  voltrain[,k] = msgarch.SR[[k]]$train
  voltest[,k] = msgarch.SR[[k]]$test
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


#get variabel input ML
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1p1t","v2p2t")
par(mfrow=c(1,1))
plot(mydata$rv, type="l")
lines(rowSums(v), col="red")
lines(c(bestresult.NN.MSGARCH$train$predict,bestresult.NN.MSGARCH$test$predict),col="green")

# form the msgarch data
time = mydata$date
rt2 = mydata$return^2

window_size = 5
window.data = sliding_window(x=v, y=rt2, window_size = window_size)
time = time[(window_size+1):nrow(v)]
rt2 = window.data$y
v = window.data$x
length(rt2)
dim(v)

base.data = data.frame(time,y=rt2,v)
dim(base.data)
data.NN.MSGARCH.NN.window5 = na.omit(base.data)

# fit NN model
data = data.NN.MSGARCH.NN.window5
head(data)
result.NN.MSGARCH.NN.window5 = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=FALSE, scale=TRUE)

# get best result
result = list()
trainactual = vector()
testactual = vector()

result = result.NN.MSGARCH.NN.window5
data = data.NN.MSGARCH.NN.window5
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]
length(trainactual)
length(result.NN.MSGARCH.NN.window5$`Neuron 1`$train)
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

bestresult.NN.MSGARCH.NN.window5 = bestresult

# plotting the prediction result
title = "MSGARCH-FFNN window5"
NNbestresult = list()
NNbestresult = bestresult.NN.MSGARCH.NN.window5
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

source("allfunction.R")
############################
# 9. MSGARCH-based FFNN -> input at sliding window 5"
############################
idx.ffnn=9
model.NN[idx.ffnn] = "ARMA-MSGARCH-FFNN window5"
msgarch.model = result.NN.ARMA.MSGARCH

NNbestresult = list()
resitrain = resitest = resi = vector()

NNbestresult = bestresult.NN.ARMA
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)

##### Essential section for MSGARCH-FFNN process clean code #####
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
lines(c(bestresult.NN.ARMA.MSGARCH$train$predict,bestresult.NN.ARMA.MSGARCH$test$predict),col="green")

# form the msgarch data
time = data.NN.ARMA$time
at2 = resi^2
trainactual = testactual = rt.hat.train = rt.hat.test = vector()

window_size = 5
window.data = sliding_window(x=v, y=at2, window_size = window_size)
time = time[(window_size+1):nrow(v)]
at2 = window.data$y
v = window.data$x
length(at2)
dim(v)

trainactual = (bestresult.NN.ARMA$train$actual[-c(1:window_size)])^2
testactual = (bestresult.NN.ARMA$test$actual)^2
rt.hat.train = bestresult.NN.ARMA$train$predict[-c(1:window_size)]
rt.hat.test = bestresult.NN.ARMA$test$predict


base.data = data.frame(time,y=at2,v)
data.NN.ARMA.MSGARCH.NN.window5 = na.omit(base.data)

# fit NN model
data = data.NN.ARMA.MSGARCH.NN.window5
head(data)
result.NN.ARMA.MSGARCH.NN.window5 = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=FALSE, scale=TRUE)

# get best result
result = list()
result = result.NN.ARMA.MSGARCH.NN.window5

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

bestresult.NN.ARMA.MSGARCH.NN.window5 = bestresult

# plotting the prediction result
title = model.NN[idx.ffnn]
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA.MSGARCH.NN.window5
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:length(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}
############################
# PERBANDINGAN AKURASI
############################
rownames(losstrain.NN) = model.NN
rownames(losstest.NN) = model.NN

which.min(rowSums(losstrain.NN))
ranktrain = data.frame(losstrain.NN, rank = rank(losstrain.NN[,1]))
ranktest = data.frame(losstest.NN, rank = rank(losstest.NN[,1]))

cat("min loss in data training is",model.NN[which.min(ranktrain$MSE)])
cat("min loss in data testing is",model.NN[which.min(ranktest$MSE)])
ranktrain
ranktest

############################
# Save all data and result
############################
save(data.NN.AR, data.NN.ARMA, data.NN.ARCH, data.NN.GARCH, data.NN.ARMA.ARCH, data.NN.ARMA.GARCH, data.NN.MSGARCH.NN, 
     data.NN.ARMA.MSGARCH.NN.window5, data.NN.ARMA.MSGARCH.NN, data.NN.ARMA.MSGARCH.NN.window5,
     file="final result/datauji_NN.RData")
save(result.NN.AR, result.NN.ARMA, result.NN.ARCH, result.NN.GARCH, result.NN.ARMA.ARCH, result.NN.ARMA.GARCH, result.NN.MSGARCH,
     result.NN.MSGARCH.NN, result.NN.MSGARCH.NN.window5, result.NN.ARMA.MSGARCH, result.NN.ARMA.MSGARCH.NN,
     result.NN.ARMA.MSGARCH.NN.window5, file="final result/result_NN.RData")
save(bestresult.NN.AR, bestresult.NN.ARMA, bestresult.NN.ARCH, bestresult.NN.GARCH, bestresult.NN.ARMA.ARCH, bestresult.NN.ARMA.GARCH,
     bestresult.NN.MSGARCH, bestresult.NN.MSGARCH.NN, bestresult.NN.MSGARCH.NN.window5, bestresult.NN.ARMA.MSGARCH, 
     bestresult.NN.ARMA.MSGARCH.NN, bestresult.NN.ARMA.MSGARCH.NN.window5, file="final result/bestresult_NN.RData")
save(losstrain.NN, losstest.NN, file="final result/loss_NN.RData")

# save(data.NN.AR.p, data.NN.ARMA.pq, data.NN.ARCH, data.NN.GARCH, data.NN.ARMA.ARCH, data.NN.ARMA.GARCH, 
#      data.NN.MSGARCH.rt,data.NN.MSGARCH.at, file = "finalresult/Datauji_NN_window5.RData")
# save(result.NN.AR.p, result.NN.ARMA.pq, result.NN.ARCH, result.NN.GARCH, result.NN.ARMA.ARCH, result.NN.ARMA.GARCH, 
#      result.NN.MSGARCH.rt, result.NN.MSGARCH.at, file="finalresult/result_NN_window5.RData")
# save(bestresult.NN.AR.p, bestresult.NN.ARMA.pq, bestresult.NN.ARCH, bestresult.NN.GARCH, bestresult.NN.ARMA.ARCH, 
#      bestresult.NN.ARMA.GARCH, bestresult.NN.MSGARCH.rt, bestresult.NN.MSGARCH.at, file="finalresult/bestresult_NN_window5.RData")
# save(losstrain.NN, losstest.NN, file="finalresult/loss_NN_window5.RData")

