setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML/")
source("getDataLuxemburg.R")
source("allfunction.R")

optARMAlag


#inisialisasi
act.fnc = "logistic"
neuron = c(1,2,3,4,5,10,15,20)

# scalling dilakukan untuk pemodelan varians karena jika untuk model mean hasilnya tidak konvergen untuk fungsi aktivasi linear
# jika pemodelan varians tanp ascalling hasil peramalan flat dan jauh diatas realisasi volatilitas di sekitar 10^-3
# scalling pada permodelan varians menunjukkan hasil peramalan tidak flat which is good dan dekat dengan nilai aktual
# tapi akurasinya masih dibawah model lain
# scalling dengan min max hasilnya mendekti 0
use_sliding_window = TRUE
window_size = 5

model.NN = vector()
lossfunction = getlossfunction()
len.loss = length(lossfunction)
losstrain.NN = matrix(nrow=6, ncol=len.loss)
losstest.NN = matrix(nrow=6, ncol=len.loss)
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
model.NN[idx.ffnn] = "ARMA-based FFNN"
ylabel = "return"
xlabel = "time index"
base.data = data.frame(mydata$date,mydata$return)
colnames(base.data) = c("time","rt")
head(base.data)

##### Model AR #####
#get data AR(p)
data.NN.AR.p = makeData(data = base.data, datalag = base.data$rt, numlag = optARMAlag$ARlag, lagtype = "rt")
data.NN.AR.p = na.omit(data.NN.AR.p)

# fit NN model
source("allfunction.R")
data = data.NN.AR.p
head(data)
result.NN.AR.p = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=TRUE, scale=FALSE)
bestresult.NN.AR.p = result.NN.AR.p[[result.NN.AR.p$opt_idx]]

# plot the prediction result
NNbestresult = list()
NNbestresult = bestresult.NN.AR.p
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(model.NN[idx.ffnn],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(model.NN[idx.ffnn],"Test"), xlabel = xlabel, ylabel=ylabel)

##### Model ARMA #####
NNbestresult = list()
resitrain = resitest = resi = vector()
base.data = data.frame()

dataall = mydata$return
base.data = data.NN.AR.p
NNbestresult = bestresult.NN.AR.p
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)

#get data only significant lag
data.NN.ARMA.pq = makeData(data = base.data, datalag = resi, numlag = optARMAlag$MAlag, lagtype = "at")
data.NN.ARMA.pq = na.omit(data.NN.ARMA.pq)

# fit NN model
data = data.NN.ARMA.pq
head(data)
resitrain = resitest = resi = vector()
result.NN.ARMA.pq = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=TRUE, scale=FALSE)
bestresult.NN.ARMA.pq = result.NN.ARMA.pq[[result.NN.ARMA.pq $opt_idx]]

# plot the prediction result
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA.pq
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(model.NN[idx.ffnn],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(model.NN[idx.ffnn],"Test"), xlabel = xlabel, ylabel=ylabel)

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
NNbestresult = bestresult.NN.ARMA.pq
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)
LMtest(resi)

############################
# 2. Model ARMA-GARCH-FFNN
############################
idx.ffnn=2
model.NN[idx.ffnn] = "ARMA-GARCH-based FFNN"
ylabel = "return kuadrat"
xlabel = "t" 
NNbestresult = list()
resitrain = resitest = resi = vector()

NNbestresult = bestresult.NN.ARMA.pq
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)
rt2 = data.NN.ARMA.pq$rt^2
at = resi
at2 = resi^2
time = data.NN.ARMA.pq[,1]
base.data = data.frame(time,rt2)
head(base.data)

#get lag signifikan
par(mfrow=c(1,2))
acf.resikuadrat = acf(at2, lag.max = maxlag, type = "correlation")
acf.resikuadrat <- acf.resikuadrat$acf[2:(maxlag+1)]
pacf.resikuadrat = pacf(at2, lag.max = maxlag)
pacf.resikuadrat <- pacf.resikuadrat$acf[1:maxlag]
batas.at2 = 1.96/sqrt(length(at2)-1)

optlag = getLagSignifikan(at2, maxlag = maxlag, batas = batas.at2, alpha = alpha, na=FALSE)
data.NN.ARCH = makeData(data = base.data, datalag = at2, numlag = optlag$PACFlag, lagtype = "at2")
data.NN.GARCH = makeData(data = data.NN.ARCH, datalag = rt2, numlag=optlag$ACFlag, lagtype = "rt2")
data.NN.GARCH = na.omit(data.NN.GARCH)

##### UJI Linearitas GARCH #####
NNbestresult = list()
resitrain = resitest = resi = vector()

NNbestresult = bestresult.NN.ARMA.pq
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)
rt2 = data.NN.ARMA.pq$rt^2
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


# fit NN model
source("allfunction.R")
data = data.NN.GARCH
head(data)
result.NN.GARCH = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=FALSE, scale=TRUE)
bestresult.NN.GARCH = result.NN.GARCH[[result.NN.GARCH$opt_idx]]

# plot the prediction result
NNbestresult = list()
NNbestresult = bestresult.NN.GARCH
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(model.NN[idx.ffnn],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(model.NN[idx.ffnn],"Test"), xlabel = xlabel, ylabel=ylabel)

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


# i = (3, 4) harus running berurutan, 
# karena proses ambil veriabel dan datanya nyambung
############################
# 3. MSGARCH -> sGARCH, norm
############################
idx.ffnn=3
model.NN[idx.ffnn] = "MSGARCH input rt"
ylabel = "return kuadrat"
xlabel="t"

msgarch.NN.rt = fitMSGARCH(data = dataTrain$return, TrainActual = dataTrain$rv, TestActual=dataTest$rv, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

# plotting the prediction result
NNbestresult = list()
NNbestresult = msgarch.NN.rt
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(model.NN[idx.ffnn],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(model.NN[idx.ffnn],"Test"),xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

##### Essential section for MSGARCH-NN process clean code #####
msgarch.model = msgarch.NN.rt
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
# 4. MSGARCH-based FFNN -> input rt "
############################
idx.ffnn=4
model.NN[idx.ffnn] = "rt MSGARCH-FFNN"
msgarch.model = msgarch.NN.rt

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
if(use_sliding_window){
  window_size = 5
  window.data = sliding_window(x=v, y=rt2, window_size = window_size)
  time = time[(window_size+1):nrow(v)]
  rt2 = window.data$y
  v = window.data$x
  length(rt2)
  dim(v)
}
base.data = data.frame(time,rt2,v)
dim(base.data)
data.NN.MSGARCH.rt= na.omit(base.data)

# fit NN model
data = data.NN.MSGARCH.rt
head(data)
result.NN.MSGARCH.rt = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=FALSE, scale=TRUE)
bestresult.NN.MSGARCH.rt = result.NN.MSGARCH.rt[[result.NN.MSGARCH.rt$opt_idx]]

# plotting the prediction result
NNbestresult = list()
NNbestresult = bestresult.NN.MSGARCH.rt
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(model.NN[idx.ffnn],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(model.NN[idx.ffnn],"Test"), xlabel = xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}


source("allfunction.R")
# i = (5,6) harus running berurutan, 
# karena proses ambil variabel dan datanya nyambung
############################
# 5. MSGARCH -> input at
############################
idx.ffnn=5
model.NN[idx.ffnn] = "MSGARCH input at"
ylabel = "return kuadrat"
xlabel="t"
NNbestresult = list()
resitrain = resitest = resi = vector()

NNbestresult = bestresult.NN.ARMA.pq
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)

msgarch.NN.at = fitMSGARCH(data = resitrain, TrainActual = NNbestresult$train$actual^2, TestActual=NNbestresult$test$actual^2, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

# plotting the prediction result
NNbestresult = list()
NNbestresult = msgarch.NN.at
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(model.NN[idx.ffnn],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(model.NN[idx.ffnn],"Test"),xlabel=xlabel, ylabel=ylabel)


# calculate the prediction error
for(j in 1:length(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

##### Essential section for MSGARCH-FFNN process clean code #####
msgarch.model = msgarch.NN.at
SR.fit <- ExtractStateFit(msgarch.model$modelfit)
K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=length(resitrain), ncol=K)
voltest = matrix(nrow=length(resitest), ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = resitrain, TrainActual = NNbestresult$train$actual, 
                               TestActual=NNbestresult$test$actual, nfore=nfore, nstate=2)
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
# 6. MSGARCH-based FFNN -> input at"
############################
idx.ffnn=6
model.NN[idx.ffnn] = "at MSGARCH-FFNN"
msgarch.model = msgarch.NN.at

#get variabel input ML
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1p1t","v2p2t")
par(mfrow=c(1,1))
plot(mydata$rv, type="l")
lines(rowSums(v), col="red")
lines(c(msgarch.model$train$predict,msgarch.model$test$predict),col="green")

# form the msgarch data
time = data.NN.ARMA.pq$time
rt2 = data.NN.ARMA.pq$rt^2
if(use_sliding_window){
  window_size = 5
  window.data = sliding_window(x=v, y=rt2, window_size = window_size)
  time = time[(window_size+1):nrow(v)]
  rt2 = window.data$y
  v = window.data$x
  length(rt2)
  dim(v)
}

base.data = data.frame(time,rt2,v)
data.NN.MSGARCH.at = na.omit(base.data)

# fit NN model
data = data.NN.MSGARCH.at
head(data)
result.NN.MSGARCH.at = fitNN(data, startTrain, endTrain, endTest, neuron, linear.output=FALSE, scale=TRUE)
bestresult.NN.MSGARCH.at = result.NN.MSGARCH.at[[result.NN.MSGARCH.at$opt_idx]]

# plotting the prediction result
NNbestresult = list()
NNbestresult = bestresult.NN.MSGARCH.at
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(model.NN[idx.ffnn],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(model.NN[idx.ffnn],"Test"), xlabel = xlabel, ylabel=ylabel)

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
ranktrain = data.frame(losstrain.NN,sum = rowSums(losstrain.NN), rank = rank(rowSums(losstrain.NN)))
ranktest = data.frame(losstest.NN,sum = rowSums(losstest.NN), rank = rank(rowSums(losstest.NN)))

cat("min loss in data training is",model.NN[which.min(ranktrain$sum)])
cat("min loss in data testing is",model.NN[which.min(ranktest$sum)])
ranktrain
ranktest

############################
# Save all data and result
############################
save(data.NN.AR.p, data.NN.ARMA.pq, data.NN.ARCH, data.NN.GARCH,data.NN.MSGARCH.rt,data.NN.MSGARCH.at, file = "data/Datauji_NN_window5.RData")
save(result.NN.AR.p, result.NN.ARMA.pq, result.NN.GARCH, result.NN.MSGARCH.rt, result.NN.MSGARCH.at, file="data/result_NN_window5.RData")
save(bestresult.NN.AR.p, bestresult.NN.ARMA.pq, bestresult.NN.GARCH, bestresult.NN.MSGARCH.rt, bestresult.NN.MSGARCH.at, file="data/bestresult_NN_window5.RData")
save(losstrain.NN, losstest.NN, file="data/loss_NN_window5.RData")

