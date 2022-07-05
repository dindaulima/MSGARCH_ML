setwd("C:/File Sharing/Kuliah/TESIS/tesisdiul/")
source("getDataLuxemburg.R")
source("allfunction.R")

optARMAlag

#inisialisasi
act.fnc = "logistic"
neuron = c(1,2,3,4,5,10,15,20)
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

#get data AR(p)ffnn
data.NN.AR.p = makeData(data = base.data, datalag = base.data$rt, numlag = optARMAlag$ARlag, lagtype = "rt")
data.NN.AR.p = na.omit(data.NN.AR.p)

#template
source("allfunction.R")
data = data.NN.AR.p
head(data)
NNresult = list()
resitrain = resitest = resi = vector()
NNresult = fitNN(data, startTrain, endTrain, endTest, neuron)

NNbestresult = NNresult[[NNresult$opt_idx]]
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(model.NN[idx.ffnn],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(model.NN[idx.ffnn],"Test"), xlabel = xlabel, ylabel=ylabel)

result.NN.AR.p = NNresult
bestresult.NN.AR.p = NNbestresult

##### Model ARMA(p,q) #####
NNresult = list()
resitrain = resitest = resi = vector()
base.data = data.frame()

dataall = mydata$return
base.data = data.NN.AR.p
NNbestresult = result.NN.AR.p[[result.NN.AR.p$opt_idx]]
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)
head(resi)

#get data only significant lag
data.NN.ARMA.pq = makeData(data = base.data, datalag = resi, numlag = optARMAlag$MAlag, lagtype = "at")
data.NN.ARMA.pq = na.omit(data.NN.ARMA.pq)

#template
data = data.NN.ARMA.pq
head(data)
NNresult = list()
resitrain = resitest = resi = vector()
NNresult = fitNN(data, startTrain, endTrain, endTest, neuron)


NNbestresult = NNresult[[NNresult$opt_idx]]
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(model.NN[idx.ffnn],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(model.NN[idx.ffnn],"Test"), xlabel = xlabel, ylabel=ylabel)

for(j in 1:len.loss){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

result.NN.ARMA.pq = NNresult
bestresult.NN.ARMA.pq = NNbestresult


source("allfunction.R")
############################
# 2. Model ARMA-GARCH-FFNN
############################
idx.ffnn=2
model.NN[idx.ffnn] = "ARMA-GARCH-based FFNN"
ylabel = "return kuadrat"
xlabel = "t" 
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
batas.at2 = 1.96/sqrt(length(at)-1)

optlag = getLagSignifikan(at2, maxlag = maxlag, batas = batas.at2, alpha = alpha, na=FALSE)
data.NN.ARCH = makeData(data = base.data, datalag = at2, numlag = optlag$ARlag, lagtype = "at2")
data.NN.GARCH = makeData(data = data.NN.ARCH, datalag = rt2, numlag=optlag$MAlag, lagtype = "rt2")
data.NN.GARCH = na.omit(data.NN.GARCH)


#template
data = data.NN.GARCH
head(data)
NNresult = list()
resitrain = resitest = resi = vector()
NNresult = fitNN(data, startTrain, endTrain, endTest, neuron)

NNbestresult = NNresult[[NNresult$opt_idx]]
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(model.NN[idx.ffnn],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(model.NN[idx.ffnn],"Test"), xlabel = xlabel, ylabel=ylabel)

for(j in seq_along(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

result.NN.GARCH = NNresult
bestresult.NN.GARCH = NNbestresult

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
NNresult = list()
ylabel = "return kuadrat"
xlabel="t"

msgarch = fitMSGARCH(data = dataTrain$return, TrainActual = dataTrain$rv, TestActual=dataTest$rv, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

makeplot(msgarch$train$actual, msgarch$train$predict, paste(model.NN[idx.ffnn],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(msgarch$test$actual, msgarch$test$predict, paste(model.NN[idx.ffnn],"Test"),xlabel=xlabel, ylabel=ylabel)

#akurasi
NNbestresult = msgarch
for(j in 1:len.loss){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
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
# 4. MSGARCH-based FFNN -> input rt "
############################
idx.ffnn=4
model.NN[idx.ffnn] = "rt MSGARCH-FFNN"
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
data.NN.MSGARCH.rt.pit= na.omit(base.data)

#template
data = data.NN.MSGARCH.rt.pit
head(data)
NNresult = list()
NNresult = fitNN(data, startTrain, endTrain, endTest, neuron)

NNbestresult = NNresult[[NNresult$opt_idx]]
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(model.NN[idx.ffnn],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(model.NN[idx.ffnn],"Test"), xlabel = xlabel, ylabel=ylabel)

for(j in 1:len.loss){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

result.NN.MSGARCH.rt.pit = NNresult
bestresult.NN.MSGARCH.rt.pit = NNbestresult


source("allfunction.R")
# i = (5,6) harus running berurutan, 
# karena proses ambil variabel dan datanya nyambung
############################
# 6. MSGARCH -> input at
############################
idx.ffnn=5
model.NN[idx.ffnn] = "MSGARCH input at"
ylabel = "return kuadrat"
xlabel="t"

resitrain = resitest = resi = vector()

NNbestresult = bestresult.NN.ARMA.pq
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)

msgarch = fitMSGARCH(data = resitrain, TrainActual = NNbestresult$train$actual^2, TestActual=NNbestresult$test$actual^2, nfore=nfore, 
                     GARCHtype="sGARCH", distribution="norm", nstate=2)

makeplot(msgarch$train$actual, msgarch$train$predict, paste(model.NN[idx.ffnn],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(msgarch$test$actual, msgarch$test$predict, paste(model.NN[idx.ffnn],"Test"),xlabel=xlabel, ylabel=ylabel)


#akurasi
NNresult = list()
NNbestresult = msgarch
for(j in 1:length(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

msgarch.at = msgarch

##### Essential section for MSGARCH-FFNN process clean code #####
msgarch.model = msgarch.at
SR.fit <- ExtractStateFit(msgarch.model$modelfit)

K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=length(resitrain), ncol=K)
voltest = matrix(nrow=length(resitest), ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = resitrain, TrainActual = NNbestresult$train$actual^2, 
                               TestActual=NNbestresult$test$actual^2, nfore=nfore, nstate=2)
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
# 6. MSGARCH-based FFNN -> input at"
############################
idx.ffnn=6
model.NN[idx.ffnn] = "at MSGARCH-FFNN"
msgarch.model = msgarch.at

#get variabel input ML
v = rbind(vtrain.pit,vtest.pit)
colnames(v) = c("v1pit","v2pit")
par(mfrow=c(1,1))
plot(mydata$rv, type="l")
lines(rowSums(v), col="red")
lines(c(msgarch.model$train$predict,msgarch.model$test$predict),col="green")

# form the msgarch data
time = data.NN.ARMA.pq$time
rt2 = data.NN.ARMA.pq$rt^2
base.data = data.frame(time,rt2,v)
data.NN.MSGARCH.at.pit = na.omit(base.data)

#template
data = data.NN.MSGARCH.at.pit
head(data)
NNresult = list()
NNresult = fitNN(data, startTrain, endTrain, endTest, neuron)

NNbestresult = NNresult[[NNresult$opt_idx]]
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(model.NN[idx.ffnn],"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(model.NN[idx.ffnn],"Test"), xlabel = xlabel, ylabel=ylabel)

for(j in 1:length(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

result.NN.MSGARCH.at.pit = NNresult
bestresult.NN.MSGARCH.at.pit = NNbestresult


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
