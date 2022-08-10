################## INI KODINGAN UNTUK CONFERENCE revisi pertama ####################

rm(list = ls(all = TRUE))
library(MSGARCH)

setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML")
source("getDataLuxemburg.R")
source("allfunction.R")
source_python('LSTM_fit.py')

# inisialisasi 
lossfunction = c("MSE")
len.loss = length(lossfunction)
model = vector()
losstrain = matrix(nrow=4, ncol=len.loss)
losstest = matrix(nrow=4,ncol=len.loss)
colnames(losstrain) = lossfunction
rownames(losstrain) = c("ARMA-SVR-MSGARCH","ARMA-SVR-MSGARCH-SVR","ARMA-LSTM-MSGARCH","ARMA-LSTM-MSGARCH-LSTM")
colnames(losstest) = colnames(losstrain)
rownames(losstest) = rownames(losstrain)

#split data train & data test 
dataTrain = mydata%>%filter(date >= as.Date(startTrain) & date <= as.Date(endTrain) )
dataTest = mydata%>%filter(date > as.Date(endTrain) & date <= as.Date(endTest) )
nfore = dim(dataTest)[1];nfore

##### get lag signifikan ARMA #####
batas = 1.96/sqrt(length(dataTrain$return)-1)
optARMAlag = getOptLagARMA(dataTrain$return, maxlag = maxlag, batas = batas, alpha = alpha)

######## SVR #########
# inisialisasi
# kernel = 'radial'
# tune_C = TRUE
# tune_gamma = FALSE
# tune_eps = FALSE

source("allfunction.R")
############################
# 0. Model ARMA-SVR
############################
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
title = "ARMA-SVR"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
##### end of Model ARMA #####


############################
# 1. ARMA-SVR-MSGARCH
############################
m=1
model[m] = "ARMA-SVR-MSGARCH"
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
title = model[m]
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA.MSGARCH
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"),xlabel=xlabel, ylabel=ylabel)

# calculate the prediction error
for(j in 1:len.loss){
  losstrain[m,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest[m,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}
losstrain
losstest

source("allfunction.R")
############################
# 2. ARMA-SVR-MSGARCH-SVR
############################
m=2
model[m] = "ARMA-SVR-MSGARCH-SVR"
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
at2 = resi^2
plot(at2, type="l")
time = data.SVR.ARMA$time
base.data = data.frame(time,y=at2,v)
data.SVR.ARMA.MSGARCH.SVR = na.omit(base.data)

# fit SVR model
data = data.SVR.ARMA.MSGARCH.SVR
head(data)
result.SVR.ARMA.MSGARCH.SVR = fitSVR(data, startTrain, endTrain, endTest, is.vol=TRUE, transform = "sq")

# get best result
SVRresult = list()
trainactual = vector()
testactual = vector()
SVRresult = result.SVR.ARMA.MSGARCH.SVR
data = data.SVR.ARMA

trainactual = bestresult.SVR.ARMA$train$actual^2
testactual = bestresult.SVR.ARMA$test$actual^2
rt.hat.train = bestresult.SVR.ARMA$train$predict
rt.hat.test = bestresult.SVR.ARMA$test$predict

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
  losstrain[m,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest[m,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}
losstrain
losstest

par(mfrow=c(1,1))
plot(bestresult.SVR.ARMA.MSGARCH.SVR$test$actual, type="l", lwd=2, ylab="Volatility", xlab="t")
lines(bestresult.SVR.ARMA.MSGARCH$test$predict, col="green", lwd=2)
lines(bestresult.SVR.ARMA.MSGARCH.SVR$test$predict, col="red", lwd=2)
legend("topright",c("Realized Volatility","MSGARCH","MSGARCH-SVR"),col=c("black","green","red"), 
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5, inset=0.1)

############################
# 0. Model ARMA(p,q)-LSTM
############################
#inisialisasi
epoch = as.integer(100000)
node_hidden = c(1:20)
layer_hidden = 1
linear_output = FALSE
allow_negative = FALSE
n.neuron = length(node_hidden)

ylabel = "return"
xlabel = "time index"
base.data = data.frame(time=mydata$date,y=mydata$return)
head(base.data)

ylabel = "return"
xlabel = "t"
base.data = data.frame(time=mydata$date,y=mydata$return)
head(base.data)

##### Model AR #####
#get data AR
data.LSTM.AR = makeData(data = base.data, datalag = mydata$return, numlag = optARMAlag$ARlag, lagtype = "rt")
data.LSTM.AR = na.omit(data.LSTM.AR)

# fit LSTM model
title = "AR-LSTM conf"
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
title = "ARMA-LSTM conf"
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

loss = matrix(nrow=n.node, ncol=2)
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
##### end of Model ARMA #####

############################
# 3. Model ARMA-LSTM-MSGARCH
############################
m=3
model[m]= "ARMA-LSTM-MSGARCH"
LSTMbestresult = list()
resitrain = resitest = resi = vector()

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
title = "ARMA-LSTM-MSGARCH conf"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.MSGARCH
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"),xlabel=xlabel, ylabel=ylabel)

#calculate the prediction error
for(j in 1:len.loss){
  losstrain[m,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest[m,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}
losstrain
losstest
############################
# 4.ARMA-LSTM-MSGARCH-LSTM
############################
m=4
model[m] = "ARMA-LSTM-MSGARCH-LSTM"
msgarch.model = result.LSTM.ARMA.MSGARCH

LSTMbestresult = list()
resitrain = resitest = resi = vector()

LSTMbestresult = bestresult.LSTM.ARMA
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
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
title = "ARMA-MSGARCH-LSTM conf"
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

bestresult = list()
bestresult = bestresult.LSTM.ARMA.MSGARCH.LSTM
makeplot(bestresult$train$actual, bestresult$train$predict, paste(model[m],"Train"), xlabel=xlabel, ylabel=ylabel)
makeplot(bestresult$test$actual, bestresult$test$predict, paste(model[m],"Test"), xlabel=xlabel, ylabel=ylabel)

for(j in 1:length(lossfunction)){
  losstrain[m,j] = hitungloss(bestresult$train$actual, bestresult$train$predict, method = lossfunction[j])
  losstest[m,j] = hitungloss(bestresult$test$actual, bestresult$test$predict, method = lossfunction[j])
}
losstrain
losstest


par(mfrow=c(1,1))
##### evaluasi MSE #####
svridx = 2
msgidx = 1
a = losstrain[msgidx,1]
b = losstrain[svridx,1]
pct = (a-b)/a*100
pct

a = losstest[msgidx,1]
b = losstest[svridx,1]
pct = (a-b)/a*100
pct

lstmidx = 4
msgidx = 3
a = losstrain[msgidx,1]
b = losstrain[lstmidx,1]
pct = (a-b)/a*100
pct

a = losstest[msgidx,1]
b = losstest[lstmidx,1]
pct = (a-b)/a*100
pct

##### make all plot #####

# plot probabilitas msgarch
msgarch.model = result.SVR.ARMA.MSGARCH
trainstate = State(object = msgarch.model$modelfit)
par(mfrow=c(3,1))
plot(trainstate, type.prob = "filtered", xlab="t")
plot.new()
title(expression(paste(italic('a'['t,SVR']),' MSGARCH ',"(In-Sample Data)")))

msgarch.model = result.LSTM.ARMA.MSGARCH
trainstate = State(object = msgarch.model$modelfit)
par(mfrow=c(3,1))
plot(trainstate, type.prob = "filtered", xlab="t")
plot.new()
title(expression(paste(italic('a'['t,LSTM']),' MSGARCH ',"(In-Sample Data)")))


# plot hasil prediksi
par(mfrow=c(1,1))
#training
train.actual = bestresult.SVR.ARMA.MSGARCH.SVR$train$actual
train.SVR = bestresult.SVR.ARMA.MSGARCH.SVR$train$predict
train.LSTM = bestresult.LSTM.ARMA.MSGARCH.LSTM$train$predict
train.SVR.ARMA.MSGARCH = bestresult.SVR.ARMA.MSGARCH$train$predict
train.LSTM.ARMA.MSGARCH = bestresult.LSTM.ARMA.MSGARCH$train$predict

#only hybrid
plot(train.actual,type="l", lwd=1, ylab="volatility", xlab="t", ylim=c(min(train.actual),max(train.actual)))
lines(train.SVR,col="green", lwd=1)
lines(train.LSTM,col="blue", lwd=1)
legend("topleft",c("Realized Volatility","MSGARCH-SVR","MSGARCH-LSTM"),col=c("black","green","blue"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
title("In-Sample Data")

#including msgarch
plot(train.actual,type="l", lwd=1, ylab="volatility", xlab="t", ylim=c(min(train.actual),max(train.actual)))
lines(train.LSTM.ARMA.MSGARCH,col="yellow", lwd=1)
lines(train.SVR.ARMA.MSGARCH, col="purple", lwd=1)
lines(train.SVR,col="green", lwd=1)
lines(train.LSTM,col="blue", lwd=1)
legend("topleft",c("Realized Volatility",expression(paste(italic('a'['t,SVR']),' MSGARCH')),
                   "MSGARCH-SVR",expression(paste(italic('a'['t,LSTM']),' MSGARCH')),"MSGARCH-LSTM"),
       col=c("black","purple","green","yellow","blue"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
title("in-Sample Data")


#testing
test.actual = bestresult.SVR.ARMA.MSGARCH.SVR$test$actual
test.SVR = bestresult.SVR.ARMA.MSGARCH.SVR$test$predict
test.LSTM = bestresult.LSTM.ARMA.MSGARCH.LSTM$test$predict
test.SVR.ARMA.MSGARCH = bestresult.SVR.ARMA.MSGARCH$test$predict
test.LSTM.ARMA.MSGARCH = bestresult.LSTM.ARMA.MSGARCH$test$predict

#only hybrid
plot(test.actual,type="l", lwd=1, ylab="volatility", xlab="t", ylim=c(min(train.actual),max(train.actual)))
lines(test.SVR,col="green", lwd=1)
lines(test.LSTM,col="blue", lwd=1)
legend("topright",c("Realized Volatility","MSGARCH-SVR","MSGARCH-LSTM"),col=c("black","green","blue"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5, inset=0.1)
title("Out-of-Sample Data")

#including msgarch
plot(test.actual,type="l", lwd=1, ylab="volatility", xlab="t", ylim=c(min(train.actual),max(train.actual)))
lines(test.SVR.ARMA.MSGARCH, col="purple", lwd=1)
lines(test.SVR,col="green", lwd=1)
lines(test.LSTM.ARMA.MSGARCH,col="yellow", lwd=1)
lines(test.LSTM,col="blue", lwd=1)
legend("topright",c("Realized Volatility",expression(paste(italic('a'['t,SVR']),' MSGARCH')),
                   "MSGARCH-SVR",expression(paste(italic('a'['t,LSTM']),' MSGARCH')),"MSGARCH-LSTM"),
       col=c("black","purple","green","yellow","blue"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5, inset=0.1)
title("Out-of-Sample Data")

# plot each model separately
ntrain = length(train.actual)
ntest = length(test.actual)
actual = c(train.actual,test.actual)
NA.train = rep(NA,1,ntrain)
NA.test = rep(NA,1,ntest)

# ARMA-SVR-MSGARCH
temp.train = c(train.SVR.ARMA.MSGARCH,NA.test)
temp.test = c(NA.train,test.SVR.ARMA.MSGARCH)
plot(actual,type="l", lwd=1, ylab="volatility", xlab="t", ylim=c(min(actual),max(actual)))
lines(temp.train, col="purple", lwd=1)
lines(temp.test, col="red", lwd=1)
legend("topleft",c("Realized Volatility",expression(paste(italic('a'['t,SVR']),' MSGARCH'," In-Sample")),
                   expression(paste(italic('a'['t,SVR']),' MSGARCH'," Out-of-Sample"))),
       col=c("black","purple","red"), lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
title(expression(paste(italic('a'['t,SVR']),' MSGARCH')))

# ARMA_SVR-MSGARCH-SVR
temp.train = c(train.SVR,NA.test)
temp.test = c(NA.train,test.SVR)
plot(actual,type="l", lwd=1, ylab="volatility", xlab="t", ylim=c(min(actual),max(actual)))
lines(temp.train, col="green", lwd=1)
lines(temp.test, col="red", lwd=1)
legend("topleft",c("Realized Volatility",expression(paste('MSGARCH-SVR'," In-Sample")),
                   expression(paste('MSGARCH-SVR'," Out-of-Sample"))),
       col=c("black","green","red"), lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
title("MSGARCH-SVR")

# ARMA-LSTM-MSGARCH
temp.train = c(train.LSTM.ARMA.MSGARCH,NA.test)
temp.test = c(NA.train,test.LSTM.ARMA.MSGARCH)
plot(actual,type="l", lwd=1, ylab="volatility", xlab="t", ylim=c(min(actual),max(actual)))
lines(temp.train, col="yellow", lwd=1)
lines(temp.test, col="red", lwd=1)
legend("topleft",c("Realized Volatility",expression(paste(italic('a'['t,LSTM']),' MSGARCH'," In-Sample")),
                   expression(paste(italic('a'['t,LSTM']),' MSGARCH'," Out-of-Sample"))),
       col=c("black","yellow","red"), lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
title(expression(paste(italic('a'['t,LSTM']),' MSGARCH')))

# ARMA-LSTM-MSGARCH-LSTM
temp.train = c(train.LSTM,NA.test)
temp.test = c(NA.train,test.LSTM)
plot(actual,type="l", lwd=1, ylab="volatility", xlab="t", ylim=c(min(actual),max(actual)))
lines(temp.train, col="blue", lwd=1)
lines(temp.test, col="red", lwd=1)
legend("topleft",c("Realized Volatility",expression(paste('MSGARCH-LSTM'," In-Sample")),
                   expression(paste('MSGARCH-LSTM'," Out-of-Sample"))),
       col=c("black","blue","red"), lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
title("MSGARCH-LSTM")


# plot MSE LSTM
result = list()
result = result.LSTM.ARMA.MSGARCH.LSTM

trainactual = testactual = rt.hat.train = rt.hat.test = vector()
trainactual = bestresult.LSTM.ARMA$train$actual^2
testactual = bestresult.LSTM.ARMA$test$actual^2
rt.hat.train = bestresult.LSTM.ARMA$train$predict
rt.hat.test = bestresult.LSTM.ARMA$test$predict

losstrain.LSTM = matrix(nrow=n.neuron, ncol=len.loss)
losstest.LSTM = matrix(nrow=n.neuron, ncol=len.loss)
colnames(losstrain.LSTM) = lossfunction
colnames(losstest.LSTM) = colnames(losstrain.LSTM)
rownames(losstrain.LSTM) = paste("Hidden_Node",node_hidden)
rownames(losstest.LSTM) = rownames(losstrain.LSTM)
for(i in 1:n.neuron){
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2
  class(rt.hat.train)
  class(attestpred)
  losstrain.LSTM[i] = hitungloss(trainactual, trainpred, method = "MSE")
  losstest.LSTM[i] = hitungloss(testactual, testpred, method = "MSE")
}
losstrain.LSTM
losstest.LSTM

#MSE training
maxMSE = max(max(losstrain.LSTM),max(losstest.LSTM))
minMSE = min(min(losstrain.LSTM),min(losstest.LSTM))
par(mfrow=c(1,1))
plot(as.ts(losstrain.LSTM[,1]),ylab=paste("MSE"),xlab="Hidden Neuron",lwd=2,axes=F, ylim=c(minMSE, maxMSE*1.1))
box()
axis(side=2,lwd=0.5,cex.axis=0.8,las=2)
axis(side=1,lwd=0.5,cex.axis=0.8,las=0,at=c(1:length(node_hidden)),labels=node_hidden)
lines(losstest.LSTM[,1],col="red",lwd=2)
title(main="MSE MSGARCH-LSTM")
legend("topleft",c("In-Sample Data","Out-of-Sample Data"),col=c("black","red"), lwd=2,cex=0.7,bty = "n", y.intersp=1.5)

which.min(losstrain.LSTM)
which.min(losstest.LSTM)

# simpan data
save(data.SVR.AR, data.SVR.ARMA, data.SVR.ARMA.MSGARCH.SVR,
    data.LSTM.AR, data.LSTM.ARMA, data.LSTM.ARMA.MSGARCH.LSTM,file = "data/revisi_datauji_conference.RData")
save(result.SVR.AR, result.SVR.ARMA, result.SVR.ARMA.MSGARCH, result.SVR.ARMA.MSGARCH.SVR,
    result.LSTM.AR, result.LSTM.ARMA, result.LSTM.ARMA.MSGARCH, result.LSTM.ARMA.MSGARCH.LSTM,
     file="data/revisi_result_conference.RData")
save(bestresult.SVR.AR, bestresult.SVR.ARMA, bestresult.SVR.ARMA.MSGARCH, bestresult.SVR.ARMA.MSGARCH.SVR,
    bestresult.LSTM.AR, bestresult.LSTM.ARMA, bestresult.LSTM.ARMA.MSGARCH, bestresult.LSTM.ARMA.MSGARCH.LSTM,
     file="data/revisi_bestresult_conference.RData")
save(losstrain.LSTM, losstest.LSTM, losstrain, losstest, file="data/revisi_loss_conference.RData")


# ##### forecast #####
# n.ahead = 5
# K=2
# 
# # SVR
# SVR.SR = ExtractStateFit(msgarch.svr$modelfit)
# SVR.vol.fore = predict(object=msgarch.svr$modelfit,nahead=n.ahead)$vol
# msgarch.SR = list(0)
# SVR.vol.fore.k = matrix(nrow=n.ahead, ncol=K)
# for(k in 1:K){
#   SVR.vol.fore.k[,k] = predict(object=SVR.SR[[k]],nahead=n.ahead)$vol^2
# }
# for(i in 1:n.ahead){
#   SVR.Pfore = State(object = msgarch.svr$modelspec, par = msgarch.svr$modelfit$par, data=SVR.vol.fore)
#   SVR.predProb.fore = SVR.Pfore$PredProb
#   SVR.vfore.pit = SVR.predProb.fore[-1,1,] * SVR.vol.fore.k
# }
# colnames(SVR.vfore.pit) = c("v1pit","v2pit")
# 
# SVR.fit = result.ARMA.SVR.MSGARCH.SVR$model.fit
# x.fore = SVR.vfore.pit
# data.fore <- data.frame(x=x.fore)
# SVR.fore = predict(SVR.fit, data.fore)
# plot(rowSums(SVR.vfore.pit), type="l")
# par(mfrow=c(1,1))
# 
# plot(SVR.vol.fore^2, type="l")
# lines(SVR.vol.fore.k[,1], col="red")
# lines(SVR.vol.fore.k[,2], col="red")
# lines(SVR.vol.fore^2, col="red")
# SVR.fore
# plot(SVR.fore, type="l")
# 
# #LSTM
# LSTM.SR = ExtractStateFit(msgarch.lstm$modelfit)
# LSTM.vol.fore = predict(object=msgarch.lstm$modelfit,nahead=n.ahead)$vol
# msgarch.SR = list(0)
# LSTM.vol.fore.k = matrix(nrow=n.ahead, ncol=K)
# for(k in 1:K){
#   LSTM.vol.fore.k[,k] = predict(object=LSTM.SR[[k]],nahead=n.ahead)$vol^2
# }
# for(i in 1:n.ahead){
#   LSTM.Pfore = State(object = msgarch.lstm$modelspec, par = msgarch.lstm$modelfit$par, data=LSTM.vol.fore)
#   LSTM.predProb.fore = LSTM.Pfore$PredProb
#   LSTM.vfore.pit = LSTM.predProb.fore[-1,1,] * LSTM.vol.fore.k
# }
# colnames(LSTM.vfore.pit) = c("v1pit","v2pit")
# LSTM.fit = bestresult.ARMA.LSTM.MSGARCH.LSTM
# source_python('LSTM_forecast.py')
# LSTM.fore = forecastLSTM(LSTM.fit$model,LSTM.vfore.pit)
