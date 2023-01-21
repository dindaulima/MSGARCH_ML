rm(list = ls(all = TRUE))
setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML/")
source("getDataLuxemburg.R")
source("allfunction.R")


source_python('LSTM_fit.py')

#inisialisasi
neuron = c(1:20)
n.neuron = length(neuron)
LSTMmodel.path = "final result/model LSTM/"

#split data train & data test 
dataTrain = mydata%>%filter(date >= as.Date(startTrain) & date <= as.Date(endTrain) )
dataTest = mydata%>%filter(date > as.Date(endTrain) & date <= as.Date(endTest) )
nfore = dim(dataTest)[1];nfore

####### analisis deskripsi #######
ggplot( data = mydata, aes( date, close )) + geom_line() + scale_x_date(date_labels = "%d-%m-%Y") +theme_bw() +
  xlab("date") + ylab("closing price")

ggplot( data = mydata, aes( date, return )) + geom_line() + scale_x_date(date_labels = "%d-%m-%Y") +theme_bw() +
  xlab("date") + ylab("log return (%)")

ggplot( data = mydata, aes( date, rv )) + geom_line() + scale_x_date(date_labels = "%d-%m-%Y") +theme_bw() +
  xlab("date") + ylab("realized volatility")

ggplot( data = mydata, aes( date, rv )) + geom_line() + scale_x_date(date_labels = "%d-%m-%Y") +theme_bw() +
  xlab("date") + ylab("realisasi volatilitas (%)")

cat("Harga sukuk terendah",min(mydata$close),"pada",as.character(as.Date(mydata$date[which.min(mydata$close)])),"\n")
cat("Harga sukuk tertinggi",max(mydata$close),"pada",as.character(as.Date(mydata$date[which.max(mydata$close)])),"\n")
cat("Return sukuk terendah",min(mydata$return),"pada",as.character(as.Date(mydata$date[which.min(mydata$return)])),"\n")
cat("Return sukuk tertinggi",max(mydata$return),"pada",as.character(as.Date(mydata$date[which.max(mydata$return)])),"\n")
desc.pt = summary(mydata$close)
desc.pt = c(desc.pt, skew=skewness(mydata$close), kurtosis=kurtosis(mydata$close))
desc.pt
desc.rt = summary(mydata$return)
desc.rt = c(desc.rt, skew=skewness(mydata$return), kurtosis=kurtosis(mydata$return))
desc.rt
####### end of analisis deskripsi #######
####### Pemodelan ARMA ####### 
source("fitARMA.R")

paramAR = rep(0,1,maxlag)
paramMA = rep(0,1,maxlag)
paramAR[optARMAlag$ARlag] = NA
paramMA[optARMAlag$MAlag] = NA
armamodel = arima(dataTrain$return,order=c(maxlag,0,maxlag), include.mean=FALSE, fixed=c(paramAR,paramMA))
coeftest(armamodel)
AIC(armamodel)
ujiljungbox(residuals(armamodel))
ujinormal(residuals(armamodel))
dataARIMA = data.frame(dataTrain$return, fitted(armamodel),residuals(armamodel))

ARMA.trainactual = dataTrain$return
ARMA.testactual = dataTest$return
ARMA.trainpred = fitted(armamodel)
ARMA.testpred = predict(armamodel, n.ahead = nfore)$pred
ARMA.testpred = as.vector(ARMA.testpred)

#### grafik perbandingan ARMA ####
par(mfrow=c(1,1))
title = "ARMA model"
xlabel = "t"
ylabel = "return (%)"
makeplot(ARMA.trainactual, ARMA.trainpred, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(ARMA.testactual, ARMA.testpred, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
min(ARMA.trainpred)
min(ARMA.testpred)

#single plot
actual = c(ARMA.trainactual,ARMA.testactual)
n.actual = length(actual)
train = c(ARMA.trainpred,rep(NA,1,length(ARMA.testpred)))
test = c(rep(NA,1,length(ARMA.trainpred)),ARMA.testpred)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
####### end of pemodelan ARMA #######

####### load ML result #######
#load FFNN result
load("final result/result_NN.RData")
load("final result/datauji_NN.RData")
load("final result/bestresult_NN.RData")
#load SVR result
load("final result/result_SVR_sq.RData")
load("final result/datauji_SVR_sq.RData")
load("final result/bestresult_SVR_sq.RData")
#load LSTM result
load("final result/result_LSTM.RData")
load("final result/datauji_LSTM.RData")
load("final result/bestresult_LSTM.RData")
####### end of load ML result #######

####### Pemodelan ARMA-ML #######
#####cek struktur input ARMA-ML #####
head(data.NN.ARMA)
head(data.SVR.ARMA)
head(data.LSTM.ARMA)

##### detail ARMA-FFNN ##### 
#### AR-FFNN ####
result = list()
result = result.NN.AR
data = data.NN.AR
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxNN.AR = which.min(loss$MSEtest);opt_idxNN.AR
lossNN.AR = loss
rownames(lossNN.AR) = paste('Neuron',neuron)
lossNN.AR

#bobot & arsitektur NN
plot(result.NN.AR[[opt_idxNN.AR]]$model_NN)
plot(result.NN.AR[[opt_idxNN.AR]]$model_NN, show.weights = FALSE)
result.NN.AR[[opt_idxNN.AR]]$model_NN$result.matrix


#### ARMA-FFNN ####
result = list()
result = result.NN.ARMA
data = data.NN.ARMA
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxNN.ARMA = which.min(loss$MSEtest);opt_idxNN.ARMA
lossNN.ARMA = loss
rownames(lossNN.ARMA) = paste('Neuron',neuron)
lossNN.ARMA

#bobot & arsitektur NN
plot(result.NN.ARMA[[opt_idxNN.ARMA]]$model_NN)
plot(result.NN.ARMA[[opt_idxNN.ARMA]]$model_NN, show.weights = FALSE)
result.NN.ARMA[[opt_idxNN.ARMA]]$model_NN$result.matrix

#### grafik perbandingan ARMA-FFNN ####
title = "mean model FFNN"
xlabel = "t"
ylabel = "return (%)"
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(NNbestresult$train$actual,NNbestresult$test$actual)
n.actual = length(actual)
train = c(NNbestresult$train$predict,rep(NA,1,length(NNbestresult$test$predict)))
test = c(rep(NA,1,length(NNbestresult$train$predict)),NNbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of detail ARMA-FFNN ##### 

##### detail ARMA-SVR ##### 
#### AR-SVR ####
result = list()
result = result.SVR.AR
data = data.SVR.AR
result$model.fit
result$w
result$b

#### ARMA-SVR ####
result = list()
result = result.SVR.ARMA
data = data.SVR.ARMA
result$model.fit
result$w
result$b

#### grafik perbandingan ARMA-SVR ####
title = "mean model SVR"
xlabel = "t"
ylabel = "return (%)"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA
par(mfrow=c(1,1))
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(SVRbestresult$train$actual,SVRbestresult$test$actual)
n.actual = length(actual)
train = c(SVRbestresult$train$predict,rep(NA,1,length(SVRbestresult$test$predict)))
test = c(rep(NA,1,length(SVRbestresult$train$predict)),SVRbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of detail ARMA-SVR ##### 

##### detail ARMA-LSTM ##### 
#### AR-LSTM ####
result = list()
result = result.LSTM.AR
data = data.LSTM.AR
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxLSTM.AR = which.min(loss$MSEtest);opt_idxLSTM.AR
lossLSTM.AR = loss
rownames(lossLSTM.AR) = paste('Neuron',neuron)
lossLSTM.AR
# source_python('LSTM_fit.py')

#### bobot & arsitektur AR-LSTM ####
nameLSTM.AR = result.LSTM.AR$model_filename[opt_idxLSTM.AR]
modLSTM.AR = loadmodel(nameLSTM.AR,opt_idxLSTM.AR,LSTMmodel.path)
modLSTM.AR
head(data.LSTM.AR)
var = colnames(data.LSTM.AR)[c(-1,-2)]
modtemp = modLSTM.AR
wi = matrix(paste(modtemp$W_i,var),ncol=ncol(modtemp$W_i),nrow=length(var))
wf = matrix(paste(modtemp$W_f,var),ncol=ncol(modtemp$W_f),nrow=length(var))
wc = matrix(paste(modtemp$W_c,var),ncol=ncol(modtemp$W_c),nrow=length(var))
wo = matrix(paste(modtemp$W_o,var),ncol=ncol(modtemp$W_o),nrow=length(var))
t(wi)
t(wf)
t(wc)
t(wo)

#### ARMA-LSTM ####
result = list()
result = result.LSTM.ARMA
data = data.LSTM.ARMA
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxLSTM.ARMA = which.min(loss$MSEtest);opt_idxLSTM.ARMA
lossLSTM.ARMA = loss
rownames(lossLSTM.ARMA) = paste('Neuron',neuron)
lossLSTM.ARMA

#### bobot & arsitektur ARMA-LSTM ####
nameLSTM.ARMA = result.LSTM.ARMA$model_filename[opt_idxLSTM.ARMA]
modLSTM.ARMA = loadmodel(nameLSTM.ARMA,opt_idxLSTM.ARMA,LSTMmodel.path)
modLSTM.ARMA
var = colnames(data.LSTM.ARMA)[c(-1,-2)]

modtemp = modLSTM.ARMA
wi = matrix(paste(modtemp$W_i,var),ncol=ncol(modtemp$W_i),nrow=length(var))
wf = matrix(paste(modtemp$W_f,var),ncol=ncol(modtemp$W_f),nrow=length(var))
wc = matrix(paste(modtemp$W_c,var),ncol=ncol(modtemp$W_c),nrow=length(var))
wo = matrix(paste(modtemp$W_o,var),ncol=ncol(modtemp$W_o),nrow=length(var))

neu = paste('neuron',seq(1,opt_idxLSTM.ARMA,1))
ui = matrix(paste(modtemp$U_i,neu),ncol=ncol(modtemp$U_i),nrow=length(neu))
uf = matrix(paste(modtemp$U_f,neu),ncol=ncol(modtemp$U_f),nrow=length(neu))
uc = matrix(paste(modtemp$U_c,neu),ncol=ncol(modtemp$U_c),nrow=length(neu))
uo = matrix(paste(modtemp$U_o,neu),ncol=ncol(modtemp$U_o),nrow=length(neu))

t(ui)
modtemp$b_i
t(wi)

t(uf)
modtemp$b_f
t(wf)

t(uc)
modtemp$b_c
t(wc)

t(uo)
modtemp$b_o
t(wo)

#### grafik perbandingan ARMA-LSTM ####
title = "mean model LSTM"
xlabel = "t"
ylabel = "return (%)"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(LSTMbestresult$train$actual,LSTMbestresult$test$actual)
n.actual = length(actual)
train = c(LSTMbestresult$train$predict,rep(NA,1,length(LSTMbestresult$test$predict)))
test = c(rep(NA,1,length(LSTMbestresult$train$predict)),LSTMbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of detail ARMA-LSTM ##### 

####### end of pemodelan ARMA-ML #######

#### Uji LM residual ARMA-ML ####
#ARMA-FFNN
bestresult = list()
resitrain = resitest = resi = vector()
bestresult = bestresult.NN.ARMA
resitrain = bestresult$train$actual - bestresult$train$predict
resitest = bestresult$test$actual - bestresult$test$predict
resi = c(resitrain,resitest)
atNN = resi
LMtest(atNN)

#ARMA-SVR
bestresult = list()
resitrain = resitest = resi = vector()
bestresult = bestresult.SVR.ARMA
resitrain = bestresult$train$actual - bestresult$train$predict
resitest = bestresult$test$actual - bestresult$test$predict
resi = c(resitrain,resitest)
atSVR = resi
LMtest(atSVR)

#ARMA-LSTM
bestresult = list()
resitrain = resitest = resi = vector()
bestresult = bestresult.LSTM.ARMA
resitrain = bestresult$train$actual - bestresult$train$predict
resitest = bestresult$test$actual - bestresult$test$predict
resi = c(resitrain,resitest)
atLSTM = resi
LMtest(atLSTM)

# Chi-square tabel
qchisq(alpha, 1)
#### end of Uji LM residual ARMA-ML ####

####### Pemodelan GARCH #######
## cuplik data ARMA-GARCH
head(data.NN.GARCH)
head(data.SVR.GARCH)
head(data.LSTM.GARCH)

#### identifikasi model GARCH ####
rt2 = mydata$return^2
par(mfrow=c(1,2))
acf.resikuadrat = acf(rt2, lag.max = maxlag, type = "correlation")
acf.resikuadrat <- acf.resikuadrat$acf[2:(maxlag+1)]
pacf.resikuadrat = pacf(rt2, lag.max = maxlag)
pacf.resikuadrat <- pacf.resikuadrat$acf[1:maxlag]
batas.rt2 = 1.96/sqrt(length(rt2)-1)
optlag = getLagSignifikan(rt2, maxlag = maxlag, batas = batas.rt2, alpha = alpha, na=FALSE)
chisq.linear = terasvirta.test(ts(rt2), lag = min(optlag$PACFlag), type = "Chisq");chisq.linear
if(chisq.linear$p.value<alpha){
  cat("Dengan Statistik uji Chisquare, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji Chisquare, Gagal Tolak H0, data linear")
}
#### detail GARCH-FFNN ####
#### ARCH-FFNN ####
result = list()
result = result.NN.ARCH
data = data.NN.ARCH
head(data)
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxNN.ARCH = which.min(loss$MSEtest);opt_idxNN.ARCH
lossNN.ARCH = loss
rownames(lossNN.ARCH) = paste('Neuron',neuron)
lossNN.ARCH

#bobot & arsitektur NN
plot(result.NN.ARCH[[opt_idxNN.ARCH]]$model_NN)
plot(result.NN.ARCH[[opt_idxNN.ARCH]]$model_NN, show.weights = FALSE)
result.NN.ARCH[[opt_idxNN.ARCH]]$model_NN$result.matrix


#### GARCH-FFNN ####
result = list()
result = result.NN.GARCH
data = data.NN.GARCH
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxNN.GARCH = which.min(loss$MSEtest);opt_idxNN.GARCH
lossNN.GARCH = loss
rownames(lossNN.GARCH) = paste('Neuron',neuron)
lossNN.GARCH

#bobot & arsitektur NN
plot(result.NN.GARCH[[opt_idxNN.GARCH]]$model_NN)
plot(result.NN.GARCH[[opt_idxNN.GARCH]]$model_NN, show.weights = FALSE)
result.NN.GARCH[[opt_idxNN.GARCH]]$model_NN$result.matrix

#### grafik perbandingan GARCH-FFNN ####
title = "FFNN-GARCH"
xlabel = "t"
ylabel = "realisasi volatilitas (%)"
NNbestresult = list()
NNbestresult = bestresult.NN.GARCH
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(NNbestresult$train$actual,NNbestresult$test$actual)
n.actual = length(actual)
train = c(NNbestresult$train$predict,rep(NA,1,length(NNbestresult$test$predict)))
test = c(rep(NA,1,length(NNbestresult$train$predict)),NNbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)

#### end of detail GARCH-FFNN ####

##### detail GARCH-SVR ##### 
#### ARCH-SVR ####
result = list()
result = result.SVR.ARCH
data = data.SVR.ARCH
result$model.fit
result$w
result$b

#### GARCH-SVR ####
result = list()
result = result.SVR.GARCH
data = data.SVR.GARCH
result$model.fit
result$w
result$b

#### grafik perbandingan GARCH-SVR ####
title = "GARCH SVR"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.GARCH
par(mfrow=c(1,1))
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(SVRbestresult$train$actual,SVRbestresult$test$actual)
n.actual = length(actual)
train = c(SVRbestresult$train$predict,rep(NA,1,length(SVRbestresult$test$predict)))
test = c(rep(NA,1,length(SVRbestresult$train$predict)),SVRbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of detail ARMA-SVR ##### 

##### detail GARCH-LSTM ##### 
#### ARCH-LSTM ####
result = list()
result = result.LSTM.ARCH
data = data.LSTM.ARCH
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxLSTM.ARCH = which.min(loss$MSEtest);opt_idxLSTM.ARCH
lossLSTM.ARCH = loss
rownames(lossLSTM.ARCH) = paste('Neuron',neuron)
lossLSTM.ARCH
source_python('LSTM_fit.py')

####bobot & arsitektur ARCH-LSTM ####
nameLSTM.ARCH = result.LSTM.ARCH$model_filename[opt_idxLSTM.ARCH]
modLSTM.ARCH = loadmodel(nameLSTM.ARCH,opt_idxLSTM.ARCH,LSTMmodel.path)
modLSTM.ARCH
head(data.LSTM.ARCH)
var = colnames(data.LSTM.ARCH)[c(-1,-2)]
modtemp = modLSTM.ARCH
wi = matrix(paste(modtemp$W_i,var),ncol=ncol(modtemp$W_i),nrow=length(var))
wf = matrix(paste(modtemp$W_f,var),ncol=ncol(modtemp$W_f),nrow=length(var))
wc = matrix(paste(modtemp$W_c,var),ncol=ncol(modtemp$W_c),nrow=length(var))
wo = matrix(paste(modtemp$W_o,var),ncol=ncol(modtemp$W_o),nrow=length(var))
t(wi)
t(wf)
t(wc)
t(wo)

#### GARCH-LSTM ####
result = list()
result = result.LSTM.GARCH
data = data.LSTM.GARCH
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxLSTM.GARCH = which.min(loss$MSEtest);opt_idxLSTM.GARCH
lossLSTM.GARCH = loss
rownames(lossLSTM.GARCH) = paste('Neuron',neuron)
lossLSTM.GARCH

#### bobot & arsitektur GARCH-LSTM ####
nameLSTM.GARCH = result.LSTM.GARCH$model_filename[opt_idxLSTM.GARCH]
modLSTM.GARCH = loadmodel(nameLSTM.GARCH,opt_idxLSTM.GARCH,LSTMmodel.path)
modLSTM.GARCH
var = colnames(data.LSTM.GARCH)[c(-1,-2)]
modtemp = modLSTM.GARCH
wi = matrix(paste(modtemp$W_i,var),ncol=ncol(modtemp$W_i),nrow=length(var))
wf = matrix(paste(modtemp$W_f,var),ncol=ncol(modtemp$W_f),nrow=length(var))
wc = matrix(paste(modtemp$W_c,var),ncol=ncol(modtemp$W_c),nrow=length(var))
wo = matrix(paste(modtemp$W_o,var),ncol=ncol(modtemp$W_o),nrow=length(var))
t(wi)
t(wf)
t(wc)
t(wo)
neu = paste('neuron',seq(1,opt_idxLSTM.GARCH,1))
ui = matrix(paste(modtemp$U_i,neu),ncol=ncol(modtemp$U_i),nrow=length(neu))
uf = matrix(paste(modtemp$U_f,neu),ncol=ncol(modtemp$U_f),nrow=length(neu))
uc = matrix(paste(modtemp$U_c,neu),ncol=ncol(modtemp$U_c),nrow=length(neu))
uo = matrix(paste(modtemp$U_o,neu),ncol=ncol(modtemp$U_o),nrow=length(neu))
t(ui)
t(uf)
t(uc)
t(uo)

#### grafik perbandingan GARCH-LSTM ####
title = "GARCH LSTM"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.GARCH
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(LSTMbestresult$train$actual,LSTMbestresult$test$actual)
n.actual = length(actual)
train = c(LSTMbestresult$train$predict,rep(NA,1,length(LSTMbestresult$test$predict)))
test = c(rep(NA,1,length(LSTMbestresult$train$predict)),LSTMbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of detail GARCH-LSTM ##### 

####### Pemodelan ARMA-GARCH #######
head(data.NN.ARMA.GARCH)
head(data.SVR.ARMA.GARCH)
head(data.LSTM.ARMA.GARCH)
##### identifikasi input ARMA-GARCH #####
#### identifikasi ARMA-FFNN-GARCH #### 
par(mfrow=c(1,2))
atNN2 = atNN^2
acf.resikuadrat = acf(atNN2, lag.max = maxlag, type = "correlation")
acf.resikuadrat <- acf.resikuadrat$acf[2:(maxlag+1)]
pacf.resikuadrat = pacf(atNN2, lag.max = maxlag)
pacf.resikuadrat <- pacf.resikuadrat$acf[1:maxlag]
batas.atNN2 = 1.96/sqrt(length(atNN2)-1)
optlag = getLagSignifikan(atNN2, maxlag = maxlag, batas = batas.atNN2, alpha = alpha, na=FALSE)
chisq.linear = terasvirta.test(ts(atNN2), lag = min(optlag$PACFlag), type = "Chisq");chisq.linear
if(chisq.linear$p.value<alpha){
  cat("Dengan Statistik uji Chisquare, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji Chisquare, Gagal Tolak H0, data linear")
}

#### identifikasi ARMA-SVR-GARCH #### 
par(mfrow=c(1,2))
atSVR2 = atSVR^2
acf.resikuadrat = acf(atSVR2, lag.max = maxlag, type = "correlation")
acf.resikuadrat <- acf.resikuadrat$acf[2:(maxlag+1)]
pacf.resikuadrat = pacf(atSVR2, lag.max = maxlag)
pacf.resikuadrat <- pacf.resikuadrat$acf[1:maxlag]
batas.atSVR2 = 1.96/sqrt(length(atNN2)-1)
optlag = getLagSignifikan(atSVR2, maxlag = maxlag, batas = batas.atSVR2, alpha = alpha, na=FALSE)
chisq.linear = terasvirta.test(ts(atSVR2), lag = min(optlag$PACFlag), type = "Chisq");chisq.linear
if(chisq.linear$p.value<alpha){
  cat("Dengan Statistik uji Chisquare, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji Chisquare, Gagal Tolak H0, data linear")
}
head(data.SVR.ARMA.GARCH)

#### identifikasi ARMA-LSTM-GARCH #### 
par(mfrow=c(1,2))
atLSTM2 = atLSTM^2
acf.resikuadrat = acf(atLSTM2, lag.max = maxlag, type = "correlation")
acf.resikuadrat <- acf.resikuadrat$acf[2:(maxlag+1)]
pacf.resikuadrat = pacf(atLSTM2, lag.max = maxlag)
pacf.resikuadrat <- pacf.resikuadrat$acf[1:maxlag]
batas.atLSTM2 = 1.96/sqrt(length(atLSTM2)-1)
optlag = getLagSignifikan(atLSTM2, maxlag = maxlag, batas = batas.atLSTM2, alpha = alpha, na=FALSE)
chisq.linear = terasvirta.test(ts(atLSTM2), lag = min(optlag$PACFlag), type = "Chisq");chisq.linear
if(chisq.linear$p.value<alpha){
  cat("Dengan Statistik uji Chisquare, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji Chisquare, Gagal Tolak H0, data linear")
}

#### detail ARMA-GARCH-FFNN ####
#### ARMA-ARCH-FFNN ####
result = list()
trainactual = vector()
testactual = vector()

result = result.NN.ARMA.ARCH
data = data.NN.ARMA
t.all = nrow(data)
n.lag = t.all-nrow(data.NN.ARMA.ARCH)

trainactual = (data$y[(n.lag+1):(t.all-nfore)])^2
testactual = (data$y[(t.all-nfore+1):t.all])^2
rt.hat.train = bestresult.NN.ARMA$train$predict[-c(1:n.lag)]
rt.hat.test = bestresult.NN.ARMA$test$predict
length(trainactual)
length(rt.hat.train)
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  #at^2.hat
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  #rt^2 = (rt.hat + at.hat)^2
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2
  
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxNN.ARMA.ARCH = which.min(loss$MSEtest);opt_idxNN.ARMA.ARCH
lossNN.ARMA.ARCH = loss
rownames(lossNN.ARMA.ARCH) = paste('Neuron',neuron)
lossNN.ARMA.ARCH

#bobot & arsitektur NN
plot(result.NN.ARMA.ARCH[[opt_idxNN.ARMA.ARCH]]$model_NN)
plot(result.NN.ARMA.ARCH[[opt_idxNN.ARMA.ARCH]]$model_NN, show.weights = FALSE)
result.NN.ARMA.ARCH[[opt_idxNN.ARMA.ARCH]]$model_NN$result.matrix

# test best and min loss
test = data.frame(best = bestresult.NN.ARMA.ARCH$train$predict, 
                  all=(rt.hat.train+sqrt(result.NN.ARMA.ARCH[[opt_idxNN.ARMA.ARCH]]$train))^2)
View(test)


#### ARMA-GARCH-FFNN ####
result = list()
trainactual = vector()
testactual = vector()

result = result.NN.ARMA.GARCH
data = data.NN.ARMA
t.all = nrow(data)
n.lag = t.all-nrow(data.NN.ARMA.GARCH)

trainactual = (data$y[(n.lag+1):(t.all-nfore)])^2
testactual = (data$y[(t.all-nfore+1):t.all])^2
rt.hat.train = bestresult.NN.ARMA$train$predict[-c(1:n.lag)]
rt.hat.test = bestresult.NN.ARMA$test$predict
length(trainactual)
length(rt.hat.train)
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  #at^2.hat
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  
  #rt^2 = (rt.hat + at.hat)^2
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2
  
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxNN.ARMA.GARCH = which.min(loss$MSEtest);opt_idxNN.ARMA.GARCH
lossNN.ARMA.GARCH = loss
rownames(lossNN.ARMA.GARCH) = paste('Neuron',neuron)
lossNN.ARMA.GARCH
#the real reason why loss$MSEtest[9] < loss$MSEtest[7]
sprintf("%10.15f", loss$MSEtest[7])
sprintf("%10.15f", loss$MSEtest[9])

#bobot & arsitektur NN
plot(result.NN.ARMA.GARCH[[opt_idxNN.ARMA.GARCH]]$model_NN)
plot(result.NN.ARMA.GARCH[[opt_idxNN.ARMA.GARCH]]$model_NN, show.weights = FALSE)
result.NN.ARMA.GARCH[[opt_idxNN.ARMA.GARCH]]$model_NN$result.matrix

# test best and min loss
test = data.frame(best = bestresult.NN.ARMA.GARCH$train$predict, 
                  all=(rt.hat.train+sqrt(result.NN.ARMA.GARCH[[opt_idxNN.ARMA.GARCH]]$train))^2)
View(test)

#### grafik perbandingan ARMA-GARCH-FFNN ####
title = "ARMA-GARCH-FFNN"
# xlabel = "t"
# ylabel = "realisasi volatilitas (%)"
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA.GARCH

par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(NNbestresult$train$actual,NNbestresult$test$actual)
n.actual = length(actual)
train = c(NNbestresult$train$predict,rep(NA,1,length(NNbestresult$test$predict)))
test = c(rep(NA,1,length(NNbestresult$train$predict)),NNbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)

#### end of detail ARMA-GARCH-FFNN ####

##### detail ARMA-GARCH-SVR ##### 
#### ARMA-ARCH-SVR ####
result = list()
result = result.SVR.ARMA.ARCH
data = data.SVR.ARMA.ARCH
result$model.fit
result$w
result$b

#### ARMA-GARCH-SVR ####
result = list()
result = result.SVR.ARMA.GARCH
data = data.SVR.ARMA.GARCH
result$model.fit
result$w
result$b

#### grafik perbandingan ARMA-GARCH-SVR ####
title = "ARMA-GARCH SVR"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA.GARCH
par(mfrow=c(1,1))
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(SVRbestresult$train$actual,SVRbestresult$test$actual)
n.actual = length(actual)
train = c(SVRbestresult$train$predict,rep(NA,1,length(SVRbestresult$test$predict)))
test = c(rep(NA,1,length(SVRbestresult$train$predict)),SVRbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of detail ARMA-SVR ##### 

##### detail ARMA-GARCH-LSTM ##### 
#### ARMA-ARCH-LSTM ####
result = list()
trainactual = vector()
testactual = vector()

result = result.LSTM.ARMA.ARCH
data = data.LSTM.ARMA
t.all = nrow(data)
n.lag = t.all-nrow(data.LSTM.ARMA.ARCH)

trainactual = (data$y[(n.lag+1):(t.all-nfore)])^2
testactual = (data$y[(t.all-nfore+1):t.all])^2
rt.hat.train = bestresult.LSTM.ARMA$train$predict[-c(1:n.lag)]
rt.hat.test = bestresult.LSTM.ARMA$test$predict

loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  #at^2.hat
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  #rt^2 = (rt.hat + at.hat)^2
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2

  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxLSTM.ARMA.ARCH = which.min(loss$MSEtest);opt_idxLSTM.ARMA.ARCH
lossLSTM.ARMA.ARCH = loss
rownames(lossLSTM.ARMA.ARCH) = paste('Neuron',neuron)
lossLSTM.ARMA.ARCH
source_python('LSTM_fit.py')


#### bobot & arsitektur ARMA-ARCH-LSTM ####
nameLSTM.ARMA.ARCH = result.LSTM.ARMA.ARCH$model_filename[opt_idxLSTM.ARMA.ARCH]
modLSTM.ARMA.ARCH = loadmodel(nameLSTM.ARMA.ARCH,opt_idxLSTM.ARMA.ARCH, LSTMmodel.path)
modLSTM.ARMA.ARCH
head(data.LSTM.ARMA.ARCH)
var = colnames(data.LSTM.ARMA.ARCH)[c(-1,-2)]
modtemp = modLSTM.ARMA.ARCH
neu = paste('h',seq(1,opt_idxLSTM.ARMA.ARCH,1))
quot1 = paste(modtemp$Wx,neu)
wi = matrix(paste(modtemp$W_i,var),ncol=ncol(modtemp$W_i),nrow=length(var))
wf = matrix(paste(modtemp$W_f,var),ncol=ncol(modtemp$W_f),nrow=length(var))
wc = matrix(paste(modtemp$W_c,var),ncol=ncol(modtemp$W_c),nrow=length(var))
wo = matrix(paste(modtemp$W_o,var),ncol=ncol(modtemp$W_o),nrow=length(var))
t(wi)
t(wf)
t(wc)
t(wo)
ui = matrix(paste(modtemp$U_i,neu),ncol=ncol(modtemp$U_i),nrow=length(neu))
uf = matrix(paste(modtemp$U_f,neu),ncol=ncol(modtemp$U_f),nrow=length(neu))
uc = matrix(paste(modtemp$U_c,neu),ncol=ncol(modtemp$U_c),nrow=length(neu))
uo = matrix(paste(modtemp$U_o,neu),ncol=ncol(modtemp$U_o),nrow=length(neu))
t(ui)
t(uf)
t(uc)
t(uo)
modtemp$b_o
test = data.frame(best = bestresult.LSTM.ARMA.ARCH$train$predict, 
                  all=(rt.hat.train+sqrt(result.LSTM.ARMA.ARCH[[opt_idxLSTM.ARMA.ARCH]]$train))^2)
View(test)

#### ARMA-GARCH-LSTM ####
result = list()
result = result.LSTM.ARMA.GARCH
data = data.LSTM.ARMA
t.all = nrow(data)
n.lag = t.all-nrow(data.LSTM.ARMA.GARCH)

trainactual = (data$y[(n.lag+1):(t.all-nfore)])^2
testactual = (data$y[(t.all-nfore+1):t.all])^2
rt.hat.train = bestresult.LSTM.ARMA$train$predict[-c(1:n.lag)]
rt.hat.test = bestresult.LSTM.ARMA$test$predict

loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  #at^2.hat
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  #rt^2 = (rt.hat + at.hat)^2
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2
  
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxLSTM.ARMA.GARCH = which.min(loss$MSEtest);opt_idxLSTM.ARMA.GARCH
lossLSTM.ARMA.GARCH = loss
rownames(lossLSTM.ARMA.GARCH) = paste('Neuron',neuron)
lossLSTM.ARMA.GARCH

##### bobot & arsitektur LSTM ####
nameLSTM.ARMA.GARCH = result.LSTM.ARMA.GARCH$model_filename[opt_idxLSTM.ARMA.GARCH]
modLSTM.ARMA.GARCH = loadmodel(nameLSTM.ARMA.GARCH,opt_idxLSTM.ARMA.GARCH,LSTMmodel.path)
modLSTM.ARMA.GARCH
head(data.LSTM.ARMA.GARCH)
var = colnames(data.LSTM.ARMA.GARCH)[c(-1,-2)]
modtemp = modLSTM.ARMA.GARCH
wi = matrix(paste(modtemp$W_i,var),ncol=ncol(modtemp$W_i),nrow=length(var))
wf = matrix(paste(modtemp$W_f,var),ncol=ncol(modtemp$W_f),nrow=length(var))
wc = matrix(paste(modtemp$W_c,var),ncol=ncol(modtemp$W_c),nrow=length(var))
wo = matrix(paste(modtemp$W_o,var),ncol=ncol(modtemp$W_o),nrow=length(var))
t(wi)
t(wf)
t(wc)
t(wo)
neu = paste('h',seq(1,opt_idxLSTM.ARMA.GARCH,1))
ui = matrix(paste(modtemp$U_i,neu),ncol=ncol(modtemp$U_i),nrow=length(neu))
uf = matrix(paste(modtemp$U_f,neu),ncol=ncol(modtemp$U_f),nrow=length(neu))
uc = matrix(paste(modtemp$U_c,neu),ncol=ncol(modtemp$U_c),nrow=length(neu))
uo = matrix(paste(modtemp$U_o,neu),ncol=ncol(modtemp$U_o),nrow=length(neu))
t(ui)
t(uf)
t(uc)
t(uo)
test = data.frame(best = bestresult.LSTM.ARMA.GARCH$train$predict, 
                  all=(rt.hat.train+sqrt(result.LSTM.ARMA.GARCH[[opt_idxLSTM.ARMA.GARCH]]$train))^2)
View(test)
#### grafik perbandingan ARMA-GARCH-LSTM ####
title = "ARMA-GARCH LSTM"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.GARCH
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(LSTMbestresult$train$actual,LSTMbestresult$test$actual)
n.actual = length(actual)
train = c(LSTMbestresult$train$predict,rep(NA,1,length(LSTMbestresult$test$predict)))
test = c(rep(NA,1,length(LSTMbestresult$train$predict)),LSTMbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of detail GARCH-LSTM ##### 

####### end of Pemodelan ARMA-GARCH #######

####### Uji Perubahan Struktur #######
source("allfunction.R")
# rt2 train
rt2 = dataTrain$return^2
chowtest = ujiperubahanstruktur(rt2, alpha)


####### MSGARCH(1,1)#######
##### MSGARCH rt #####
MSGARCH.rt = fitMSGARCH(data = dataTrain$return, TrainActual = dataTrain$return^2, 
                               TestActual=dataTest$return^2, nfore=nfore, 
                               GARCHtype="sGARCH", distribution="norm", nstate=2)
msgarchmodel = MSGARCH.rt
msgarchmodel$modelfit
state = State(object = msgarchmodel$modelfit)
par(mfrow=c(2,1))
plot(state, type.prob = "filtered",xlab="t")

SR.fit <- ExtractStateFit(msgarchmodel$modelfit)
K = 2
msgarch.SR = list(0)
voltrain = matrix(nrow=dim(dataTrain)[1], ncol=K)
voltest = matrix(nrow=dim(dataTest)[1], ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = dataTrain$return^2, TrainActual = dataTrain$return^2, 
                               TestActual=dataTest$return^2, nfore, nstate=2)
  
  voltrain[,k] = msgarch.SR[[k]]$train
  voltest[,k] = msgarch.SR[[k]]$test
  plot(voltrain[,k], type = "l",xlab="t",ylab="volatilitas (%)", main=paste("Regime",k))
}

#uji normal
ujinormal(dataTrain$return)
#regime 1
par(mfrow=c(1,1))
pit <- PIT(object = SR.fit[[1]], do.norm = TRUE, do.its = TRUE)
pit = pit[is.finite(pit)]
hist(pit, breaks="Scott")
qqnorm(pit)
qqline(pit)
pit.rt1 = pit
ujinormal(pit.rt1)
resi.regime1 = dataTrain$return^2 - voltrain[,1]
ujinormal(resi.regime1)

#regime 2
pit <- PIT(object = SR.fit[[2]], do.norm = TRUE, do.its = TRUE)
pit = pit[is.finite(pit)]
qqnorm(pit)
qqline(pit)
pit.rt2 = pit
ujinormal(pit.rt2)
resi.regime2 = dataTrain$return^2 - voltrain[,2]
ujinormal(resi.regime2)

# untuk paper
for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = dataTrain$return^2, TrainActual = dataTrain$return^2, 
                               TestActual=dataTest$return^2, nfore, nstate=2)
  
  voltrain[,k] = msgarch.SR[[k]]$train
  voltest[,k] = msgarch.SR[[k]]$test
  plot(voltrain[,k], type = "l",xlab="t",ylab="realized volatility (%)", main=paste("Regime",k))
  
}
# ylim = c(min(voltrain[,1],voltrain[,2]),max(voltrain[,1],voltrain[,2]))
# for(k in 1:K){
# plot(voltrain[,k], type = "l",xlab="t",ylab="volatilitas (%)", ylim = ylim)
# }

#single plot
par(mfrow=c(1,1))
actual = c(dataTrain$return^2,dataTest$return^2)
n.actual = length(actual)
train = c(msgarchmodel$train,rep(NA,1,length(msgarchmodel$test)))
test = c(rep(NA,1,length(msgarchmodel$train)),msgarchmodel$test)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)

##### MSGARCH at_FFNN #####
NNbestresult = list()
resitrain = resitest = resi = vector()
NNbestresult = bestresult.NN.ARMA
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict

trainactual = NNbestresult$train$actual^2
testactual = NNbestresult$test$actual^2

MSGARCH.at_FFNN = fitMSGARCH(data = resitrain, TrainActual = trainactual, 
                        TestActual=testactual, nfore=nfore, 
                        GARCHtype="sGARCH", distribution="norm", nstate=2)
msgarchmodel = MSGARCH.at_FFNN
msgarchmodel$modelfit
state = State(object = msgarchmodel$modelfit)
par(mfrow=c(2,1))
plot(state, type.prob = "filtered",xlab="t")

SR.fit <- ExtractStateFit(msgarchmodel$modelfit)
K = 2
msgarch.SR = list(0)
voltrain = rt2hat.train = matrix(nrow=length(trainactual), ncol=K)
voltest = rt2hat.test = matrix(nrow=length(testactual), ncol=K)

for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = resitrain, TrainActual = trainactual, 
                               TestActual=testactual, nfore, nstate=2)
  
  voltrain[,k] = msgarch.SR[[k]]$train
  voltest[,k] = msgarch.SR[[k]]$test

  rt2hat.train[,k] = (sqrt(voltrain[,k]) + resitrain)^2
  rt2hat.test[,k] = (sqrt(voltest[,k]) + resitest)^2
  
  plot(rt2hat.train[,k], type = "l",xlab="t",ylab="volatilitas (%)", main=paste("Regime",k))
}

#uji normal
ujinormal(resitrain)
par(mfrow=c(1,1))
#regime 1
pit <- PIT(object = SR.fit[[1]], do.norm = TRUE, do.its = TRUE)
pit = pit[is.finite(pit)]
qqnorm(pit)
qqline(pit)
pit.ffnn1 = pit
ujinormal(pit.ffnn1)
resi.regime1 = resitrain^2 - voltrain[,1]
ujinormal(resi.regime1)

#regime 2
pit <- PIT(object = SR.fit[[2]], do.norm = TRUE, do.its = TRUE)
pit = pit[is.finite(pit)]
qqnorm(pit)
qqline(pit)
pit.ffnn2 = pit
ujinormal(pit.ffnn2)
resi.regime2 = resitrain^2 - voltrain[,2]
ujinormal(resi.regime2)

#### grafik perbandingan MSGARCH at_FFNN ####
title = "ARMA-FFNN-GARCH"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA.MSGARCH
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(NNbestresult$train$actual,NNbestresult$test$actual)
n.actual = length(actual)
train = c(NNbestresult$train$predict,rep(NA,1,length(NNbestresult$test$predict)))
test = c(rep(NA,1,length(NNbestresult$train$predict)),NNbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)

##### MSGARCH at_SVR #####
SVRbestresult = list()
resitrain = resitest = resi = vector()
SVRbestresult = bestresult.SVR.ARMA
resitrain = SVRbestresult$train$actual - SVRbestresult$train$predict
resitest = SVRbestresult$test$actual - SVRbestresult$test$predict

trainactual = SVRbestresult$train$actual^2
testactual = SVRbestresult$test$actual^2

MSGARCH.at_SVR = fitMSGARCH(data = resitrain, TrainActual = trainactual, 
                             TestActual=testactual, nfore=nfore, 
                             GARCHtype="sGARCH", distribution="norm", nstate=2)
msgarchmodel = MSGARCH.at_SVR
msgarchmodel$modelfit
state = State(object = msgarchmodel$modelfit)
par(mfrow=c(2,1))
plot(state, type.prob = "filtered",xlab="t")

SR.fit <- ExtractStateFit(msgarchmodel$modelfit)
K = 2
msgarch.SR = list(0)
voltrain = rt2hat.train = matrix(nrow=length(trainactual), ncol=K)
voltest = rt2hat.test = matrix(nrow=length(testactual), ncol=K)

par(mfrow=c(2,1))
for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = resitrain, TrainActual = trainactual, 
                               TestActual=testactual, nfore, nstate=2)
  
  voltrain[,k] = msgarch.SR[[k]]$train
  voltest[,k] = msgarch.SR[[k]]$test
  
  rt2hat.train[,k] = (sqrt(voltrain[,k]) + resitrain)^2
  rt2hat.test[,k] = (sqrt(voltest[,k]) + resitest)^2
  
  plot(rt2hat.train[,k], type = "l",xlab="t",ylab="volatilitas (%)", main=paste("Regime",k))
}

#uji normal
ujinormal(resitrain)
par(mfrow=c(1,1))
#regime 1
pit <- PIT(object = SR.fit[[1]], do.norm = TRUE, do.its = TRUE)
pit = pit[is.finite(pit)]
qqnorm(pit)
qqline(pit)
pit.svr1 = pit
ujinormal(pit.svr1)
resi.regime1 = resitrain^2 - voltrain[,1]
ujinormal(resi.regime1)

#regime 2
pit <- PIT(object = SR.fit[[2]], do.norm = TRUE, do.its = TRUE)
pit = pit[is.finite(pit)]
qqnorm(pit)
qqline(pit)
pit.svr2 = pit
ujinormal(pit.svr2)
resi.regime2 = resitrain^2 - voltrain[,2]
ujinormal(resi.regime2)

#### grafik perbandingan ARMA-SVR-MSGARCH ####
title = "ARMA-SVR-MSGARCH"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA.MSGARCH
par(mfrow=c(1,1))
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(SVRbestresult$train$actual,SVRbestresult$test$actual)
n.actual = length(actual)
train = c(SVRbestresult$train$predict,rep(NA,1,length(SVRbestresult$test$predict)))
test = c(rep(NA,1,length(SVRbestresult$train$predict)),SVRbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)

##### MSGARCH at_LSTM #####
LSTMbestresult = list()
resitrain = resitest = resi = vector()
LSTMbestresult = bestresult.LSTM.ARMA
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict

trainactual = LSTMbestresult$train$actual^2
testactual = LSTMbestresult$test$actual^2

MSGARCH.at_LSTM = fitMSGARCH(data = resitrain, TrainActual = trainactual, 
                            TestActual=testactual, nfore=nfore, 
                            GARCHtype="sGARCH", distribution="norm", nstate=2)
msgarchmodel = MSGARCH.at_LSTM
msgarchmodel$modelfit
msgarchmodel$modelfit$par
state = State(object = msgarchmodel$modelfit)
par(mfrow=c(2,1))
plot(state, type.prob = "filtered",xlab="t")

#untuk paper
par(mfrow=c(3,1))
plot(state, type.prob = "filtered")
plot.new()
plottitle = expression(paste(italic('a'['LSTM,t']),' MS-ARMA-GARCH (In-sample Data)'))
title(plottitle)

SR.fit <- ExtractStateFit(msgarchmodel$modelfit)
K = 2
msgarch.SR = list(0)
voltrain = rt2hat.train = matrix(nrow=length(trainactual), ncol=K)
voltest = rt2hat.test = matrix(nrow=length(testactual), ncol=K)

par(mfrow=c(2,1))
for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = resitrain, TrainActual = trainactual, 
                               TestActual=testactual, nfore, nstate=2)
  
  voltrain[,k] = msgarch.SR[[k]]$train
  voltest[,k] = msgarch.SR[[k]]$test
  
  rt2hat.train[,k] = (sqrt(voltrain[,k]) + resitrain)^2
  rt2hat.test[,k] = (sqrt(voltest[,k]) + resitest)^2
  
  plot(rt2hat.train[,k], type = "l",xlab="t",ylab="volatilitas (%)", main=paste("Regime",k))
}

#uji normal
ujinormal(resitrain)
par(mfrow=c(1,1))
#regime 1
pit <- PIT(object = SR.fit[[1]], do.norm = TRUE, do.its = TRUE)
pit = pit[is.finite(pit)]
qqnorm(pit)
qqline(pit)
pit.lstm1 = pit
ujinormal(pit.lstm1)
resi.regime1 = resitrain^2 - voltrain[,1]
ujinormal(resi.regime1)

#regime 2
pit <- PIT(object = SR.fit[[2]], do.norm = TRUE, do.its = TRUE)
pit = pit[is.finite(pit)]
qqnorm(pit)
qqline(pit)
pit.lstm2 = pit
ujinormal(pit.lstm2)
resi.regime2 = resitrain^2 - voltrain[,2]
ujinormal(resi.regime2)

#### grafik perbandingan ARMA-LSTM-MSGARCH ####
title = "ARMA-LSTM-GARCH"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.MSGARCH
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(LSTMbestresult$train$actual,LSTMbestresult$test$actual)
n.actual = length(actual)
train = c(LSTMbestresult$train$predict,rep(NA,1,length(LSTMbestresult$test$predict)))
test = c(rep(NA,1,length(LSTMbestresult$train$predict)),LSTMbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)




####### MSGARCH(1,1)-ML#######
##### MSGARCH-FFNN #####
result = list()
result = result.NN.MSGARCH.NN
data = data.NN.MSGARCH.NN
head(data)
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxNN.MSGARCH.NN = which.min(loss$MSEtest);opt_idxNN.MSGARCH.NN
lossNN.MSGARCH = loss
rownames(lossNN.MSGARCH) = paste('Neuron',neuron)
lossNN.MSGARCH

#bobot & arsitektur NN
plot(result.NN.MSGARCH.NN[[opt_idxNN.MSGARCH.NN]]$model_NN)
plot(result.NN.MSGARCH.NN[[opt_idxNN.MSGARCH.NN]]$model_NN, show.weights = FALSE)
result.NN.MSGARCH.NN[[opt_idxNN.MSGARCH.NN]]$model_NN$result.matrix

#### grafik perbandingan MSGARCH-FFNN ####
title = "MSGARCH FFNN 2 Variabel"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
NNbestresult = list()
NNbestresult = bestresult.NN.MSGARCH.NN
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(NNbestresult$train$actual,NNbestresult$test$actual)
n.actual = length(actual)
train = c(NNbestresult$train$predict,rep(NA,1,length(NNbestresult$test$predict)))
test = c(rep(NA,1,length(NNbestresult$train$predict)),NNbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of MSGARCH-FFNN ##### 

##### MSGARCH-FFNN-window5 #####
result = list()
result = result.NN.MSGARCH.NN.window5
data = data.NN.MSGARCH.NN.window5
head(data)
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxNN.MSGARCH.NN.window5 = which.min(loss$MSEtest);opt_idxNN.MSGARCH.NN.window5
lossNN.MSGARCH.window5 = loss
rownames(lossNN.MSGARCH.window5) = paste('Neuron',neuron)
lossNN.MSGARCH.window5


#bobot & arsitektur NN
plot(result.NN.MSGARCH.NN.window5[[opt_idxNN.MSGARCH.NN.window5]]$model_NN)
plot(result.NN.MSGARCH.NN.window5[[opt_idxNN.MSGARCH.NN.window5]]$model_NN, show.weights = FALSE)
result.NN.MSGARCH.NN.window5[[opt_idxNN.MSGARCH.NN.window5]]$model_NN$result.matrix

#### grafik perbandingan MSGARCH-FFNN-window5 ####
title = "MSGARCH FFNN 12 Variabel"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
NNbestresult = list()
NNbestresult = bestresult.NN.MSGARCH.NN.window5
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(NNbestresult$train$actual,NNbestresult$test$actual)
n.actual = length(actual)
train = c(NNbestresult$train$predict,rep(NA,1,length(NNbestresult$test$predict)))
test = c(rep(NA,1,length(NNbestresult$train$predict)),NNbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of MSGARCH-FFNN-window 5 ##### 

##### detail MSGARCH-SVR ##### 
#### MSARCH-SVR ####
result = list()
result = result.SVR.MSGARCH.SVR
data = data.SVR.MSGARCH.SVR
result$model.fit
result$w
result$b

#### grafik perbandingan MSGARCH-SVR ####
title = "MSGARCH SVR"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.MSGARCH.SVR
par(mfrow=c(1,1))
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(SVRbestresult$train$actual,SVRbestresult$test$actual)
n.actual = length(actual)
train = c(SVRbestresult$train$predict,rep(NA,1,length(SVRbestresult$test$predict)))
test = c(rep(NA,1,length(SVRbestresult$train$predict)),SVRbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)

#### MSARCH-SVR-window5 ####
result = list()
result = result.SVR.MSGARCH.SVR.window5
data = data.SVR.MSGARCH.SVR.window5
result$model.fit
result$w
result$b

#### grafik perbandingan MSGARCH-SVR-window5 ####
title = "MSGARCH SVR window 5"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.MSGARCH.SVR.window5
par(mfrow=c(1,1))
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(SVRbestresult$train$actual,SVRbestresult$test$actual)
n.actual = length(actual)
train = c(SVRbestresult$train$predict,rep(NA,1,length(SVRbestresult$test$predict)))
test = c(rep(NA,1,length(SVRbestresult$train$predict)),SVRbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of detail MSGARCH-SVR ##### 

##### detail MSGARCH-LSTM ##### 
#### MSGARCH-LSTM ####
result = list()
result = result.LSTM.MSGARCH.LSTM

data = data.LSTM.MSGARCH.LSTM
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxLSTM.MSGARCH.LSTM = which.min(loss$MSEtest);opt_idxLSTM.MSGARCH.LSTM
lossLSTM.MSGARCH = loss
rownames(lossLSTM.MSGARCH) = paste('Neuron',neuron)
lossLSTM.MSGARCH

# ployt MSE LSTM
result = list()
result = result.LSTM.MSGARCH.LSTM

trainactual = testactual = rt.hat.train = rt.hat.test = vector()
trainactual = bestresult.LSTM.ARMA$train$actual^2
testactual = bestresult.LSTM.ARMA$test$actual^2
rt.hat.train = bestresult.LSTM.ARMA$train$predict
rt.hat.test = bestresult.LSTM.ARMA$test$predict

losstrain.LSTM = matrix(nrow=n.neuron, ncol=1)
losstest.LSTM = matrix(nrow=n.neuron, ncol=1)
colnames(losstrain.LSTM) = c('MSE')
colnames(losstest.LSTM) = colnames(losstrain.LSTM)
rownames(losstrain.LSTM) = paste("Hidden_Node",neuron)
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

maxMSE = max(max(losstrain.LSTM),max(losstest.LSTM))
minMSE = min(min(losstrain.LSTM),min(losstest.LSTM))
par(mfrow=c(1,1))
plot(as.ts(losstrain.LSTM[,1]),ylab=paste("MSE"),xlab="Hidden Neuron",lwd=2,axes=F, ylim=c(minMSE, maxMSE*1.1))
box()
axis(side=2,lwd=0.5,cex.axis=0.8,las=2)
axis(side=1,lwd=0.5,cex.axis=0.8,las=0,at=c(1:length(neuron)),labels=neuron)
lines(losstest.LSTM[,1],col="red",lwd=2)
title(main="MSE MSGARCH-LSTM")
legend("topleft",c("In-Sample Data","Out-of-Sample Data"),col=c("black","red"), lwd=2,cex=0.7,bty = "n", y.intersp=1.5)

which.min(losstrain.LSTM)
which.min(losstest.LSTM)

####bobot & arsitektur MSGARCH-LSTM ####
nameLSTM.MSGARCH.LSTM = result.LSTM.MSGARCH.LSTM$model_filename[opt_idxLSTM.MSGARCH.LSTM]
modLSTM.MSGARCH.LSTM = loadmodel(nameLSTM.MSGARCH.LSTM,opt_idxLSTM.MSGARCH.LSTM,LSTMmodel.path)
modLSTM.MSGARCH.LSTM
head(data.LSTM.MSGARCH.LSTM)
var = colnames(data.LSTM.MSGARCH.LSTM)[c(-1,-2)]
modtemp = modLSTM.MSGARCH.LSTM
wi = matrix(paste(modtemp$W_i,var),ncol=ncol(modtemp$W_i),nrow=length(var))
wf = matrix(paste(modtemp$W_f,var),ncol=ncol(modtemp$W_f),nrow=length(var))
wc = matrix(paste(modtemp$W_c,var),ncol=ncol(modtemp$W_c),nrow=length(var))
wo = matrix(paste(modtemp$W_o,var),ncol=ncol(modtemp$W_o),nrow=length(var))


neu = paste('h',seq(1,opt_idxLSTM.MSGARCH.LSTM,1))
ui = matrix(paste(modtemp$U_i,neu),ncol=ncol(modtemp$U_i),nrow=length(neu))
uf = matrix(paste(modtemp$U_f,neu),ncol=ncol(modtemp$U_f),nrow=length(neu))
uc = matrix(paste(modtemp$U_c,neu),ncol=ncol(modtemp$U_c),nrow=length(neu))
uo = matrix(paste(modtemp$U_o,neu),ncol=ncol(modtemp$U_o),nrow=length(neu))

t(wi)
t(ui)
modLSTM.MSGARCH.LSTM$b_i

t(wf)
t(uf)
modLSTM.MSGARCH.LSTM$b_f

t(wc)
t(uc)
modLSTM.MSGARCH.LSTM$b_c

t(wo)
t(uo)
modLSTM.MSGARCH.LSTM$b_o

#### grafik perbandingan MSGARCH-LSTM ####
title = "MSGARCH LSTM"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.MSGARCH.LSTM
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(LSTMbestresult$train$actual,LSTMbestresult$test$actual)
n.actual = length(actual)
train = c(LSTMbestresult$train$predict,rep(NA,1,length(LSTMbestresult$test$predict)))
test = c(rep(NA,1,length(LSTMbestresult$train$predict)),LSTMbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
#### MSGARCH-LSTM-window5 ####
result = list()
result = result.LSTM.MSGARCH.LSTM.window5

data = data.LSTM.MSGARCH.LSTM.window5
t.all = nrow(data)
trainactual = data$y[1:(t.all-nfore)]
testactual = data$y[(t.all-nfore+1):t.all]
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  trainpred =  result[[i]]$train
  testpred = result[[i]]$test
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxLSTM.MSGARCH.LSTM.window5 = which.min(loss$MSEtest);opt_idxLSTM.MSGARCH.LSTM.window5
lossLSTM.MSGARCH.window5 = loss
rownames(lossLSTM.MSGARCH.window5) = paste('Neuron',neuron)
lossLSTM.MSGARCH.window5

# ployt MSE LSTM
result = list()
result = result.LSTM.MSGARCH.LSTM.window5

trainactual = testactual = rt.hat.train = rt.hat.test = vector()
trainactual = bestresult.LSTM.ARMA$train$actual^2
testactual = bestresult.LSTM.ARMA$test$actual^2
rt.hat.train = bestresult.LSTM.ARMA$train$predict
rt.hat.test = bestresult.LSTM.ARMA$test$predict

losstrain.LSTM = matrix(nrow=n.neuron, ncol=1)
losstest.LSTM = matrix(nrow=n.neuron, ncol=1)
colnames(losstrain.LSTM) = c('MSE')
colnames(losstest.LSTM) = colnames(losstrain.LSTM)
rownames(losstrain.LSTM) = paste("Hidden_Node",neuron)
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

maxMSE = max(max(losstrain.LSTM),max(losstest.LSTM))
minMSE = min(min(losstrain.LSTM),min(losstest.LSTM))
par(mfrow=c(1,1))
plot(as.ts(losstrain.LSTM[,1]),ylab=paste("MSE"),xlab="Hidden Neuron",lwd=2,axes=F, ylim=c(minMSE, maxMSE*1.1))
box()
axis(side=2,lwd=0.5,cex.axis=0.8,las=2)
axis(side=1,lwd=0.5,cex.axis=0.8,las=0,at=c(1:length(neuron)),labels=neuron)
lines(losstest.LSTM[,1],col="red",lwd=2)
title(main="MSE MSGARCH-LSTM")
legend("topleft",c("In-Sample Data","Out-of-Sample Data"),col=c("black","red"), lwd=2,cex=0.7,bty = "n", y.intersp=1.5)

which.min(losstrain.LSTM)
which.min(losstest.LSTM)

#### bobot & arsitektur MSGARCH-LSTM ####
nameLSTM.MSGARCH.LSTM.window5 = result.LSTM.MSGARCH.LSTM.window5$model_filename[opt_idxLSTM.MSGARCH.LSTM.window5]
modLSTM.MSGARCH.LSTM.window5 = loadmodel(nameLSTM.MSGARCH.LSTM.window5,opt_idxLSTM.MSGARCH.LSTM.window5,LSTMmodel.path)
modLSTM.MSGARCH.LSTM.window5
head(data.LSTM.MSGARCH.LSTM.window5)
var = colnames(data.LSTM.MSGARCH.LSTM)[c(-1,-2)]
modtemp = modLSTM.MSGARCH.LSTM.window5
wi = matrix(paste(modtemp$W_i,var),ncol=ncol(modtemp$W_i),nrow=length(var))
wf = matrix(paste(modtemp$W_f,var),ncol=ncol(modtemp$W_f),nrow=length(var))
wc = matrix(paste(modtemp$W_c,var),ncol=ncol(modtemp$W_c),nrow=length(var))
wo = matrix(paste(modtemp$W_o,var),ncol=ncol(modtemp$W_o),nrow=length(var))


neu = paste('h',seq(1,opt_idxLSTM.MSGARCH.LSTM.window5,1))
ui = matrix(paste(modtemp$U_i,neu),ncol=ncol(modtemp$U_i),nrow=length(neu))
uf = matrix(paste(modtemp$U_f,neu),ncol=ncol(modtemp$U_f),nrow=length(neu))
uc = matrix(paste(modtemp$U_c,neu),ncol=ncol(modtemp$U_c),nrow=length(neu))
uo = matrix(paste(modtemp$U_o,neu),ncol=ncol(modtemp$U_o),nrow=length(neu))

t(wi)
t(ui)
modtemp$b_i

t(wf)
t(uf)
modtemp$b_f

t(wc)
t(uc)
modtemp$b_c

t(wo)
t(uo)
modtemp$b_o
 dim(modtemp$W_i)

#### grafik perbandingan MSGARCH-LSTM-window5 ####
title = "MSGARCH LSTM window5"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.MSGARCH.LSTM.window5
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(LSTMbestresult$train$actual,LSTMbestresult$test$actual)
n.actual = length(actual)
train = c(LSTMbestresult$train$predict,rep(NA,1,length(LSTMbestresult$test$predict)))
test = c(rep(NA,1,length(LSTMbestresult$train$predict)),LSTMbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of detail MSGARCH-LSTM ##### 


####### ARMA-MSGARCH(1,1)-ML#######
##### ARMA-MSGARCH-FFNN #####
result = list()
result = result.NN.ARMA.MSGARCH.NN
head(data.NN.ARMA.MSGARCH.NN)
head(data.NN.ARMA)

trainactual = testactual = rt.hat.train =  rt.hat.test = vector()
trainactual = bestresult.NN.ARMA$train$actual^2
testactual = bestresult.NN.ARMA$test$actual^2
rt.hat.train = bestresult.NN.ARMA$train$predict
rt.hat.test = bestresult.NN.ARMA$test$predict

loss = matrix(nrow=n.neuron, ncol=4)
attrainpred = attestpred = trainpred =  testpred = vector()
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2
  
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxNN.ARMA.MSGARCH.NN = which.min(loss$MSEtest);opt_idxNN.ARMA.MSGARCH.NN
lossNN.ARMA.MSGARCH = loss
rownames(lossNN.ARMA.MSGARCH) = paste('Neuron',neuron)
lossNN.ARMA.MSGARCH

#bobot & arsitektur NN
plot(result.NN.ARMA.MSGARCH.NN[[opt_idxNN.ARMA.MSGARCH.NN]]$model_NN)
plot(result.NN.ARMA.MSGARCH.NN[[opt_idxNN.ARMA.MSGARCH.NN]]$model_NN, show.weights = FALSE)
result.NN.ARMA.MSGARCH.NN[[opt_idxNN.ARMA.MSGARCH.NN]]$model_NN$result.matrix

#### grafik perbandingan ARMA-MSGARCH-FFNN ####
title = "MSGARCH FFNN 2 Variabel"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA.MSGARCH.NN
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(NNbestresult$train$actual,NNbestresult$test$actual)
n.actual = length(actual)
train = c(NNbestresult$train$predict,rep(NA,1,length(NNbestresult$test$predict)))
test = c(rep(NA,1,length(NNbestresult$train$predict)),NNbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of MSGARCH-FFNN ##### 

##### ARMA-MSGARCH-FFNN-window5 #####
result = list()
result = result.NN.ARMA.MSGARCH.NN.window5

head(data.NN.ARMA.MSGARCH.NN.window5)
trainactual = testactual = rt.hat.train =  rt.hat.test = vector()
trainactual = bestresult.NN.ARMA$train$actual^2
testactual = bestresult.NN.ARMA$test$actual^2
rt.hat.train = bestresult.NN.ARMA$train$predict
rt.hat.test = bestresult.NN.ARMA$test$predict

loss = matrix(nrow=n.neuron, ncol=4)
attrainpred = attestpred = trainpred =  testpred = vector()
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2
  
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxNN.ARMA.MSGARCH.NN.window5 = which.min(loss$MSEtest);opt_idxNN.ARMA.MSGARCH.NN.window5
lossNN.ARMA.MSGARCH.window5 = loss
rownames(lossNN.ARMA.MSGARCH.window5) = paste('Neuron',neuron)
lossNN.ARMA.MSGARCH.window5


#bobot & arsitektur NN
plot(result.NN.ARMA.MSGARCH.NN.window5[[opt_idxNN.ARMA.MSGARCH.NN.window5]]$model_NN)
plot(result.NN.ARMA.MSGARCH.NN.window5[[opt_idxNN.ARMA.MSGARCH.NN.window5]]$model_NN, show.weights = FALSE)
result.NN.ARMA.MSGARCH.NN.window5[[opt_idxNN.ARMA.MSGARCH.NN.window5]]$model_NN$result.matrix

#### grafik perbandingan MSGARCH-FFNN-window5 ####
title = "ARMA MSGARCH FFNN 12 Variabel"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA.MSGARCH.NN.window5
par(mfrow=c(1,1))
makeplot(NNbestresult$train$actual, NNbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(NNbestresult$test$actual, NNbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(NNbestresult$train$actual,NNbestresult$test$actual)
n.actual = length(actual)
train = c(NNbestresult$train$predict,rep(NA,1,length(NNbestresult$test$predict)))
test = c(rep(NA,1,length(NNbestresult$train$predict)),NNbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of MSGARCH-FFNN-window 5 ##### 

##### detail ARMA-MSGARCH-SVR ##### 
#### ARMA-MSGARCH-SVR ####
result = list()
result = result.SVR.ARMA.MSGARCH.SVR
data = data.SVR.ARMA.MSGARCH.SVR
result$model.fit
result$w
result$b

#### grafik perbandingan ARMA-MSGARCH-SVR ####
title = "ARMA MSGARCH SVR"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA.MSGARCH.SVR
par(mfrow=c(1,1))
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(SVRbestresult$train$actual,SVRbestresult$test$actual)
n.actual = length(actual)
train = c(SVRbestresult$train$predict,rep(NA,1,length(SVRbestresult$test$predict)))
test = c(rep(NA,1,length(SVRbestresult$train$predict)),SVRbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)

#### ARMA-MSARCH-SVR-window5 ####
result = list()
result = result.SVR.ARMA.MSGARCH.SVR.window5
data = data.SVR.ARMA.MSGARCH.SVR.window5
result$model.fit
result$w
result$b

#### grafik perbandingan ARMA-MSGARCH-SVR-window5 ####
title = "ARMA MSGARCH SVR window 5"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA.MSGARCH.SVR.window5
par(mfrow=c(1,1))
makeplot(SVRbestresult$train$actual, SVRbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(SVRbestresult$test$actual, SVRbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(SVRbestresult$train$actual,SVRbestresult$test$actual)
n.actual = length(actual)
train = c(SVRbestresult$train$predict,rep(NA,1,length(SVRbestresult$test$predict)))
test = c(rep(NA,1,length(SVRbestresult$train$predict)),SVRbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of detail ARMA-MSGARCH-SVR ##### 

##### detail ARMA-MSGARCH-LSTM ##### 
#### ARMA-MSGARCH-LSTM ####
result = list()
result = result.LSTM.ARMA.MSGARCH.LSTM
data = data.LSTM.ARMA.MSGARCH.LSTM
# head(data.LSTM.ARMA.MSGARCH.LSTM)
# head(data.LSTM.ARMA)
# dim(data.LSTM.ARMA.MSGARCH.LSTM)
# dim(data.LSTM.ARMA)

trainactual = testactual = rt.hat.train =  rt.hat.test = vector()
trainactual = bestresult.LSTM.ARMA$train$actual^2
testactual = bestresult.LSTM.ARMA$test$actual^2
rt.hat.train = bestresult.LSTM.ARMA$train$predict
rt.hat.test = bestresult.LSTM.ARMA$test$predict


loss = matrix(nrow=n.neuron, ncol=4)
attrainpred = attestpred = trainpred =  testpred = vector()
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2
  
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxLSTM.ARMA.MSGARCH.LSTM = which.min(loss$MSEtest);opt_idxLSTM.ARMA.MSGARCH.LSTM
lossLSTM.ARMA.MSGARCH = loss
rownames(lossLSTM.ARMA.MSGARCH) = paste('Neuron',neuron)
lossLSTM.ARMA.MSGARCH

# plot MSE LSTM
result = list()
result = result.LSTM.ARMA.MSGARCH.LSTM

trainactual = testactual = rt.hat.train = rt.hat.test = vector()
trainactual = bestresult.LSTM.ARMA$train$actual^2
testactual = bestresult.LSTM.ARMA$test$actual^2
rt.hat.train = bestresult.LSTM.ARMA$train$predict
rt.hat.test = bestresult.LSTM.ARMA$test$predict

losstrain.LSTM = matrix(nrow=n.neuron, ncol=1)
losstest.LSTM = matrix(nrow=n.neuron, ncol=1)
colnames(losstrain.LSTM) = c('MSE')
colnames(losstest.LSTM) = colnames(losstrain.LSTM)
rownames(losstrain.LSTM) = paste("Hidden_Node",neuron)
rownames(losstest.LSTM) = rownames(losstrain.LSTM)
for(i in 1:n.neuron){
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2

  losstrain.LSTM[i] = hitungloss(trainactual, trainpred, method = "MSE")
  losstest.LSTM[i] = hitungloss(testactual, testpred, method = "MSE")
}
losstrain.LSTM
losstest.LSTM

maxMSE = max(max(losstrain.LSTM),max(losstest.LSTM))
minMSE = min(min(losstrain.LSTM),min(losstest.LSTM))
par(mfrow=c(1,1))
plot(as.ts(losstrain.LSTM[,1]),ylab=paste("MSE"),xlab="Hidden Neuron",lwd=2,axes=F, ylim=c(minMSE, maxMSE*1.1))
box()
axis(side=2,lwd=0.5,cex.axis=0.8,las=2)
axis(side=1,lwd=0.5,cex.axis=0.8,las=0,at=c(1:length(neuron)),labels=neuron)
lines(losstest.LSTM[,1],col="red",lwd=2)
title(main="MSE ARMA-MSGARCH-LSTM")
legend("topleft",c("In-Sample Data","Out-of-Sample Data"),col=c("black","red"), lwd=2,cex=0.7,bty = "n", y.intersp=1.5)

which.min(losstrain.LSTM)
which.min(losstest.LSTM)

####bobot & arsitektur ARMA-MSGARCH-LSTM ####
nameLSTM.ARMA.MSGARCH.LSTM = result.LSTM.ARMA.MSGARCH.LSTM$model_filename[opt_idxLSTM.ARMA.MSGARCH.LSTM]
modLSTM.ARMA.MSGARCH.LSTM = loadmodel(nameLSTM.ARMA.MSGARCH.LSTM,opt_idxLSTM.ARMA.MSGARCH.LSTM,LSTMmodel.path)
modLSTM.ARMA.MSGARCH.LSTM
head(data.LSTM.ARMA.MSGARCH.LSTM)
var = colnames(data.LSTM.ARMA.MSGARCH.LSTM)[c(-1,-2)]
modtemp = modLSTM.ARMA.MSGARCH.LSTM
wi = matrix(paste(modtemp$W_i,var),ncol=ncol(modtemp$W_i),nrow=length(var))
wf = matrix(paste(modtemp$W_f,var),ncol=ncol(modtemp$W_f),nrow=length(var))
wc = matrix(paste(modtemp$W_c,var),ncol=ncol(modtemp$W_c),nrow=length(var))
wo = matrix(paste(modtemp$W_o,var),ncol=ncol(modtemp$W_o),nrow=length(var))


neu = paste('h',seq(1,opt_idxLSTM.ARMA.MSGARCH.LSTM,1))
ui = matrix(paste(modtemp$U_i,neu),ncol=ncol(modtemp$U_i),nrow=length(neu))
uf = matrix(paste(modtemp$U_f,neu),ncol=ncol(modtemp$U_f),nrow=length(neu))
uc = matrix(paste(modtemp$U_c,neu),ncol=ncol(modtemp$U_c),nrow=length(neu))
uo = matrix(paste(modtemp$U_o,neu),ncol=ncol(modtemp$U_o),nrow=length(neu))

t(wi)
t(ui)
modtemp$b_i

t(wf)
t(uf)
modtemp$b_f

t(wc)
t(uc)
modtemp$b_c

t(wo)
t(uo)
modtemp$b_o

#### grafik perbandingan ARMA-MSGARCH-LSTM ####
title = "ARMA-MSGARCH LSTM"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.MSGARCH.LSTM
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(LSTMbestresult$train$actual,LSTMbestresult$test$actual)
n.actual = length(actual)
train = c(LSTMbestresult$train$predict,rep(NA,1,length(LSTMbestresult$test$predict)))
test = c(rep(NA,1,length(LSTMbestresult$train$predict)),LSTMbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)

#### ARMA-MSGARCH-LSTM-window5 ####
result = list()
result = result.LSTM.ARMA.MSGARCH.LSTM.window5
window=5

data = data.LSTM.ARMA.MSGARCH.LSTM.window5
trainactual = testactual = rt.hat.train =  rt.hat.test = vector()
trainactual = bestresult.LSTM.ARMA$train$actual[-c(1:window)]^2
testactual = bestresult.LSTM.ARMA$test$actual^2
rt.hat.train = bestresult.LSTM.ARMA$train$predict[-c(1:window)]
rt.hat.test = bestresult.LSTM.ARMA$test$predict

length(trainactual)
loss = matrix(nrow=n.neuron, ncol=4)
colnames(loss) = c("MSEtrain","sMAPEtrain","MSEtest","sMAPEtest")
for(i in 1:n.neuron){
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2
  
  loss[i,1] = hitungloss(trainactual, trainpred, method = "MSE")
  loss[i,2] = hitungloss(trainactual, trainpred, method = "sMAPE")
  loss[i,3] = hitungloss(testactual, testpred, method = "MSE")
  loss[i,4] = hitungloss(testactual, testpred, method = "sMAPE")
}
loss = data.frame(loss)
loss
opt_idxLSTM.ARMA.MSGARCH.LSTM.window5 = which.min(loss$MSEtest);opt_idxLSTM.ARMA.MSGARCH.LSTM.window5
lossLSTM.ARMA.MSGARCH.window5 = loss
rownames(lossLSTM.ARMA.MSGARCH.window5) = paste('Neuron',neuron)
lossLSTM.ARMA.MSGARCH.window5

# plot MSE LSTM
result = list()
result = result.LSTM.ARMA.MSGARCH.LSTM.window5

trainactual = testactual = rt.hat.train = rt.hat.test = vector()
trainactual = bestresult.LSTM.ARMA$train$actual[-c(1:window)]^2
testactual = bestresult.LSTM.ARMA$test$actual^2
rt.hat.train = bestresult.LSTM.ARMA$train$predict[-c(1:window)]
rt.hat.test = bestresult.LSTM.ARMA$test$predict

losstrain.LSTM = matrix(nrow=n.neuron, ncol=1)
losstest.LSTM = matrix(nrow=n.neuron, ncol=1)
colnames(losstrain.LSTM) = c('MSE')
colnames(losstest.LSTM) = colnames(losstrain.LSTM)
rownames(losstrain.LSTM) = paste("Hidden_Node",neuron)
rownames(losstest.LSTM) = rownames(losstrain.LSTM)
for(i in 1:n.neuron){
  attrainpred =  sqrt(result[[i]]$train)
  attestpred = sqrt(result[[i]]$test)
  
  trainpred = (rt.hat.train + attrainpred)^2
  testpred = (rt.hat.test + attestpred)^2

  
  losstrain.LSTM[i] = hitungloss(trainactual, trainpred, method = "MSE")
  losstest.LSTM[i] = hitungloss(testactual, testpred, method = "MSE")
}
losstrain.LSTM
losstest.LSTM

maxMSE = max(max(losstrain.LSTM),max(losstest.LSTM))
minMSE = min(min(losstrain.LSTM),min(losstest.LSTM))
par(mfrow=c(1,1))
plot(as.ts(losstrain.LSTM[,1]),ylab=paste("MSE"),xlab="Hidden Neuron",lwd=2,axes=F, ylim=c(minMSE, maxMSE*1.1))
box()
axis(side=2,lwd=0.5,cex.axis=0.8,las=2)
axis(side=1,lwd=0.5,cex.axis=0.8,las=0,at=c(1:length(neuron)),labels=neuron)
lines(losstest.LSTM[,1],col="red",lwd=2)
title(main="MSE MSGARCH-LSTM")
legend("topleft",c("In-Sample Data","Out-of-Sample Data"),col=c("black","red"), lwd=2,cex=0.7,bty = "n", y.intersp=1.5)

which.min(losstrain.LSTM)
which.min(losstest.LSTM)

#### bobot & arsitektur ARMA-MSGARCH-LSTM window5 ####
nameLSTM.ARMA.MSGARCH.LSTM.window5 = result.LSTM.ARMA.MSGARCH.LSTM.window5$model_filename[opt_idxLSTM.ARMA.MSGARCH.LSTM.window5]
modLSTM.ARMA.MSGARCH.LSTM.window5 = loadmodel(nameLSTM.ARMA.MSGARCH.LSTM.window5,opt_idxLSTM.ARMA.MSGARCH.LSTM.window5,LSTMmodel.path)
modLSTM.ARMA.MSGARCH.LSTM.window5
head(data.LSTM.ARMA.MSGARCH.LSTM.window5)
var = colnames(data.LSTM.MSGARCH.LSTM)[c(-1,-2)]
modtemp = modLSTM.ARMA.MSGARCH.LSTM.window5
wi = matrix(paste(modtemp$W_i,var),ncol=ncol(modtemp$W_i),nrow=length(var))
wf = matrix(paste(modtemp$W_f,var),ncol=ncol(modtemp$W_f),nrow=length(var))
wc = matrix(paste(modtemp$W_c,var),ncol=ncol(modtemp$W_c),nrow=length(var))
wo = matrix(paste(modtemp$W_o,var),ncol=ncol(modtemp$W_o),nrow=length(var))


neu = paste('h',seq(1,opt_idxLSTM.ARMA.MSGARCH.LSTM.window5,1))
ui = matrix(paste(modtemp$U_i,neu),ncol=ncol(modtemp$U_i),nrow=length(neu))
uf = matrix(paste(modtemp$U_f,neu),ncol=ncol(modtemp$U_f),nrow=length(neu))
uc = matrix(paste(modtemp$U_c,neu),ncol=ncol(modtemp$U_c),nrow=length(neu))
uo = matrix(paste(modtemp$U_o,neu),ncol=ncol(modtemp$U_o),nrow=length(neu))

t(wi)
t(ui)
modtemp$b_i

t(wf)
t(uf)
modtemp$b_f

t(wc)
t(uc)
modtemp$b_c

t(wo)
t(uo)
modtemp$b_o
dim(modtemp$W_i)

#### grafik perbandingan MSGARCH-LSTM-window5 ####
title = "MSGARCH LSTM window5"
# xlabel = "t"
# ylabel = "return kuadrat (%)"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.MSGARCH.LSTM.window5
par(mfrow=c(1,1))
makeplot(LSTMbestresult$train$actual, LSTMbestresult$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(LSTMbestresult$test$actual, LSTMbestresult$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
#single plot
actual = c(LSTMbestresult$train$actual,LSTMbestresult$test$actual)
n.actual = length(actual)
train = c(LSTMbestresult$train$predict,rep(NA,1,length(LSTMbestresult$test$predict)))
test = c(rep(NA,1,length(LSTMbestresult$train$predict)),LSTMbestresult$test$predict)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of detail MSGARCH-LSTM ##### 


###### Perbandingan ######
lossfunction = getlossfunction()
losstrain.NN = losstest.NN = matrix(NA,nrow=7, ncol=2)
losstrain.SVR = losstest.SVR = matrix(NA,nrow=7, ncol=2)
losstrain.LSTM = losstest.LSTM = matrix(NA,nrow=7, ncol=2)
model.ffnn = model.svr = model.lstm = vector(length=7)

##### all FFNN #####
idx.ffnn = 1
model.ffnn[idx.ffnn] = "GARCH"
NNbestresult = list()
NNbestresult = bestresult.NN.GARCH
for(j in 1:length(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

idx.ffnn = 2
model.ffnn[idx.ffnn] = "ARMA-GARCH"
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA.GARCH
for(j in 1:length(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

idx.ffnn = 3
model.ffnn[idx.ffnn] = "MS-ARMA-GARCH"
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA.MSGARCH
for(j in 1:length(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}
idx.ffnn = 4
model.ffnn[idx.ffnn] = "MSGARCH-FFNN-2"
NNbestresult = list()
NNbestresult = bestresult.NN.MSGARCH.NN
for(j in 1:length(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

idx.ffnn = 5
model.ffnn[idx.ffnn] = "MSGARCH-FFNN-12"
NNbestresult = list()
NNbestresult = bestresult.NN.MSGARCH.NN.window5
for(j in 1:length(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

idx.ffnn = 6
model.ffnn[idx.ffnn] = "MS-ARMA-GARCH-FFNN-2"
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA.MSGARCH.NN
for(j in 1:length(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

idx.ffnn = 7
model.ffnn[idx.ffnn] = "MS-ARMA-GARCH-FFNN-12"
NNbestresult = list()
NNbestresult = bestresult.NN.ARMA.MSGARCH.NN.window5
for(j in 1:length(lossfunction)){
  losstrain.NN[idx.ffnn,j] = hitungloss(NNbestresult$train$actual, NNbestresult$train$predict, method = lossfunction[j])
  losstest.NN[idx.ffnn,j] = hitungloss(NNbestresult$test$actual, NNbestresult$test$predict, method = lossfunction[j])
}

rownames(losstrain.NN) = rownames(losstest.NN) = model.ffnn
colnames(losstrain.NN) = colnames(losstest.NN) = lossfunction
losstrain.NN
losstest.NN
##### end of all FFNN #####
##### all SVR #####
idx.svr = 1
model.svr[idx.svr] = "GARCH"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.GARCH
for(j in 1:length(lossfunction)){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}

idx.svr = 2
model.svr[idx.svr] = "ARMA-GARCH"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA.GARCH
for(j in 1:length(lossfunction)){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}

idx.svr = 3
model.svr[idx.svr] = "MS-ARMA-GARCH"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA.MSGARCH
for(j in 1:length(lossfunction)){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}

idx.svr = 4
model.svr[idx.svr] = "MSGARCH-SVR-2"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.MSGARCH.SVR
for(j in 1:length(lossfunction)){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}

idx.svr = 5
model.svr[idx.svr] = "MSGARCH-SVR-12"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.MSGARCH.SVR.window5
for(j in 1:length(lossfunction)){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}

idx.svr = 6
model.svr[idx.svr] = "MS-ARMA-GARCH-SVR-2"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA.MSGARCH.SVR
for(j in 1:length(lossfunction)){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}

idx.svr = 7
model.svr[idx.svr] = "MS-ARMA-GARCH-SVR-12"
SVRbestresult = list()
SVRbestresult = bestresult.SVR.ARMA.MSGARCH.SVR.window5
for(j in 1:length(lossfunction)){
  losstrain.SVR[idx.svr,j] = hitungloss(SVRbestresult$train$actual, SVRbestresult$train$predict, method = lossfunction[j])
  losstest.SVR[idx.svr,j] = hitungloss(SVRbestresult$test$actual, SVRbestresult$test$predict, method = lossfunction[j])
}

rownames(losstrain.SVR) = rownames(losstest.SVR) = model.svr
colnames(losstrain.SVR) = colnames(losstest.SVR) = lossfunction
losstrain.SVR
losstest.SVR
##### end of all SVR #####
##### all LSTM #####
idx.lstm = 1
model.lstm[idx.lstm] = "GARCH"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.GARCH
for(j in 1:length(lossfunction)){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}

idx.lstm = 2
model.lstm[idx.lstm] = "ARMA-GARCH"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.GARCH
for(j in 1:length(lossfunction)){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}

idx.lstm = 3
model.lstm[idx.lstm] = "MS-ARMA-GARCH"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.MSGARCH
for(j in 1:length(lossfunction)){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}

idx.lstm = 4
model.lstm[idx.lstm] = "MSGARCH-LSTM-2"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.MSGARCH.LSTM
for(j in 1:length(lossfunction)){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}

idx.lstm = 5
model.lstm[idx.lstm] = "MSGARCH-LSTM-12"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.MSGARCH.LSTM.window5
for(j in 1:length(lossfunction)){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}

idx.lstm = 6
model.lstm[idx.lstm] = "MS-ARMA-GARCH-LSTM-2"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.MSGARCH.LSTM
for(j in 1:length(lossfunction)){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}

idx.lstm = 7
model.lstm[idx.lstm] = "MS-ARMA-GARCH-LSTM-12"
LSTMbestresult = list()
LSTMbestresult = bestresult.LSTM.ARMA.MSGARCH.LSTM.window5
for(j in 1:length(lossfunction)){
  losstrain.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$train$actual, LSTMbestresult$train$predict, method = lossfunction[j])
  losstest.LSTM[idx.lstm,j] = hitungloss(LSTMbestresult$test$actual, LSTMbestresult$test$predict, method = lossfunction[j])
}

rownames(losstrain.LSTM) = rownames(losstest.LSTM) = model.lstm
colnames(losstrain.LSTM) = colnames(losstest.LSTM) = lossfunction
losstrain.LSTM
losstest.LSTM

##### end of all LSTM #####


##### MSGARCH #####
msgarch.loss.train = msgarch.loss.test = matrix(NA,nrow=1, ncol=2)

result.MSGARCH = fitMSGARCH(data = dataTrain$return, TrainActual = dataTrain$return^2, TestActual=dataTest$return^2, nfore=nfore, 
                            GARCHtype="sGARCH", distribution="norm", nstate=2)
for(j in 1:length(lossfunction)){
  msgarch.loss.train[1,j] = hitungloss(dataTrain$return^2, result.MSGARCH$train, method = lossfunction[j])
  msgarch.loss.test[1,j] = hitungloss(dataTest$return^2, result.MSGARCH$test, method = lossfunction[j])
}

colnames(msgarch.loss.train) = colnames(msgarch.loss.test) = lossfunction
# rownames(msgarch.loss.train) = rownames(msgarch.loss.train) = c("FFNN","SVR","LSTM","Manual")
msgarch.loss.train
msgarch.loss.test


title = "MSGARCH"
xlabel = "t"
ylabel = "return kuadrat (%)"
par(mfrow=c(1,1))
#single plot
actual = c(dataTrain$return^2,dataTest$return^2)
n.actual = length(actual)
train = c(result.MSGARCH$train,rep(NA,1,length(result.MSGARCH$test)))
test = c(rep(NA,1,length(result.MSGARCH$train)),result.MSGARCH$test)
plot(actual,type="l",xlab = xlabel, ylab=ylabel)
lines(train,type="l",col="red")
lines(test,type="l",col="green")
legend("topleft",c("Actual","Forecast In-sample","Forecast Out-of-sample"),
       col=c("black","red","green"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
##### end of MSGARCH #####

#### prediction result ####
#testing
test.actual = dataTest$return^2
test.msgarch = result.MSGARCH$test
test.msgarch.NN.2 = bestresult.NN.MSGARCH.NN$test$predict
test.msgarch.NN.12 = bestresult.NN.MSGARCH.NN.window5$test$predict
test.msgarch.SVR.12 = bestresult.SVR.MSGARCH.SVR.window5$test$predict

plot(test.actual,type="l", lwd=2, ylab="volatility (%)", xlab="t", ylim=c(0,max(test.actual)))
lines(test.msgarch, col="red", lwd=2)
lines(test.msgarch.NN.2,col="yellow", lwd=2)
lines(test.msgarch.NN.12,col="green", lwd=2)
lines(test.msgarch.SVR.12,col="blue", lwd=2)
legend("topright",c("Realized Volatility","MSGARCH","MSGARCH-FFNN 2 variabel input","MSGARCH-FFNN 12 variabel input",
                    "MSGARCH-SVR 12 variabel input"), col=c("black","red","yellow","green","blue"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5, inset=0.1)
title("Out-of-Sample Data")
##### perbandingan MSGARCH dan MSGARCH-NN #####

###### end of Perbandingan ######

###### FOR PAPER ######
##### grafik all untuk paper #####
#### Probability #### 
## a_(SVR,t) MS-ARMA-GARCH
msgarchmodel = MSGARCH.at_SVR
msgarchmodel$modelfit
state = State(object = msgarchmodel$modelfit)

par(mfrow=c(3,1))
plot(state, type.prob = "filtered")
plot.new()
plottitle = expression(paste(italic('a'['SVR,t']),' MS-ARMA-GARCH (In-sample Data)'))
title(plottitle)

SR.fit <- ExtractStateFit(msgarchmodel$modelfit)
K = 2
msgarch.SR = list(0)
voltrain = rt2hat.train = matrix(nrow=length(trainactual), ncol=K)
voltest = rt2hat.test = matrix(nrow=length(testactual), ncol=K)

par(mfrow=c(3,1))
for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = resitrain, TrainActual = trainactual, 
                               TestActual=testactual, nfore, nstate=2)
  
  voltrain[,k] = msgarch.SR[[k]]$train
  voltest[,k] = msgarch.SR[[k]]$test
  
  rt2hat.train[,k] = (sqrt(voltrain[,k]) + resitrain)^2
  rt2hat.test[,k] = (sqrt(voltest[,k]) + resitest)^2
  
  plot(rt2hat.train[,k], type = "l",xlab="t",ylab="volatility (%)", main=paste("Regime",k))
}
plot.new()
plottitle = expression(paste(italic('a'['SVR,t']),' MS-ARMA-GARCH (In-sample Data)'))
title(plottitle)

## a_(LSTM,t) MS-ARMA-GARCH
msgarchmodel = MSGARCH.at_LSTM
msgarchmodel$modelfit
msgarchmodel$modelfit$par
state = State(object = msgarchmodel$modelfit)

par(mfrow=c(3,1))
plot(state, type.prob = "filtered")
plot.new()
plottitle = expression(paste(italic('a'['LSTM,t']),' MS-ARMA-GARCH (In-sample Data)'))
title(plottitle)

SR.fit <- ExtractStateFit(msgarchmodel$modelfit)
K = 2
msgarch.SR = list(0)
voltrain = rt2hat.train = matrix(nrow=length(trainactual), ncol=K)
voltest = rt2hat.test = matrix(nrow=length(testactual), ncol=K)

par(mfrow=c(3,1))
for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = resitrain, TrainActual = trainactual, 
                               TestActual=testactual, nfore, nstate=2)
  
  voltrain[,k] = msgarch.SR[[k]]$train
  voltest[,k] = msgarch.SR[[k]]$test
  
  rt2hat.train[,k] = (sqrt(voltrain[,k]) + resitrain)^2
  rt2hat.test[,k] = (sqrt(voltest[,k]) + resitest)^2
  
  plot(rt2hat.train[,k], type = "l",xlab="t",ylab="volatility (%)", main=paste("Regime",k))
}
plot.new()
plottitle = expression(paste(italic('a'['LSTM,t']),' MS-ARMA-GARCH (In-sample Data)'))
title(plottitle)
#### prediction result ####
#training
train.actual = bestresult.SVR.ARMA.MSGARCH.SVR$train$actual
train.SVR = bestresult.SVR.ARMA.MSGARCH.SVR$train$predict
train.LSTM = bestresult.LSTM.ARMA.MSGARCH.LSTM$train$predict
train.SVR.ARMA.MSGARCH = bestresult.SVR.ARMA.MSGARCH$train$predict
train.LSTM.ARMA.MSGARCH = bestresult.LSTM.ARMA.MSGARCH$train$predict
#testing
test.actual = bestresult.SVR.ARMA.MSGARCH.SVR$test$actual
test.SVR = bestresult.SVR.ARMA.MSGARCH.SVR$test$predict
test.LSTM = bestresult.LSTM.ARMA.MSGARCH.LSTM$test$predict
test.SVR.ARMA.MSGARCH = bestresult.SVR.ARMA.MSGARCH$test$predict
test.LSTM.ARMA.MSGARCH = bestresult.LSTM.ARMA.MSGARCH$test$predict

# plot each model separately
ntrain = length(train.actual)
ntest = length(test.actual)
actual = c(train.actual,test.actual)
NA.train = rep(NA,1,ntrain)
NA.test = rep(NA,1,ntest)
par(mfrow=c(1,1))

# ARMA-SVR-MSGARCH
temp.train = c(train.SVR.ARMA.MSGARCH,NA.test)
temp.test = c(NA.train,test.SVR.ARMA.MSGARCH)
plot(actual,type="l", lwd=1, ylab="volatility (%)", xlab="t", ylim=c(min(actual),max(actual)))
lines(temp.train, col="purple", lwd=1)
lines(temp.test, col="red", lwd=1)
title.insample = expression(paste(italic('a'['SVR,t']),' MS-ARMA-GARCH'," In-Sample"))
title.outsample = expression(paste(italic('a'['SVR,t']),' MS-ARMA-GARCH'," Out-Sample"))
legend("topleft",c("Realized Volatility","In-Sample","Out-of-Sample"),
# legend("topleft",c("Realized Volatility",title.insample,title.outsample), 
       col=c("black","purple","red"), lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
title(expression(paste(italic('a'['SVR,t']),' MS-ARMA-GARCH')))

# ARMA-SVR-MSGARCH-SVR
temp.train = c(train.SVR,NA.test)
temp.test = c(NA.train,test.SVR)
plot(actual,type="l", lwd=1, ylab="volatility (%)", xlab="t", ylim=c(min(actual),max(actual)))
lines(temp.train, col="green", lwd=1)
lines(temp.test, col="red", lwd=1)
title.insample = expression(paste(italic(''),'MS-ARMA-GARCH-SVR'," In-Sample"))
title.outsample = expression(paste(italic(''),'MS-ARMA-GARCH-SVR'," Out-of-Sample"))
legend("topleft",c("Realized Volatility","In-Sample","Out-of-Sample"),
# legend("topleft",c("Realized Volatility",title.insample,title.outsample),
       col=c("black","green","red"), lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
title(expression(paste("MS-ARMA-GARCH-SVR")))

# ARMA-LSTM-MSGARCH
temp.train = c(train.LSTM.ARMA.MSGARCH,NA.test)
temp.test = c(NA.train,test.LSTM.ARMA.MSGARCH)
plot(actual,type="l", lwd=1, ylab="volatility (%)", xlab="t", ylim=c(min(actual),max(actual)))
lines(temp.train, col="yellow", lwd=1)
lines(temp.test, col="red", lwd=1)
title.insample = expression(paste(italic('a'['LSTM,t']),' MS-ARMA-GARCH'," In-Sample"))
title.outsample = expression(paste(italic('a'['LSTM,t']),' MS-ARMA-GARCH'," Out-of-Sample"))
legend("topleft",c("Realized Volatility","In-Sample","Out-of-Sample"), 
# legend("topleft",c("Realized Volatility",title.insample,title.outsample), 
  col=c("black","yellow","red"), lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
title(expression(paste(italic('a'['LSTM,t']),' MS-ARMA-GARCH')))

# ARMA-LSTM-MSGARCH-LSTM
temp.train = c(train.LSTM,NA.test)
temp.test = c(NA.train,test.LSTM)
plot(actual,type="l", lwd=1, ylab="volatility (%)", xlab="t", ylim=c(min(actual),max(actual)))
lines(temp.train, col="blue", lwd=1)
lines(temp.test, col="red", lwd=1)
title.insample = expression(paste(italic(''),'MS-ARMA-GARCH-LSTM'," In-Sample"))
title.outsample = expression(paste(italic(''),'MS-ARMA-GARCH-LSTM'," Out-of-Sample"))
legend("topleft",c("Realized Volatility","In-Sample","Out-of-Sample"), 
# legend("topleft",c("Realized Volatility",title.insample,title.outsample),
       col=c("black","blue","red"), lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
title(expression(paste("MS-ARMA-GARCH-LSTM")))

# training plot including msgarch
plot(train.actual,type="l", lwd=1, ylab="volatility (%)", xlab="t", ylim=c(min(train.actual),max(train.actual)))
lines(train.LSTM.ARMA.MSGARCH,col="yellow", lwd=1)
lines(train.SVR.ARMA.MSGARCH, col="purple", lwd=1)
lines(train.SVR,col="green", lwd=1)
lines(train.LSTM,col="blue", lwd=1)
legend("topleft",c("Realized Volatility",expression(paste(italic('a'['SVR,t']),' MS-ARMA-GARCH')),
                   "MS-ARMA-GARCH-SVR",expression(paste(italic('a'['LSTM,t']),' MS-ARMA-GARCH')),"MS-ARMA-GARCH-LSTM"),
       col=c("black","purple","green","yellow","blue"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
title("in-Sample Data")

# testing plot including msgarch
plot(test.actual,type="l", lwd=1, ylab="volatility (%)", xlab="t", ylim=c(min(train.actual),max(train.actual)))
lines(test.SVR.ARMA.MSGARCH, col="purple", lwd=1)
lines(test.SVR,col="green", lwd=1)
lines(test.LSTM.ARMA.MSGARCH,col="yellow", lwd=1)
lines(test.LSTM,col="blue", lwd=1)
legend("topright",c("Realized Volatility",expression(paste(italic('a'['SVR,t']),' MS-ARMA-GARCH')),
                    "MS-ARMA-GARCH-SVR",expression(paste(italic('a'['LSTM,t']),' MS-ARMA-GARCH')),"MS-ARMA-GARCH-LSTM"),
       col=c("black","purple","green","yellow","blue"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5, inset=0.1)
title("Out-of-Sample Data")
##### end of grafik all untuk paper #####
##### MSE of 4 models #####
lossMSE.paper = matrix(NA,ncol=2, nrow=4)
#insample
lossMSE.paper[1,1] = hitungloss(train.actual, train.SVR.ARMA.MSGARCH, method = 'MSE')
lossMSE.paper[2,1] = hitungloss(train.actual, train.SVR, method = 'MSE')
lossMSE.paper[3,1] = hitungloss(train.actual, train.LSTM.ARMA.MSGARCH, method = 'MSE')
lossMSE.paper[4,1] = hitungloss(train.actual, train.LSTM, method = 'MSE')
#outsample
lossMSE.paper[1,2] = hitungloss(test.actual, bestresult.SVR.ARMA.MSGARCH$test$predict, method = 'MSE')
lossMSE.paper[2,2] = hitungloss(test.actual, bestresult.SVR.ARMA.MSGARCH.SVR$test$predict, method = 'MSE')
lossMSE.paper[3,2] = hitungloss(test.actual, bestresult.LSTM.ARMA.MSGARCH$test$predict, method = 'MSE')
lossMSE.paper[4,2] = hitungloss(test.actual, bestresult.LSTM.ARMA.MSGARCH.LSTM$test$predict, method = 'MSE')

colnames(lossMSE.paper) = c("in-sample MSE", "Out-of-sample MSE")
rownames(lossMSE.paper) = c("a.SVR.t MS-ARMA-MSGARCH","MS-ARMA-GARCH-SVR","a.LSTM.t MS-ARMA-MSGARCH","MS-ARMA-GARCH-LSTM")
lossMSE.paper

# beda dengan yang disimpan, pakai yang sesuai tesis
# load("data/revisi_loss_conference.RData")
# losstrain
# losstest
##### end of MSE of 4 models #####

###### end of FOR PAPER ######
