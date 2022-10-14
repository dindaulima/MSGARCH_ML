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
# di fitARMA.R 
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
source_python('LSTM_fit.py')

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
t(wi)
t(wf)
t(wc)
t(wo)
neu = paste('neuron',seq(1,opt_idxLSTM.ARMA,1))
ui = matrix(paste(modtemp$U_i,neu),ncol=ncol(modtemp$U_i),nrow=length(neu))
uf = matrix(paste(modtemp$U_f,neu),ncol=ncol(modtemp$U_f),nrow=length(neu))
uc = matrix(paste(modtemp$U_c,neu),ncol=ncol(modtemp$U_c),nrow=length(neu))
uo = matrix(paste(modtemp$U_o,neu),ncol=ncol(modtemp$U_o),nrow=length(neu))
t(ui)
t(uf)
t(uc)
t(uo)

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
#### end of Uji LM residual ARMA-ML ####

####### Pemodelan GARCH #######
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

#### grafik perbandingan ARMA-FFNN ####
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
xlabel = "t"
ylabel = "return kuadrat (%)"
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
xlabel = "t"
ylabel = "return kuadrat (%)"
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

# masih sampai sini dokumen bab 4 nya
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

#bobot & arsitektur NN
plot(result.NN.ARMA.GARCH[[opt_idxNN.ARMA.GARCH]]$model_NN)
plot(result.NN.ARMA.GARCH[[opt_idxNN.ARMA.GARCH]]$model_NN, show.weights = FALSE)
result.NN.ARMA.GARCH[[opt_idxNN.ARMA.GARCH]]$model_NN$result.matrix

#### grafik perbandingan ARMA-GARCH-FFNN ####
title = "ARMA-GARCH-FFNN"
xlabel = "t"
ylabel = "realisasi volatilitas (%)"
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
xlabel = "t"
ylabel = "return kuadrat (%)"
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
modtemp
wi = matrix(paste(modtemp$W_i,var),ncol=ncol(modtemp$W_i),nrow=length(var))
wf = matrix(paste(modtemp$W_f,var),ncol=ncol(modtemp$W_f),nrow=length(var))
wc = matrix(paste(modtemp$W_c,var),ncol=ncol(modtemp$W_c),nrow=length(var))
wo = matrix(paste(modtemp$W_o,var),ncol=ncol(modtemp$W_o),nrow=length(var))
t(wi)
t(wf)
t(wc)
t(wo)

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
neu = paste('neuron',seq(1,opt_idxLSTM.ARMA.GARCH,1))
ui = matrix(paste(modtemp$U_i,neu),ncol=ncol(modtemp$U_i),nrow=length(neu))
uf = matrix(paste(modtemp$U_f,neu),ncol=ncol(modtemp$U_f),nrow=length(neu))
uc = matrix(paste(modtemp$U_c,neu),ncol=ncol(modtemp$U_c),nrow=length(neu))
uo = matrix(paste(modtemp$U_o,neu),ncol=ncol(modtemp$U_o),nrow=length(neu))
t(ui)
t(uf)
t(uc)
t(uo)

#### grafik perbandingan ARMA-GARCH-LSTM ####
title = "ARMA-GARCH LSTM"
xlabel = "t"
ylabel = "return kuadrat (%)"
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