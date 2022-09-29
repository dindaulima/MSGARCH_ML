rm(list = ls(all = TRUE))
setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML/")
source("getDataLuxemburg.R")
source("allfunction.R")

#inisialisasi
neuron = c(1:20)
n.neuron = length(neuron)

#split data train & data test 
dataTrain = mydata%>%filter(date >= as.Date(startTrain) & date <= as.Date(endTrain) )
dataTest = mydata%>%filter(date > as.Date(endTrain) & date <= as.Date(endTest) )
nfore = dim(dataTest)[1];nfore

#### analisis deskripsi ####
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
#### end of analisis deskripsi ####
#### Pemodelan ARMA ####
# di fitARMA.R 
#### end of pemodelan ARMA ####

#### load ML result ####
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
#### end of load ML result ####

#### Pemodelan ARMA-ML ####
#cek struktur input ARMA-ML
head(data.NN.ARMA)
head(data.SVR.ARMA)
head(data.LSTM.ARMA)

#detail ARMA-FFNN 

##FFNN
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
  loss[i,4] = hitungloss(trainactual, trainpred, method = "sMAPE")
}
loss = data.frame(loss)
opt_idxNN = which.min(loss$MSEtest);opt_idxNN
lossNN = loss
rownames(lossNN) = paste('Neuron',neuron)
lossNN

#bobot & arsitektur NN
plot(result.NN.ARMA[[opt_idxNN]]$model_NN)
plot(result.NN.ARMA[[opt_idxNN]]$model_NN, show.weights = FALSE)
result.NN.ARMA[[opt_idxNN]]$model_NN$result.matrix

# grafik perbandingan
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

##LSTM
result = list()
result = result.LSTM.ARMA
data = data.LSTM.ARMA
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
opt_idxLSTM = which.min(loss[,2]);opt_idxLSTM
lossLSTM = loss

MSE.NN.LSTM = data.frame(MSE_FFNN = lossNN[,2],MSE_LSTM = lossLSTM[,2])
rownames(MSE.NN.LSTM) = paste('Neuron',neuron)
MSE.NN.LSTM


#Model ARMA-SVR
result.SVR.ARMA
#### end of pemodelan ARMA-ML ####

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

#### Pemodelan GARCH ####
# identifikasi model GARCH
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

# identifikasi ARMA-FFNN-GARCH
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

# identifikasi ARMA-SVR-GARCH
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
# identifikasi ARMA-LSTM-GARCH
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

#### end of Pemodelan GARCH ####