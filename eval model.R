rm(list = ls(all = TRUE))
setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML/")
source("getDataLuxemburg.R")
source("allfunction.R")

load("data/loss_SVR_tune_c.RData")
load("data/result_SVR_tune_c.RData")
load("data/Datauji_SVR_tune_c.RData")
load("data/loss_LSTM_100k.RData")
load("data/bestresult_LSTM_100k.RData")
load("data/Datauji_LSTM_100k.RData")

#split data train & data test 
dataTrain = mydata%>%filter(date >= as.Date(startTrain) & date <= as.Date(endTrain) )
dataTest = mydata%>%filter(date > as.Date(endTrain) & date <= as.Date(endTest) )
nfore = dim(dataTest)[1];nfore

xlabel = "t"
ylabel = "realisasi volatilitas"

##### plot MSGARCH rt ##### 
msgarch.rt = fitMSGARCH(data = dataTrain$return, TrainActual = dataTrain$rv, TestActual=dataTest$rv, nfore=nfore, 
                            GARCHtype="sGARCH", distribution="norm", nstate=2)
msgarch.rt$modelfit
par(mfrow=c(1,1))
makeplot(msgarch.rt$train$actual, msgarch.rt$train$predict, "MSGARCH rt data training", xlabel=xlabel, ylabel=ylabel)
makeplot(msgarch.rt$test$actual, msgarch.rt$test$predict, "MSGARCH rt data testing",xlabel=xlabel, ylabel=ylabel)
actual = c(msgarch.rt$train$actual,msgarch.rt$test$actual)
trainingpred = testingpred = rep(NA,1,length(actual))
trainingpred[1:length(msgarch.rt$train$actual)] = msgarch.rt$train$predict
testingpred[(length(msgarch.rt$train$actual)+1):length(actual)] = msgarch.rt$test$predict

par(mfrow=c(1,1))
plot(actual, type="l", lwd=1, ylab = "realisasi volatilitas", xlab="t")
lines(trainingpred, type="l", col="blue", lwd=1)
lines(testingpred, type="l", col="red", lwd=1)
legend("topleft", as.vector(c("Actual","In-sample","Out-sample")), col=c("black","blue","red"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
title("MSGARCH rt")

Prob = State(object = msgarch.rt$modelfit)
par(mfrow=c(3,1))
plot(Prob, type="filter")
plot.new()
title("Filtered Probability of MSGARCH rt model")

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
par(mfrow=c(2,1))
plot(voltrain[,1], type="l", xlab="Periode ke-t", ylab="volatilitas", main="Regime 1")
plot(voltrain[,2], type="l", xlab="Periode ke-t", ylab="volatilitas", main="Regime 2")


##### plot MSGARCH at.SVR ##### 
SVRresult = list()
resitrain = resitest = resi = vector()

SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)

msgarch.SVR.at = fitMSGARCH(data = resitrain, TrainActual = SVRresult$train$actual^2, TestActual=SVRresult$test$actual^2, nfore=nfore, 
                            GARCHtype="sGARCH", distribution="norm", nstate=2)
msgarch.SVR.at$modelfit
makeplot(msgarch.SVR.at$train$actual, msgarch.SVR.at$train$predict, "at SVR MSGARCH data training", xlabel=xlabel, ylabel=ylabel)
makeplot(msgarch.SVR.at$test$actual, msgarch.SVR.at$test$predict, "at SVR MSGARCH data testing",xlabel=xlabel, ylabel=ylabel)
actual = c(msgarch.SVR.at$train$actual,msgarch.SVR.at$test$actual)
trainingpred = testingpred = rep(NA,1,length(actual))
trainingpred[1:length(msgarch.SVR.at$train$actual)] = msgarch.SVR.at$train$predict
testingpred[(length(msgarch.SVR.at$train$actual)+1):length(actual)] = msgarch.SVR.at$test$predict

par(mfrow=c(1,1))
plot(actual, type="l", lwd=1, ylab = "realisasi volatilitas", xlab="t")
lines(trainingpred, type="l", col="blue", lwd=1)
lines(testingpred, type="l", col="red", lwd=1)
legend("topleft", as.vector(c("Actual","In-sample","Out-sample")), col=c("black","blue","red"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
title(main="at SVR MSGARCH")

Prob = State(object = msgarch.SVR.at$modelfit)
par(mfrow=c(3,1))
plot(Prob, type="filter")
plot.new()
title("Filtered Probability of at SVR MSGARCH model")


##### plot MSGARCH at.LSTM ##### 
LSTMbestresult = list()
resitrain = resitest = resi = vector()

LSTMbestresult = bestresult.LSTM.ARMA.pq
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)

msgarch.LSTM.at = fitMSGARCH(data = resitrain, TrainActual = LSTMbestresult$train$actual^2, TestActual=LSTMbestresult$test$actual^2, nfore=nfore, 
                            GARCHtype="sGARCH", distribution="norm", nstate=2)
msgarch.LSTM.at$modelfit
par(mfrow=c(1,1))
mod = msgarch.LSTM.at
makeplot(mod$train$actual, mod$train$predict, "at LSTM MSGARCH data training", xlabel=xlabel, ylabel=ylabel)
makeplot(mod$test$actual, mod$test$predict, "at LSTM MSGARCH data testing",xlabel=xlabel, ylabel=ylabel)
actual = c(mod$train$actual,mod$test$actual)
trainingpred = testingpred = rep(NA,1,length(actual))
trainingpred[1:length(mod$train$actual)] = mod$train$predict
testingpred[(length(mod$train$actual)+1):length(actual)] = mod$test$predict

par(mfrow=c(1,1))
plot(actual, type="l", lwd=1, ylab = "realisasi volatilitas", xlab="t")
lines(trainingpred, type="l", col="blue", lwd=1)
lines(testingpred, type="l", col="red", lwd=1)
legend("topleft", as.vector(c("Actual","In-sample","Out-sample")), col=c("black","blue","red"),
       lwd=2,cex=0.7,bty = "n", y.intersp=1.5)
title(main="at LSTM MSGARCH")

Prob = State(object = mod$modelfit)
par(mfrow=c(3,1))
plot(Prob, type="filter")
plot.new()
title("Filtered Probability of at LSTM MSGARCH model")
