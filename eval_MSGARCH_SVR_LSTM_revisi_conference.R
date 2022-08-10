rm(list = ls(all = TRUE))
library(MSGARCH)

setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML")
source("allfunction.R")
source("getDataLuxemburg.R")

#inisialisasi
#split data train & data test 
dataTrain = mydata%>%filter(date >= as.Date(startTrain) & date <= as.Date(endTrain) )
dataTest = mydata%>%filter(date > as.Date(endTrain) & date <= as.Date(endTest) )
nfore = dim(dataTest)[1];nfore
node_hidden = c(1:20)
n.neuron = length(node_hidden)
lossfunction = c("MSE")
len.loss = length(lossfunction)

################# LOSSTEST ambil dari sini #################
# saat revisi sudah dibenahi jadi mengambil dari sini semua
load("data/revisi_loss_conference.RData")
load("data/revisi_result_conference.RData")
load("data/revisi_bestresult_conference.RData")
load("data/revisi_datauji_conference.RData")

##### make all plot #####
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
