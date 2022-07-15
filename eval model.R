rm(list = ls(all = TRUE))

setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML/")
source("allfunction.R")
source("getDataLuxemburg.R")
source("fitARMA.R")

# load all data
load("data/loss_NN_window5.RData")
load("data/bestresult_NN_window5.RData")
load("data/Datauji_NN_window5.RData")

load("data/loss_SVR_tune_cge_min.RData")
load("data/result_SVR_tune_cge_min.RData")
load("data/Datauji_SVR_tune_cge_min.RData")

load("data/loss_LSTM_window5.RData")
load("data/bestresult_LSTM_window5.RData")
load("data/Datauji_LSTM_window5.RData")

#split data train & data test 
dataTrain = mydata%>%filter(date >= as.Date(startTrain) & date <= as.Date(endTrain) )
dataTest = mydata%>%filter(date > as.Date(endTrain) & date <= as.Date(endTest) )
nfore = dim(dataTest)[1];nfore

xlabel = "t"
ylabel = "realisasi volatilitas"

##### Uji Linearitas ARMA model #####
optARMAlag
chisq.linear = terasvirta.test(ts(mydata$return), lag = min(optARMAlag$ARlag), type = "Chisq");chisq.linear
if(chisq.linear$p.value<alpha){
  cat("Dengan Statistik uji Chisquare, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji Chisquare, Gagal Tolak H0, data linear")
}
F.linear = terasvirta.test(ts(mydata$return), lag = min(optARMAlag$ARlag), type = "F");F.linear
if(F.linear$p.value<alpha){
  cat("Dengan Statistik uji F, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji F, Gagal Tolak H0, data linear")
}


##### plot ACF PACF resi kuadrat identifikasi model GARCH #####
##### plot ACF PACF resi kuadrat ARMA-FFNN #####
NNbestresult = list()
resitrain = resitest = resi = vector()

NNbestresult = bestresult.NN.ARMA.pq
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)
rt2 = data.NN.ARMA.pq$rt^2
at = resi
at2 = resi^2

par(mfrow=c(1,2))
acf.resikuadrat = acf(at2, lag.max = maxlag, type = "correlation", plot=FALSE)
plot(acf.resikuadrat, main="at FFNN kuadrat")
pacf.resikuadrat = pacf(at2, lag.max = maxlag, plot=FALSE)
plot(pacf.resikuadrat, main="at FFNN kuadrat")

# get lag signifikan
batas.at2 = 1.96/sqrt(length(at2)-1)
optlag = getLagSignifikan(at2, maxlag = maxlag, batas = batas.at2, alpha = alpha, na=FALSE)

# LM test
LMtest(resi)

# Uji linearitas GARCH 
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

##### plot ACF PACF resi kuadrat ARMA-SVR #####
SVRresult = list()
resitrain = resitest = resi = vector()

SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)
at = resi
at2 = at^2
rt2 = data.SVR.ARMA.pq$rt^2
time = data.SVR.ARMA.pq$time
base.data = data.frame(time,rt2)
head(base.data)

par(mfrow=c(1,2))
acf.resikuadrat = acf(at2, lag.max = maxlag, type = "correlation", plot=FALSE)
plot(acf.resikuadrat, main="at SVR kuadrat")
pacf.resikuadrat = pacf(at2, lag.max = maxlag, plot=FALSE)
plot(pacf.resikuadrat, main="at SVR kuadrat")

# get lag signifikan
batas.at2 = 1.96/sqrt(length(at2)-1)
optlag = getLagSignifikan(at2, maxlag = maxlag, batas = batas.at2, alpha = alpha, na=FALSE)

# LM test
LMtest(resi)

# Uji linearitas GARCH 
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


##### plot ACF PACF resi kuadrat ARMA-LSTM #####
LSTMbestresult = list()
resitrain = resitest = resi = vector()

LSTMbestresult = bestresult.LSTM.ARMA.pq
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)
rt2 = data.LSTM.ARMA.pq$rt^2
at = resi
at2 = resi^2
time = data.LSTM.ARMA.pq[,1]
base.data = data.frame(time,rt2)
head(base.data)

par(mfrow=c(1,2))
acf.resikuadrat = acf(at2, lag.max = maxlag, type = "correlation", plot=FALSE)
plot(acf.resikuadrat, main="at LSTM kuadrat")
pacf.resikuadrat = pacf(at2, lag.max = maxlag, plot=FALSE)
plot(pacf.resikuadrat, main="at LSTM kuadrat")

# get lag signifikan
batas.at2 = 1.96/sqrt(length(at2)-1)
optlag = getLagSignifikan(at2, maxlag = maxlag, batas = batas.at2, alpha = alpha, na=FALSE)

# LM test
LMtest(resi)

# Uji linearitas GARCH 
chisq.linear = terasvirta.test(ts(rt2), lag = min(optlag$ACFlag), type = "Chisq"); chisq.linear
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


##### Uji Perubahan struktur #####
##### Uji Perubahan struktur FFNN #####
chowtest = ujiperubahanstruktur(data.NN.GARCH, startTrain, endTrain, endTest, alpha)

##### Uji Perubahan struktur SVR #####
chowtest = ujiperubahanstruktur(data.SVR.GARCH, startTrain, endTrain, endTest, alpha)

##### Uji Perubahan struktur LSTM #####
chowtest = ujiperubahanstruktur(data.LSTM.GARCH, startTrain, endTrain, endTest, alpha)



##### plot MSGARCH model #####
##### plot MSGARCH rt ##### 
title = "MSGARCH rt"
data = dataTrain$return
TrainActual = dataTrain$rv
TestActual = dataTest$rv
nrowvoltrain = nrow(dataTrain)
nrowvoltest = nrow(dataTest)
K = 2

msgarch.rt = fitMSGARCH(data = data, TrainActual = TrainActual, TestActual = TestActual, nfore=nfore, 
                            GARCHtype="sGARCH", distribution="norm", nstate=2)
msgarch.rt$modelfit
mod = msgarch.rt

par(mfrow=c(1,1))
makeplot(mod$train$actual, mod$train$predict, paste(title,"data training"), xlabel=xlabel, ylabel=ylabel)
makeplot(mod$test$actual, mod$test$predict, paste(title,"data testing"),xlabel=xlabel, ylabel=ylabel)
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
title(main=title)

Prob = State(object = mod$modelfit)
par(mfrow=c(3,1))
plot(Prob, type="filter")
plot.new()
title(paste("Filtered Probability of",title,"model"))

SR.fit <- ExtractStateFit(mod$modelfit)
msgarch.SR = list(0)
voltrain = matrix(nrow=nrowvoltrain, ncol=K)
voltest = matrix(nrow=nrowvoltest, ncol=K)
for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = data, TrainActual = TrainActual, 
                               TestActual=TestActual, nfore, nstate=2)
  voltrain[,k] = msgarch.SR[[k]]$train$predict
  voltest[,k] = msgarch.SR[[k]]$test$predict
}
par(mfrow=c(3,1))
plot(voltrain[,1], type="l", xlab="Periode ke-t", ylab="volatilitas", main="Regime 1")
plot(voltrain[,2], type="l", xlab="Periode ke-t", ylab="volatilitas", main="Regime 2")
plot.new()
title(paste("volatilitas masing2 regime",title))



##### plot MSGARCH at.NN ##### 
NNbestresult = list()
resitrain = resitest = resi = vector()

NNbestresult = bestresult.NN.ARMA.pq
resitrain = NNbestresult$train$actual - NNbestresult$train$predict
resitest = NNbestresult$test$actual - NNbestresult$test$predict
resi = c(resitrain,resitest)

title = "at NN MSGARCH"
data = resitrain
TrainActual = NNbestresult$train$actual^2
TestActual = NNbestresult$test$actual^2
nrowvoltrain = length(resitrain)
nrowvoltest = length(resitest)

msgarch.NN.at = fitMSGARCH(data = data, TrainActual = TrainActual, TestActual=TestActual, nfore=nfore, 
                           GARCHtype="sGARCH", distribution="norm", nstate=2)
msgarch.NN.at$modelfit
mod = msgarch.NN.at

par(mfrow=c(1,1))
makeplot(mod$train$actual, mod$train$predict, paste(title,"data training"), xlabel=xlabel, ylabel=ylabel)
makeplot(mod$test$actual, mod$test$predict, paste(title,"data testing"),xlabel=xlabel, ylabel=ylabel)
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
title(main=title)

Prob = State(object = mod$modelfit)
par(mfrow=c(3,1))
plot(Prob, type="filter")
plot.new()
title(paste("Filtered Probability of",title,"model"))

SR.fit <- ExtractStateFit(mod$modelfit)
msgarch.SR = list(0)
voltrain = matrix(nrow=nrowvoltrain, ncol=K)
voltest = matrix(nrow=nrowvoltest, ncol=K)
for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = data, TrainActual = TrainActual, 
                               TestActual=TestActual, nfore, nstate=2)
  voltrain[,k] = msgarch.SR[[k]]$train$predict
  voltest[,k] = msgarch.SR[[k]]$test$predict
}
par(mfrow=c(3,1))
plot(voltrain[,1], type="l", xlab="Periode ke-t", ylab="volatilitas", main="Regime 1")
plot(voltrain[,2], type="l", xlab="Periode ke-t", ylab="volatilitas", main="Regime 2")
plot.new()
title(paste("volatilitas masing2 regime",title))


##### plot MSGARCH at.SVR ##### 
SVRresult = list()
resitrain = resitest = resi = vector()

SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)

title = "at SVR MSGARCH"
data = resitrain
TrainActual = SVRresult$train$actual^2
TestActual = SVRresult$test$actual^2
nrowvoltrain = length(resitrain)
nrowvoltest = length(resitest)

msgarch.SVR.at = fitMSGARCH(data = data, TrainActual = TrainActual, TestActual=TestActual, nfore=nfore, 
                            GARCHtype="sGARCH", distribution="norm", nstate=2)
msgarch.SVR.at$modelfit
mod = msgarch.SVR.at


par(mfrow=c(1,1))
makeplot(mod$train$actual, mod$train$predict, paste(title,"data training"), xlabel=xlabel, ylabel=ylabel)
makeplot(mod$test$actual, mod$test$predict, paste(title,"data testing"),xlabel=xlabel, ylabel=ylabel)
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
title(main=title)

Prob = State(object = mod$modelfit)
par(mfrow=c(3,1))
plot(Prob, type="filter")
plot.new()
title(paste("Filtered Probability of",title,"model"))

SR.fit <- ExtractStateFit(mod$modelfit)
msgarch.SR = list(0)
voltrain = matrix(nrow=nrowvoltrain, ncol=K)
voltest = matrix(nrow=nrowvoltest, ncol=K)
for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = data, TrainActual = TrainActual, 
                               TestActual=TestActual, nfore, nstate=2)
  voltrain[,k] = msgarch.SR[[k]]$train$predict
  voltest[,k] = msgarch.SR[[k]]$test$predict
}
par(mfrow=c(3,1))
plot(voltrain[,1], type="l", xlab="Periode ke-t", ylab="volatilitas", main="Regime 1")
plot(voltrain[,2], type="l", xlab="Periode ke-t", ylab="volatilitas", main="Regime 2")
plot.new()
title(paste("volatilitas masing2 regime",title))


##### plot MSGARCH at.LSTM ##### 
LSTMbestresult = list()
resitrain = resitest = resi = vector()

LSTMbestresult = bestresult.LSTM.ARMA.pq
resitrain = LSTMbestresult$train$actual - LSTMbestresult$train$predict
resitest = LSTMbestresult$test$actual - LSTMbestresult$test$predict
resi = c(resitrain,resitest)

title = "at LSTM MSGARCH"
data = resitrain
TrainActual = LSTMbestresult$train$actual^2
TestActual = LSTMbestresult$test$actual^2
nrowvoltrain = length(resitrain)
nrowvoltest = length(resitest)

msgarch.LSTM.at = fitMSGARCH(data = data, TrainActual = TrainActual, TestActual=TestActual, nfore=nfore, 
                            GARCHtype="sGARCH", distribution="norm", nstate=2)
msgarch.LSTM.at$modelfit
mod = msgarch.LSTM.at

par(mfrow=c(1,1))
makeplot(mod$train$actual, mod$train$predict, paste(title,"data training"), xlabel=xlabel, ylabel=ylabel)
makeplot(mod$test$actual, mod$test$predict, paste(title,"data testing"),xlabel=xlabel, ylabel=ylabel)
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
title(main=title)

Prob = State(object = mod$modelfit)
par(mfrow=c(3,1))
plot(Prob, type="filter")
plot.new()
title(paste("Filtered Probability of",title,"model"))

SR.fit <- ExtractStateFit(mod$modelfit)
msgarch.SR = list(0)
voltrain = matrix(nrow=nrowvoltrain, ncol=K)
voltest = matrix(nrow=nrowvoltest, ncol=K)
for(k in 1:K){
  msgarch.SR[[k]] = fitMSGARCH(model.fit = SR.fit[[k]], data = data, TrainActual = TrainActual, 
                               TestActual=TestActual, nfore, nstate=2)
  voltrain[,k] = msgarch.SR[[k]]$train$predict
  voltest[,k] = msgarch.SR[[k]]$test$predict
}
par(mfrow=c(3,1))
plot(voltrain[,1], type="l", xlab="Periode ke-t", ylab="volatilitas", main="Regime 1")
plot(voltrain[,2], type="l", xlab="Periode ke-t", ylab="volatilitas", main="Regime 2")
plot.new()
title(paste("volatilitas masing2 regime",title))


##### Plot MSGARCH-ML #####
##### Plot MSGARCH-FFNN #####
### MSGARCH-FFNN input rt
mod = list()
mod = bestresult.NN.MSGARCH.rt
title = "MSGARCH-FFNN input rt"
par(mfrow=c(1,1))
makeplot(mod$train$actual, mod$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(mod$test$actual, mod$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
# plot training testing
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
title(main=title)

### MSGARCH-FFNN input at
mod = list()
mod = bestresult.NN.MSGARCH.at
title = "MSGARCH-FFNN input at"
par(mfrow=c(1,1))
makeplot(mod$train$actual, mod$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(mod$test$actual, mod$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
# plot training testing
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
title(main=title)

##### Plot MSGARCH-SVR #####
### MSGARCH-SVR input rt
mod = list()
mod = result.SVR.MSGARCH.rt
title = "MSGARCH-SVR input rt"
par(mfrow=c(1,1))
makeplot(mod$train$actual, mod$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(mod$test$actual, mod$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
# plot training testing
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
title(main=title)

### MSGARCH-SVR input rt window5
mod = list()
mod = result.SVR.MSGARCH.rt.window5
title = "MSGARCH-SVR input rt window5"
par(mfrow=c(1,1))
makeplot(mod$train$actual, mod$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(mod$test$actual, mod$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
# plot training testing
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
title(main=title)

### MSGARCH-SVR input at
mod = list()
mod = result.SVR.MSGARCH.at
title = "MSGARCH-SVR input at"
par(mfrow=c(1,1))
makeplot(mod$train$actual, mod$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(mod$test$actual, mod$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
# plot training testing
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
title(main=title)

### MSGARCH-SVR input at window5
mod = list()
mod = result.SVR.MSGARCH.at.window5
title = "MSGARCH-SVR input at window5"
par(mfrow=c(1,1))
makeplot(mod$train$actual, mod$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(mod$test$actual, mod$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
# plot training testing
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
title(main=title)

##### Plot MSGARCH-LSTM #####
### MSGARCH-LSTM input rt
mod = list()
mod = bestresult.LSTM.MSGARCH.rt
title = "MSGARCH-LSTM input rt"
par(mfrow=c(1,1))
makeplot(mod$train$actual, mod$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(mod$test$actual, mod$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
# plot training testing
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
title(main=title)

### MSGARCH-LSTM input at
mod = list()
mod = bestresult.LSTM.MSGARCH.at
title = "MSGARCH-LSTM input at"
par(mfrow=c(1,1))
makeplot(mod$train$actual, mod$train$predict, paste(title,"Train"), xlabel = xlabel, ylabel=ylabel)
makeplot(mod$test$actual, mod$test$predict, paste(title,"Test"), xlabel = xlabel, ylabel=ylabel)
# plot training testing
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
title(main=title)

##### cek loss ####
losstrain.LSTM
losstest.LSTM
losstrain.NN
losstest.NN
losstrain.SVR
losstest.SVR
