# rm(list = ls(all = TRUE))

setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML/")
source("getDataLuxemburg.R")
source("allfunction.R")

#split data train & data test 
dataTrain = mydata%>%filter(date >= as.Date(startTrain) & date <= as.Date(endTrain) )
dataTest = mydata%>%filter(date > as.Date(endTrain) & date <= as.Date(endTest) )
nfore = dim(dataTest)[1];nfore

# uji ADF
adf.test(dataTrain$return, k=1)
adf.test(dataTrain$return)

dim(dataTrain)
############################
# Model ARMA(p,q)
############################
par(mfrow=c(1,2))
acf(dataTrain$return, lag.max = maxlag)
pacf(dataTrain$return, lag.max = maxlag)

#get data only optimal lag
batas = 1.96/sqrt(length(dataTrain$return)-1)
getLagSignifikan(dataTrain$return, batas, maxlag, alpha, na=TRUE)
optARMAlag = getOptLagARMA(dataTrain$return, maxlag = maxlag, batas = batas, alpha = alpha)

# #check conventional arma model
# paramAR = rep(0,1,maxlag)
# paramMA = rep(0,1,maxlag)
# paramAR[optARMAlag$ARlag] = NA
# paramMA[optARMAlag$MAlag] = NA
# armamodel = arima(dataTrain$return,order=c(maxlag,0,maxlag), include.mean=FALSE, fixed=c(paramAR,paramMA))
# coeftest(armamodel)
# AIC(armamodel)
# ujiljungbox(residuals(armamodel))
# ujinormal(residuals(armamodel))
# dataARIMA = data.frame(dataTrain$return, fitted(armamodel),residuals(armamodel))

############################
# UJI LINEARITAS ARMA
############################
chisq.linear = terasvirta.test(ts(dataTrain$return), lag = min(optARMAlag$ARlag), type = "Chisq")
chisq.linear
if(chisq.linear$p.value<alpha){
  cat("Dengan Statistik uji Chisquare, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji Chisquare, Gagal Tolak H0, data linear")
}
F.linear = terasvirta.test(ts(mydata$return), lag = min(optARMAlag$ARlag), type = "F")
F.linear
if(F.linear$p.value<alpha){
  cat("Dengan Statistik uji F, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji F, Gagal Tolak H0, data linear")
}
