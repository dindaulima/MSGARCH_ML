setwd("C:/File Sharing/Kuliah/TESIS/tesisdiul/")
source("getDataLuxemburg.R")
source("allfunction.R")

#split data train & data test 
dataTrain = mydata%>%filter(date >= as.Date(startTrain) & date <= as.Date(endTrain) )
dataTest = mydata%>%filter(date > as.Date(endTrain) & date <= as.Date(endTest) )
nfore = dim(dataTest)[1];nfore

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

#check conventional arma model 
maxAR = max(optARMAlag$ARlag)
maxMA = max(optARMAlag$MAlag)
orderAR = rep(0,1,maxAR)
orderMA = rep(0,1,maxMA)
orderAR[optARMAlag$ARlag] = NA
orderMA[optARMAlag$MAlag] = NA
armamodel = arima(dataTrain$return, order=c(maxAR,0,maxMA), include.mean = FALSE,fixed=c(orderAR,orderMA))
coeftest(armamodel)
ujiljungbox(residuals(armamodel))
ujinormal(residuals(armamodel))
adf.test(dataTrain$return)

############################
# UJI LINEARITAS ARMA
############################
chisq.linear = terasvirta.test(ts(dataTrain$return), lag = min(optARMAlag$ARlag), type = "Chisq")
if(chisq.linear$p.value<alpha){
  cat("Dengan Statistik uji Chisquare, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji Chisquare, Gagal Tolak H0, data linear")
}
F.linear = terasvirta.test(ts(mydata$return), lag = min(optARMAlag$ARlag), type = "F")
if(F.linear$p.value<alpha){
  cat("Dengan Statistik uji F, Tolak H0, data tidak linear")
} else {
  cat("Dengan Statistik uji F, Gagal Tolak H0, data linear")
}
