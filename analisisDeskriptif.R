rm(list = ls(all = TRUE))

setwd("C:/File Sharing/Kuliah/TESIS/TESIS dindaulima/MSGARCH_ML/")
source("getDataLuxemburg.R")
source("allfunction.R")

str(mydata)
ts.plot(mydata$close)

head(mydata)
ggplot( data = mydata, aes( date, close )) + geom_line() + scale_x_date(date_labels = "%d-%m-%Y") +theme_bw()+
  xlab("date") + ylab("closing price")
ggplot( data = mydata[1:360,], aes( date, close )) + geom_line() + scale_x_date(date_labels = "%d-%m-%Y") +theme_bw()

ggplot( data = mydata[1:813,], aes( date, return )) + geom_line() + scale_x_date(date_labels = "%d-%m-%Y") +theme_bw() +
  xlab("date") + ylab("return")
ggplot( data = mydata[300:400,], aes( date, return )) + geom_line() + scale_x_date(date_labels = "%d-%m-%Y") +theme_bw() +
  xlab("t") + ylab("log return")

ggplot( data = mydata, aes( date, rv )) + geom_line() + scale_x_date(date_labels = "%d-%m-%Y") +theme_bw() +
  xlab("t") + ylab("realized volatility")

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

hist(logreturn)
lines(x = density(x = mydata$return), col = "red")
dens=density(logreturn)
plot(dens$x,length(logreturn)*dens$y,type="l",xlab="Value",ylab="Count estimate")