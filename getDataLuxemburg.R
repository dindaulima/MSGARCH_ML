library(SciViews) #for function ln()
library(dplyr)

luxemburg = read.csv("data/Franklin Global Sukuk Fund Luxemburg.csv")

mydata = data.frame(luxemburg)
colnames(mydata) = c("date","close","open","high","low","change")

mydata$date = as.Date(mydata$date, format = c("%d/%m/%Y"))

mydata$close = gsub(",", ".", mydata$close)
mydata$close = as.numeric(mydata$close)

mydata$open = gsub(",", ".", mydata$open)
mydata$open = as.numeric(mydata$open)

mydata$high = gsub(",", ".", mydata$high)
mydata$high = as.numeric(mydata$high)

mydata$low = gsub(",", ".", mydata$low)
mydata$low = as.numeric(mydata$low)

mydata$change = gsub(",", ".", mydata$change)
mydata$change = gsub("%", "", mydata$change)
mydata$change = as.numeric(mydata$change)

#sort data
mydata <-mydata[order(mydata$date),]

#hitung return
mydata$return <- 100 * (log(mydata$close) - log(lag(mydata$close)))

# filter data
startdate = "2018-11-21"
enddate = "2021-12-31"
mydata = mydata%>%filter(date >= as.Date(startdate) & date <= as.Date(enddate) )

#hitung realized volatility
rv = mydata$return^2
mydata$rv = rv

mydata = na.omit(mydata)

#inisialisasi
alpha = 0.05
startTrain = "2018-11-21"
endTrain = "2021-04-30"
endTest = "2021-12-31"
maxlag = 24
