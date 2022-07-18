source("allfunction.R")
finvsigmoid(fsigmoid(14))
head(data[,2:5])
tmp = svm(rt~., data=data[,2:5] , kernel ="linear", cost =0.1,scale =FALSE )
tmpnu <- svm(rt~., data=data[,2:5], type='nu-regression', kernel="radial", cost=10, scale=FALSE)
ynu = predict(tmpnu, data[-1,2:5])
tmp <- svm(rt~., data=data[,2:5], kernel="radial", cost=0.3)
ypred = predict(tmp, data[-1,2:5])

plot(ynu, type="l")
lines(ypred,type="l", col="red")
tmpnu$nu

SVRresult = list()
resitrain = resitest = resi = vector()
SVRresult = result.SVR.ARMA.pq
resitrain = SVRresult$train$actual - SVRresult$train$predict
resitest = SVRresult$test$actual - SVRresult$test$predict
resi = c(resitrain,resitest)
at2 = resi^2
plot(at, type="l")
plot(at2, type="l")
lines(rt2, type="l", col="red")
dftmp = data.frame(actual=SVRresult$tes$actual, predict=SVRresult$tes$predict)
head(dftmp,40)

finvsigmoid(5)


exp(-51)
ln(7.095474e-23)
log(0)
