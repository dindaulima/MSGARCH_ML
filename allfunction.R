library(nortest) #for function adtest()
library(FinTS) #for function ArchTest()
library(forecast) #for function accuracy()
library(strucchange) # for function sctest()
library(e1071)
library(neuralnet)
library(MSGARCH)
library(tseries) #for terasvirta test
library(lmtest) #for coeftest
library(ggplot2)
library(reticulate) # butuh Rcpp
library(nortsTest)
# library(MCS)

py_config()

getLagSignifikan = function(data, batas, maxlag, alpha, na=FALSE){
  acf = acf(data, lag.max = maxlag)
  acf = acf$acf[2:(maxlag+1)]
  
  pacf = pacf(data, lag.max = maxlag)
  pacf = pacf$acf[1:maxlag]
  
  ACFlag = rep(NA,1,maxlag)
  PACFlag = rep(NA,1,maxlag)
  
  for (i in 1:maxlag){
    if (abs(acf[i]) > batas) {
      ACFlag[i] <- i
    }
    if (abs(pacf[i]) > batas) {
      PACFlag[i] <- i
    }
  }
 
  if(!na){
    PACFlag = PACFlag[!is.na(PACFlag)]
    ACFlag = ACFlag[!is.na(ACFlag)]
  }
  
  cat("PACF lag : ", PACFlag,"\n")
  cat("ACF lag : ", ACFlag,"\n")
  
  return(list(PACFlag = PACFlag, ACFlag = ACFlag))
}

getOptLagARMA = function(data, batas, maxlag, alpha){
  
  lagsig = getLagSignifikan(data, batas, maxlag, alpha, na=TRUE)
  ARlag = lagsig$PACFlag
  MAlag = lagsig$ACFlag

  cat("AR lag : ", ARlag,"\n")
  cat("MA lag : ", MAlag,"\n")
  
  paramAR = rep(NA,1,maxlag)
  paramMA = rep(NA,1,maxlag)
  for(i in 1:maxlag){
    if(is.na(MAlag[i])){
      paramMA[i]=0
    }
    if(is.na(ARlag[i])){
      paramAR[i]=0
    }
  }
  
  optim = FALSE
  iter = 1
  while(!optim){

    print(paramAR)
    print(paramMA)
    nvec = vector()
    
    armamodel = arima(data,order=c(maxlag,0,maxlag), include.mean=FALSE, fixed=c(paramAR,paramMA))
    coef = coeftest(armamodel)
    cat("Kandidat model",iter,"\n")
    aic = AIC(armamodel)
    print(coef)
    cat("AIC",aic,"\n")
    print("uji ljungbox"); print(ujiljungbox(residuals(armamodel)))
    print("uji normal"); print(ujinormal(residuals(armamodel)))

    newcoef = data.frame(dimnames(coef)[[1]],coef[,4])
    colnames(newcoef) = c("param","pval")
    
    newcoef = rbind(newcoef[which(newcoef[,2]>alpha),],newcoef[which(is.na(newcoef$pval)),])
    print(newcoef)
    
    nvec = newcoef[,1]
    
    if(length(nvec)==0){
      optim=TRUE
      next
    }
    
    idxcoefAR = which(substr(nvec,1,2)=="ar")
    idxcoefMA = which(substr(nvec,1,2)=="ma")
    nMA = as.numeric(substr(nvec,3,nchar(nvec))[idxcoefMA])
    nAR = as.numeric(substr(nvec,3,nchar(nvec))[idxcoefAR])
    if(length(nMA)==0){
      nMA = 0
    }
    
    if(length(nAR)==0){
      nAR = 0
    }
    
    if(max(nMA) >= max(nAR)){
      paramMA[max(nMA)]=0
    } else {
      paramAR[max(nAR)]=0
    }

    iter = iter+1
  }
  optARlag = which(is.na(paramAR))
  optMAlag = which(is.na(paramMA))

  cat("AR lag optimal : ", optARlag,"\n")
  cat("MA lag optimal: ", optMAlag,"\n")
  
  return(list(ARlag = optARlag, MAlag = optMAlag))
}


makeData = function(data, datalag, numlag, lagtype){
  l = length(numlag)
  n = nrow(data)
  X <- matrix(ncol=l,nrow=n)

  for(i in 1:l){
    X[,i] <- lag(datalag, numlag[i])
  }

  if(length(numlag)!=0){
    col = gsub(" ", "", paste(lagtype,"Lag",numlag))
    
    df <- data.frame(data, X)
    colnames(df) = c(colnames(data),col)
  } 
  
  return(df)
}

splitData = function(data, startTrain, endTrain, endTest){
  startTest = endTrain
  
  #col1 = waktu
  #col2 = respon
  #col3-end = prediktor

  head(data)
  Train = data%>%filter(time >= as.Date(startTrain) & time <= as.Date(endTrain) )
  Test= data%>%filter(time > as.Date(startTest) & time <= as.Date(endTest) )
  
  #split X y train test
  ytrain = as.matrix(Train[2])
  ytest = as.matrix(Test[2])
  Xtrain = as.matrix(Train[c(3:ncol(Train))])
  Xtest = as.matrix(Test[c(3:ncol(Train))])
  dim(ytrain); dim(ytest); dim(Xtrain); dim(Xtest)
  
  return (list(Xtrain=Xtrain, Xtest=Xtest, ytrain=ytrain, ytest=ytest))
}

ujiperubahanstruktur = function(data, startTrain, endTrain, endTest, alpha){

  # datauji = splitData(data, startTrain, endTrain, endTest)
  # result = sctest(datauji$ytrain~datauji$Xtrain,type="Chow")

  rt.model <- glm(rt2 ~ 1, data = data, family = gaussian)
  result = sctest(rt.model,type="Chow")
  
  if(result$p.value < alpha){
    msg = "Tolak H0, Terdapat perubahan struktur pada data"
  } else {
    msg = "Gagal Tolak H0, Tidak terdapat perubahan struktur pada data"
  }
  print(result)
  print(msg)
  return(result)
}

ujiljungbox = function(resi){
  lags <- c(4,8,12,16,20,24)
  hasilLB<-matrix(0,length(lags),2)
  for(i in seq_along(lags))
  {
    ujiLB=Box.test (resi, lag = lags[i])
    hasilLB[i,1]=ujiLB$statistic
    hasilLB[i,2]=ujiLB$p.value
    rownames(hasilLB)<-lags
  }
  colnames(hasilLB)<-c("statistics","p.value")
  return(hasilLB)
}

ujinormal = function(resi){
  hasil = data.frame(matrix(nrow=3,ncol=3))
  
  sw = shapiro.test(resi)
  hasil[1,] = cbind(sw$method,signif(sw$statistic,6),signif(sw$p.value,6))
  
  ks = ks.test(resi,"pnorm")
  hasil[2,] = cbind(ks$method,signif(ks$statistic,6),signif(ks$p.value,6))
  
  ad = ad.test(resi)
  hasil[3,] = cbind(ad$method,signif(ad$statistic,6),signif(ad$p.value,6))
  
  colnames(hasil) = c("method","statistics","pvalue")
  rownames(hasil) = hasil$method
  hasil = hasil[c(-1)]
  return(hasil)
}

LMtest = function(resi){
  #Autoregressive heteroscedasticity test(H0=no ARCH effect)
  ujiLM = Lm.test(resi)
  return(ujiLM)
}

getlossfunction = function(){
  lossfunction = c("MSE","sMAPE")
    return(lossfunction)
}

hitungloss = function(actual, prediction, method="MSE"){
  # methods = c("MSE","QLIKE","MAPE")
  loss = NA
  if(method=="MSE")
    loss = mean((prediction - actual)^2)
  if(method=="QLIKE")
    # loss = mean(actual/prediction - log(actual/prediction)-1)
    loss = mean(log(prediction)-actual/prediction)
  if(method=="MAPE")
    loss = mean(abs((actual-prediction)/actual)) * 100
  if(method=="MAE")
    loss = mean(abs(actual-prediction))
  if(method=="sMAPE")
    loss = mean((2*abs(actual-prediction))/(abs(actual) +abs(prediction))) * 100
  return(loss)
}

fitMSGARCH = function(model.fit = NULL, data, TrainActual, TestActual, nfore, GARCHtype="sGARCH", distribution="norm", nstate){
  
  lux.spec = CreateSpec(variance.spec = list(model = c(GARCHtype)),
                        distribution.spec = list(distribution = c(distribution)),
                        switch.spec = list(K=nstate))

  if(is.null(model.fit)){
    summary(lux.spec)
    fit.ml = FitML(spec=lux.spec, data = data)
    model.fit = fit.ml
  }
  
  volatilitas <- Volatility(object = model.fit)
  fitted = volatilitas^2

  #forecast
  pred <- predict(object = model.fit, nahead = nfore, do.return.draw = TRUE)
  pred = (pred$vol)^2

  #resi
  resiMSGARCHtrain = TrainActual - fitted
  resiMSGARCHtest = TestActual - pred
  resiMSGARCH = c(resiMSGARCHtrain, resiMSGARCHtest)

  resultMSGARCH = list()
  resultMSGARCH$modelfit = model.fit
  resultMSGARCH$modelspec = lux.spec
  resultMSGARCH$train$actual = TrainActual
  resultMSGARCH$train$predict = fitted
  resultMSGARCH$train$residual = resiMSGARCHtrain
  resultMSGARCH$test$actual = TestActual
  resultMSGARCH$test$predict = pred
  resultMSGARCH$test$residual = resiMSGARCHtest
  
  return(resultMSGARCH)
}

makeplot = function(actual, prediction, title, xlabel=NULL, ylabel=NULL){
  # par(mfrow=c(1,1))
  ymin = min(c(min(prediction),min(actual)))
  ymax = max(c(max(prediction),max(actual)))
  plot(actual,type="l", ylim=c(ymin,ymax), xlab=xlabel, ylab=ylabel)
  lines(prediction,col="red")
  title(title)
}

sliding_window = function(x, y, window_size){
  newsize = length(y)-window_size
  ynew = vector(length=newsize)
  xnew = matrix(nrow=newsize, ncol=(ncol(x)*(window_size+1)))

  for (i in 1:newsize){
    ynew[i] = y[(i+window_size)]
    feature = as.matrix(x[i,])

    for(j in 1:window_size){
      feature = cbind(feature,as.matrix(x[i+j,]))
    }
    
    feature = as.vector(feature)
    xnew[i,] = feature 
  }

  xlab = gsub(" ","",paste(colnames(x),"_",window_size))
  for(j in 1:(window_size-1)){
    xlab = c(xlab, gsub(" ","",paste(colnames(x),"_",window_size-j)))
  }
  xlab = c(xlab,colnames(x))
  colnames(xnew) = xlab
  
  return (list(x=xnew,y=ynew))
}

fitLSTM = function(data, startTrain, endTrain, endTest, node_hidden, epoch, allow_negative=0, window_size = 1){
  datauji = splitData(data, startTrain, endTrain, endTest)
  ytrain = datauji$ytrain
  ytest = datauji$ytest
  Xtrain = datauji$Xtrain
  Xtest = datauji$Xtest
  cat("ytrain shape :",dim(ytrain),"\n", "ytest shape :",dim(ytest),"\n","Xtrain shape :",dim(Xtrain),"\n", "Xtest shape :", dim(Xtest),"\n")
  
  node_hidden = as.vector(node_hidden)
  n_neuron = length(node_hidden)
  window_size = as.integer(window_size)
  resultLSTM = list()

  if(window_size>1){
    ytrain_reducted = ytrain[(window_size+1):length(ytrain)]
    # ytest_reducted = ytest[(window_size+1):length(ytest)]
  } else {
    ytrain_reducted = ytrain
    # ytest_reducted = ytest
  }
  
  # ytest tidak direduksi karena mengambil window_size data terakhir dari data training
  ytest_reducted = ytest
  
  trainpredict =  matrix(0,length(ytrain_reducted),n_neuron)
  testpredict =  matrix(0,length(ytest_reducted),n_neuron)

  for(i in 1:n_neuron){
    cat("hidden node :",node_hidden[i],"\n")

    resultLSTM[[i]] = lstmfit(Xtrain, ytrain, Xtest, ytest, node_hidden[i], epoch, allow_negative, window_size)

    resultLSTM[[i]]$train = py_to_r(resultLSTM[[i]]$train)
    resultLSTM[[i]]$test = py_to_r(resultLSTM[[i]]$test)

    trainpredict[,i] = resultLSTM[[i]]$train$predict
    testpredict[,i] = resultLSTM[[i]]$test$predict
  }
  resultlabel = paste("Neuron", node_hidden)

  #accuracy measurement 
  lossfunction = getlossfunction()
  len.loss = length(lossfunction)
  loss = matrix(0,n_neuron,(2*len.loss))
  colnames(loss) = c(paste(lossfunction,"training"),paste(lossfunction,"testing"))
  for(i in 1:n_neuron){
    for(j in 1:len.loss){
      loss[i,j] = hitungloss(ytrain_reducted, trainpredict[,i], method = lossfunction[j])
      loss[i,j+len.loss] = hitungloss(ytest_reducted, testpredict[,i], method = lossfunction[j])
    }
  }
  
  opt_idx =   which.min(rowSums(loss[,1:len.loss]))
  cat("hidden node optimal",node_hidden[opt_idx],"\n")
  resultlabel = c(resultlabel,"opt_idx")
  
  resultLSTM[[i+1]] = opt_idx
  names(resultLSTM) <- resultlabel
  
  return(resultLSTM)
}

fitSVR = function(data, startTrain, endTrain, endTest, kernel="radial", tune_C=TRUE, tune_gamma=TRUE, tune_eps=TRUE){
  datauji = splitData(data, startTrain, endTrain, endTest)
  head(datauji$Xtrain)
  resultSVR = list(0)
  
  #inisialisasi
  lossfunction = getlossfunction()
  len.loss = length(lossfunction)

  #default -> mengikuti default svm di R
  opt_c = 1
  opt_eps = 0.1
  opt_gamma = 1/ncol(datauji$Xtrain)
  
  #training
  x.train = datauji$Xtrain
  y.train = datauji$ytrain
  colnames(y.train) = c("y")
  t.train = c(1:length(y.train))
  data.train <- data.frame(y=y.train,x=x.train)
  head(data.train)
  
  minloss = 999999
  if(tune_C){
    # tuning parameter C
    iter.range = seq(10^-1,10^2,0.1)
    iter = length(iter.range)
    c_loss = matrix(0,iter,len.loss)
    colnames(c_loss) = lossfunction

    par(mfrow=c(1,1))
    plot(t.train,y.train,ylim=c(min(y.train),max(y.train)),type='l')

    for( i in 1:iter) {
      svm.fit <- svm(y~., data=data.train, type='eps-regression', kernel=kernel, cost=iter.range[i])
      pred <- predict(svm.fit, data.train[-1])
      points(x=t.train, y=pred, type='l', col=i)

      for(j in 1:len.loss){
        c_loss[i,j] = hitungloss(data.train$y, pred, method = lossfunction[j])
      }
    }

    opt_idxc = which.min(rowSums(c_loss))
    opt_c_new = iter.range[opt_idxc]

    if (minloss > sum(c_loss[opt_idxc,])){
      minloss = sum(c_loss[opt_idxc,])
      opt_c = opt_c_new
    }
  }
  
  if(tune_gamma){
    # tuning parameter gamma
    iter.range = seq(10^-1,10^2,0.1)
    iter = length(iter.range)
    gamma_loss = matrix(0,iter,len.loss)
    colnames(gamma_loss) = lossfunction

    plot(t.train,y.train,ylim=c(min(y.train),max(y.train)),type='l')
    for( i in 1:iter) {
      svm.fit <- svm(y~., data=data.train, type='eps-regression', kernel=kernel, cost=opt_c, gamma=iter.range[i])
      pred <- predict(svm.fit, data.train[-1])
      points(x=t.train, y=pred, type='l', col=i)

      for(j in 1:len.loss){
        gamma_loss[i,j] = hitungloss(data.train$y, pred, method = lossfunction[j])
      }
    }

    opt_idxgamma = which.min(rowSums(gamma_loss))
    opt_gamma_new = iter.range[opt_idxgamma]

    if (minloss > sum(gamma_loss[opt_idxgamma,])){
      minloss = sum(gamma_loss[opt_idxgamma,])
      opt_gamma = opt_gamma_new
    }
  }
  
  if(tune_eps){
    # tuning parameter epsilon
    iter.range = seq(10^-1,1,0.1)
    iter = length(iter.range)
    eps_loss = matrix(0,iter,len.loss)
    colnames(eps_loss) = lossfunction

    # par(mfrow=c(1,1))
    plot(t.train,y.train,ylim=c(min(y.train),max(y.train)),type='l')

    for( i in 1:iter) {
      svm.fit <- svm(y~., data=data.train, type='eps-regression', kernel=kernel, cost=opt_c, gamma=opt_gamma, epsilon = iter.range[i])
      pred <- predict(svm.fit, data.train[-1])
      points(x=t.train, y=pred, type='l', col=i)
      for(j in 1:len.loss){
        eps_loss[i,j] = hitungloss(data.train$y, pred, method = lossfunction[j])
      }
    }

    opt_idxeps = which.min(rowSums(eps_loss))
    opt_eps_new = iter.range[opt_idxeps]
    
    if (minloss > sum(eps_loss[opt_idxeps,])){
      minloss = sum(eps_loss[opt_idxeps,])
      opt_eps = opt_eps_new
    }
  }
  
  svm.fit = svm(y~., data=data.train, type='eps-regression', kernel=kernel, cost=opt_c, gamma=opt_gamma, epsilon=opt_eps)
  
  pred = predict(svm.fit, data.train[-1])
  plot(t.train,y.train,ylim=c(min(y.train),max(y.train)),type='l')
  points(x=t.train, y=pred, type='l', col="red")
  legend("bottomright", as.vector(c("Actual","Predicted")), fill=c("black","red"))


  resulttrain = data.frame(t.train, y.train, pred)
  colnames(resulttrain) = c("idx","actual","predict")
  head(resulttrain)

  # testing
  x.test = datauji$Xtest
  y.test = datauji$ytest
  colnames(y.test) = c("y")
  t.test = c(1:length(y.test))
  datatest <- data.frame(y=y.test,x=x.test)
  fore = predict(svm.fit, datatest[-1])
  plot(t.test,y.test,ylim=c(min(y.test),max(y.test)),type='l')
  points(x=t.test, y=fore, type='l', col="red")
  legend("bottomright", as.vector(c("Actual","Predicted")), fill=c("black","red"))

  dim(datatest)
  length(fore)

  resulttest = data.frame(t.test, y.test, fore)
  colnames(resulttest) = c("idx","actual","predict")
  head(resulttest)

  svm.fit
  resultSVR[[1]] = resulttrain
  resultSVR[[2]] = resulttest
  resultSVR[[3]] = svm.fit
  names(resultSVR) = c("train","test","model.fit")
  return (resultSVR)
}

fitNN = function(data, startTrain, endTrain, endTest, neuron, act.fnc = "logistic", linear.output=FALSE, scale=TRUE){
  n_neuron = length(neuron)

  dataori = splitData(data, startTrain, endTrain, endTest)
  ytrain.ori = dataori$ytrain
  ytest.ori = dataori$ytest

  if(scale){
    #standardized data
    mean.y = mean(data[,2])
    sd.y = sd(data[,2])
    data.scaled = scale(data[,-1])
    data.scaled = data.frame(data[,1],data.scaled)
    colnames(data.scaled) = colnames(data)

  } else {
    data.scaled = data
  }

  datauji = splitData(data.scaled, startTrain, endTrain, endTest)
  head(datauji$Xtrain)
  ytrain = datauji$ytrain
  ytest = datauji$ytest
  Xtrain = datauji$Xtrain
  Xtest = datauji$Xtest
  
  colnames(ytrain) = c("y")
  colnames(ytest) = c("y")
  
  dataTrain <- data.frame(ytrain,Xtrain)
  dataTest <- data.frame(ytest,Xtest)
  head(dataTrain)
  
  resultNN = list(0)
  trainpredict =  matrix(0,length(ytrain),n_neuron)
  testpredict =  matrix(0,length(ytest),n_neuron)
  
  #index 
  ttrain = c(1:length(ytrain))
  ttest = c(1:length(ytest))
  
  
  n_Ytrain = length(ytrain)
  n_fore = length(ytest)
  
  best.model_NN = list()

  for(k in seq_along(neuron)){
    result = list()
    set.seed(1234)
    model_NN = neuralnet(y ~ . , data=dataTrain, hidden=neuron[k], act.fct = act.fnc, linear.output=linear.output, likelihood=TRUE)  
    trainpredict[,k] = (as.ts(unlist(model_NN$net.result)))
    # print(k)
    # print(head(trainpredict[,k]))
    
    best.model_NN[[k]] = model_NN

    #architecture of neural network
    #plot(best.model_NN[[k]])
    
    #forecast k-step ahead
    ypred = c( trainpredict[,k] ,rep(NA,n_fore))
    j=1
    for(i in (n_Ytrain+1):(n_Ytrain+n_fore)){
      X = t(as.matrix(Xtest[j,]))
      ypred[i] = neuralnet::compute(best.model_NN[[k]],covariate = X)$net.result
      
      j=j+1
    }
    testpredict[,k] = ypred[(n_Ytrain+1):(n_Ytrain+n_fore)] 

    if(scale){
      trainpredict[,k] = trainpredict[,k] * sd.y + mean.y
      testpredict[,k] = testpredict[,k] * sd.y + mean.y
    } 

    result$train = data.frame(ttrain, ytrain.ori, trainpredict[,k])
    colnames(result$train) = c("idx","actual","predict")
    
    result$test = data.frame(ttest, ytest.ori, testpredict[,k])
    colnames(result$test) = c("idx","actual","predict")
    result$model_NN = model_NN

    resultNN[[k]] = result

  }
  
  #column names for matrix forecast result
  colnames(trainpredict)= paste("Neuron", neuron)
  colnames(testpredict)= paste("Neuron", neuron)

  #accuracy measurement
  lossfunction = getlossfunction()
  len.loss = length(lossfunction)
  loss = matrix(0,n_neuron,2*len.loss)
  colnames(loss) = c(paste(lossfunction,"Training"),paste(lossfunction,"Testing"))
  
  for(i in 1:n_neuron){
    for(j in 1:len.loss){
      loss[i,j] = hitungloss(ytrain.ori, trainpredict[,i], method = lossfunction[j])
      loss[i,j+len.loss] = hitungloss(ytest.ori, testpredict[,i], method = lossfunction[j])
    }
  
  }
  
  loss
  opt_idx =   which.min(rowSums(loss[,1:len.loss]))
  cat("hidden node optimal",neuron[opt_idx],"\n")

  resultlabel = paste("Neuron", neuron)

  resultNN[[i+1]] = opt_idx
  resultlabel = c(resultlabel,"opt_idx")
  
  resultNN[[i+2]] = best.model_NN
  resultlabel = c(resultlabel,"model_NN")
  
  names(resultNN) <- resultlabel

  return(resultNN)
}

forecastLSTM = function(lstm.model, X){
  X = as.matrix(X)
  fore = lstmforecast(lstm.model,X)
  return(fore)
}