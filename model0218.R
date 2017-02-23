setwd("logoscrape/model/model0218")

library(mxnet)
library(unbalanced)
library(e1071)
library(MASS)
library(xgboost)
library(party)
library(randomForest)
require(neuralnet) 
require(nnet) 
library(class)
library(mxnet)
library(caret)
library(Matrix)
library(glmnet)
library(unbalanced)
library(e1071)
library(rpart)
library(fastAdaboost)
library(ggplot2)
library(readr)

DF = read.csv("modeldata.csv")
DF = DF[,-c(1,2,12:19)]
#'''transform date between 0 and 1.'''
DF[,9] = as.numeric(substr(formatC(DF[,9], width=7, flag="0"),1,3))
DF$agree = as.factor(ifelse(DF$agree==T, 1, 0))


set.seed(2016)
sampling = sample(1:nrow(DF),nrow(DF)*0.7,replace=F)
train = DF[sampling,]
test = DF[-sampling,]
testtarget = na.exclude(test)$agree


#'''preprocessing testset'''
for ( i in c(1,4:9))
{
  test[,i] = (test[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
test = test[,-c(2,3)]

#'''preprocessing trainset'''
train = ubUnder(train, Y=train$agree)$X
for ( i in c(1,4:9))
{
  train[,i] = (train[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
train = train[,-2]

#'''(2,100),(0.01,100)'''
result =  svm(agree ~ ., data = train, gamma = 2^-4.5, cost=2^17.5)
Ypred = predict(result,newdata=na.exclude(test))
sum(testtarget==Ypred)/length(testtarget)
print( table(Ypred,testtarget) )

#'''Grid Search SVM'''
gammarange = seq(-15,15,0.25)
costrange = seq(16,20,0.25)
gs = c(0)
for (g in gammarange)
{
	for (C in costrange)
	{
		result =  svm(agree ~ ., data = train, gamma = 2^(g), cost = 2^(C))
		Ypred = predict(result,newdata=na.exclude(test))
		gs[length(gs)+1] = sum(testtarget==Ypred)/length(testtarget)
		
	}
}
gridsearchmatrix = matrix(gs[-1], length(costrange), length(gammarange))
rownames(gridsearchmatrix) = paste("2^",costrange,sep="")
colnames(gridsearchmatrix) = paste("2^",gammarange,sep="")
print(paste("C:",rownames(gridsearchmatrix)[which(gridsearchmatrix == max(gridsearchmatrix), arr.ind = T)[1]],
		", gamma:",colnames(gridsearchmatrix)[which(gridsearchmatrix == max(gridsearchmatrix), arr.ind = T)[2]]))

write.csv(gridsearchmatrix,"gstable1414.csv")




ggplot(data=DF, aes(x=log(EX15), y=log(IM15),colour=agree)) + 
	geom_point(size=3, alpha=.6)

ggplot(data=DF, aes(x=city,colour=agree)) + 
	geom_histogram()


