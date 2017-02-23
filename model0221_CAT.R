setwd("logoscrape/model/model0218")

library(rpart)
require(neuralnet) 
require(nnet) 
library(mxnet)
library(caret)
library(Matrix)
library(glmnet)
library(unbalanced)
library(e1071)
library(ggplot2)
library(fastICA)


DF = read.csv("modeldata.csv")
DF[,12] = as.numeric(substr(DF[,12],1,2))
DF[,14] = as.numeric(substr(DF[,14],1,2))
DF[,16] = as.numeric(substr(DF[,16],1,2))
DF[,18] = as.numeric(substr(DF[,18],1,2))
DF$cat1 = ifelse(DF[,12]<34,"C",ifelse(DF[,12]>58,"J","G"))
DF$cat2 = ifelse(DF[,14]<34,"C",ifelse(DF[,14]>58,"J","G"))
DF$cat3 = ifelse(DF[,16]<34,"C",ifelse(DF[,16]>58,"J","G"))
DF$cat4 = ifelse(DF[,18]<34,"C",ifelse(DF[,18]>58,"J","G"))
DF[which(DF$cat1 == "G" | DF$cat2 == "G" | DF$cat3 == "G" | DF$cat4 == "G"),24] = 1
DF[which(DF$cat1 == "C" | DF$cat2 == "C" | DF$cat3 == "C" | DF$cat4 == "C"),25] = 1
DF[which(DF$cat1 == "J" | DF$cat2 == "J" | DF$cat3 == "J" | DF$cat4 == "J"),26] = 1
DF[is.na(DF)] = 0
DF = DF[,c(3:11,24:26)]
#DF[,9] = as.numeric(substr(formatC(as.character(DF[,9]), width=7, flag="0"),1,3))
colnames(DF)[9] = "date"

DF[,2] = as.factor(DF[,2])
DF[,10] = as.factor(DF[,10])
DF[,11] = as.factor(DF[,11])
DF[,12] = as.factor(DF[,12])

set.seed(2014)
sampling = sample(1:nrow(DF),nrow(DF)*0.7,replace=F)
train = DF[sampling,]
train = ubUnder(train, Y=train$agree)$X
test = DF[-sampling,]
testtarget = test$agree


real = real[(!is.na(real$isC))& (real$符合行業別 == T),]
nrow(DF[(DF$BAN_1 %in% real$BAN_1) & (DF$agree==T),] )
 & (real$attendevent >0)

result = rpart(factor(agree)~. , data=train)
plot(result)
text(result)
Ypred=predict(result,newdata=test[,-3],type="class")
sum(testtarget==Ypred)/length(testtarget)
print( table(Ypred,testtarget) )

###################################     SVM     #####################################
real = read.csv("threedata2.csv")
real[,6] = as.numeric(substr(formatC(real[,6], width=7, flag="0"),1,3))
colnames(real)[6] = "date"
DF = read.csv("modeldata.csv")
DF = DF[,-c(1,2,12:19)]
DF = na.exclude(DF)
#'''transform date between 0 and 1.'''
DF[,9] = as.numeric(substr(formatC(DF[,9], width=7, flag="0"),1,3))
DF$agree = ifelse(DF$agree==T, 1, 0)
colnames(DF)[9] = "date"


for(year in 2001:2017)
{
set.seed(year)
sampling = sample(1:nrow(DF),nrow(DF)*0.7,replace=F)
train = DF[sampling,]
test = DF[-sampling,]
testtarget = test$agree


#'''preprocessing testset'''
for ( i in c(1,4:9))
{
  test[,i] = (test[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
test = test[,-c(2,3)]

#'''preprocessing realset'''
for ( i in c("CAPITAL","EX15","date","attendevent"))
{
  real[,i] = (real[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
real = real[,c("CAPITAL","EX15","date","attendevent")]

#'''preprocessing trainset'''
train = ubUnder(train, Y=factor(train$agree))$X
for ( i in c(1,4:9))
{
  train[,i] = (train[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
train = train[,-2]


result1=svm(factor(agree)~.,data=train[,c(1,2,3,6,8)],gamma=2^(7.75), cost=2^(5.25))
pred=predict(result1,newdata=test)
print(year)
print(sum(testtarget==pred)/length(testtarget))
print(table(pred,testtarget)[2,2]/(table(pred,testtarget)[2,1]+table(pred,testtarget)[2,2]))
print( table(pred,testtarget) )

#presult = prcomp(train[,-c(2)],center=F)
#result1=naiveBayes(factor(agree)~.,data=data.frame(presult$x[,1:4],train["agree"]))
#pred=predict(result1,newdata=data.matrix(test) %*% data.matrix(presult$rotation[,1:4]))
#print(year)
#print(sum(testtarget==pred)/length(testtarget))
#print(table(pred,testtarget)[2,2]/(table(pred,testtarget)[2,1]+table(pred,testtarget)[2,2]))
#print( table(pred,testtarget) )

pred.real <- predict(result1, newdata=data.matrix(real))
print(table(pred.real))

}





####################################     NB     #####################################
real = read.csv("threedata2.csv")
real[,6] = as.numeric(substr(formatC(real[,6], width=7, flag="0"),1,3))
colnames(real)[6] = "date"
DF = read.csv("modeldata.csv")
DF = DF[,-c(1,2,12:19)]
DF = na.exclude(DF)
#'''transform date between 0 and 1.'''
DF[,9] = as.numeric(substr(formatC(DF[,9], width=7, flag="0"),1,3))
DF$agree = ifelse(DF$agree==T, 1, 0)
colnames(DF)[9] = "date"


for(year in 2001:2017)
{
set.seed(year)
sampling = sample(1:nrow(DF),nrow(DF)*0.7,replace=F)
train = DF[sampling,]
test = DF[-sampling,]
testtarget = test$agree


#'''preprocessing testset'''
for ( i in c(1,4:9))
{
  test[,i] = (test[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
test = test[,-c(2,3)]

#'''preprocessing realset'''
for ( i in c("CAPITAL","EX15","date","attendevent"))
{
  real[,i] = (real[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
real = real[,c("CAPITAL","EX15","date","attendevent")]

#'''preprocessing trainset'''
train = ubSMOTE(train, Y=factor(train$agree))$X
for ( i in c(1,4:9))
{
  train[,i] = (train[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
train = train[,-2]


result1=naiveBayes(factor(agree)~.,data=train[,c(1,2,3,6,8)])
pred=predict(result1,newdata=test)
print(year)
print(sum(testtarget==pred)/length(testtarget))
print(table(pred,testtarget)[2,2]/(table(pred,testtarget)[2,1]+table(pred,testtarget)[2,2]))
print( table(pred,testtarget) )

#presult = prcomp(train[,-c(2)],center=F)
#result1=naiveBayes(factor(agree)~.,data=data.frame(presult$x[,1:4],train["agree"]))
#pred=predict(result1,newdata=data.matrix(test) %*% data.matrix(presult$rotation[,1:4]))
#print(year)
#print(sum(testtarget==pred)/length(testtarget))
#print(table(pred,testtarget)[2,2]/(table(pred,testtarget)[2,1]+table(pred,testtarget)[2,2]))
#print( table(pred,testtarget) )

pred.real <- predict(result1, newdata=data.matrix(real))
print(table(pred.real))

}

#############################################################################
ggplot(data = data.frame(fastICA(train[,-c(2,3,10:12)], n.comp=2)$S, train['agree']), aes(x=X1, y=X2))+
  geom_point(aes(color=factor(agree)))

ggplot(data = train, aes(x=(EX15),y=(date),color=factor(agree)))+
  geom_point(size=3,alpha=.6)

ggplot(data = DF, aes(x=log(CAPITAL)))+
  geom_density()


####################################  ANN  ################################## 
real = read.csv("threedata2.csv")
real[,6] = as.numeric(substr(formatC(real[,6], width=7, flag="0"),1,3))
colnames(real)[6] = "date"
DF = read.csv("modeldata.csv")
DF = DF[,-c(1,2,12:19)]
DF = na.exclude(DF)
#'''transform date between 0 and 1.'''
DF[,9] = as.numeric(substr(formatC(DF[,9], width=7, flag="0"),1,3))
DF$agree = ifelse(DF$agree==T, 1, 0)
colnames(DF)[9] = "date"



set.seed(year)
sampling = sample(1:nrow(DF),nrow(DF)*0.7,replace=F)
train = DF[sampling,]
test = DF[-sampling,]
testtarget = test$agree


#'''preprocessing testset'''
for ( i in c(1,4:9))
{
  test[,i] = (test[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
test = test[,-c(2,3)]

#'''preprocessing realset'''
for ( i in c("CAPITAL","EX15","date","attendevent"))
{
  real[,i] = (real[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
real = real[,c("CAPITAL","EX15","date","attendevent")]

#'''preprocessing trainset'''
train = ubSMOTE(train, Y=factor(train$agree))$X
for ( i in c(1,4:9))
{
  train[,i] = (train[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
train = train[,-2]

formula.bpn <- agree ~ CAPITAL +EX15 +attendevent +date


		bpn <- neuralnet(formula = formula.bpn, 
                  data = train,
                  hidden = c(2,2,2,2),       # 一個隱藏層：2個node
                  learningrate = 0.001, # learning rate
                  threshold = 0.01,    # partial derivatives of the error function, a stopping criteria
                  stepmax = 5e8        # 最大的ieration數 = 500000(5*10^5)
                  )
		plot(bpn)
		pred <- compute(bpn, test[,c("CAPITAL","EX15","attendevent","date")])
		pred.result <- round(pred$net.result)
		sum(testtarget==pred.result)/length(testtarget)
		print( table(pred.result,testtarget) )
		pred.real <- compute(bpn, real[,c("CAPITAL","EX15","attendevent","date")])
		pred.real.result = round(pred.real$net.result)
		print( table(pred.real.result) )
		pred.train <- compute(bpn, train[,c("CAPITAL","EX15","attendevent","date")])
		pred.train.result <- round(pred.train$net.result)
		sum(train$agree==pred.train.result)/length(train$agree)
		print( table(pred.train.result,train$agree) )

real = read.csv("threedata2.csv")
write.csv(data.frame(pred.real.result, real),"DEEP0223_1007.csv",row.names=F)
############################     LASSO     ###############################

for(year in 2001:2017)
{
real = read.csv("threedata2.csv")
real[,6] = as.numeric(substr(formatC(real[,6], width=7, flag="0"),1,3))
colnames(real)[6] = "date"
DF = read.csv("modeldata.csv")
DF = DF[,-c(1,2,12:19)]
DF = na.exclude(DF)
#'''transform date between 0 and 1.'''
DF[,9] = as.numeric(substr(formatC(DF[,9], width=7, flag="0"),1,3))
DF$agree = ifelse(DF$agree==T, 1, 0)
colnames(DF)[9] = "date"



set.seed(year)
sampling = sample(1:nrow(DF),nrow(DF)*0.7,replace=F)
train = DF[sampling,]
test = DF[-sampling,]
testtarget = test$agree


#'''preprocessing testset'''
for ( i in c(1,4:9))
{
  test[,i] = (test[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
test = test[,-c(2,3)]

#'''preprocessing realset'''
for ( i in c("CAPITAL","EX15","date","attendevent"))
{
  real[,i] = (real[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
real = real[,c("CAPITAL","EX15","date","attendevent")]

#'''preprocessing trainset'''
train = ubSMOTE(train, Y=factor(train$agree))$X
for ( i in c(1,4:9))
{
  train[,i] = (train[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
train = train[,-2]


cv.glmmod <- cv.glmnet(x=data.matrix(train[,c(1,3,6,8)]), y=train$agree, alpha=1, family="binomial", nfolds=10)
model.final <- cv.glmmod$glmnet.fit
model.coef <- coef(model.final, s = cv.glmmod$lambda.1se)
all.coef <- coef(model.final, s =  min(model.final$lambda))
pred <- predict(model.final, data.matrix(test[,c(1,2,5,7)]), s=cv.glmmod$lambda.1se, type="class")
print(year)
print(sum(testtarget==pred)/length(testtarget))
print(table(pred,testtarget)[2,2]/(table(pred,testtarget)[2,1]+table(pred,testtarget)[2,2]))
print( table(pred,testtarget) )

pred.real <- predict(model.final, data.matrix(real), s=cv.glmmod$lambda.1se, type="response")
print(table(pred.real))
rm(list=ls(all=TRUE))
}


