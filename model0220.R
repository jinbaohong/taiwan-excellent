setwd("logoscrape/model/model0218")


require(neuralnet) 
require(nnet) 
library(mxnet)
library(caret)
library(Matrix)
library(glmnet)
library(unbalanced)
library(e1071)
library(ggplot2)


DF = read.csv("modeldata.csv")
DF = DF[,-c(1,2,12:19)]
DF = na.exclude(DF)
#'''transform date between 0 and 1.'''
DF[,9] = as.numeric(substr(formatC(DF[,9], width=7, flag="0"),1,3))
DF$agree = ifelse(DF$agree==T, 1, 0)
colnames(DF)[9] = "date"

set.seed(2014)
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

#'''preprocessing trainset'''
train = ubUnder(train, Y=train$agree)$X
for ( i in c(1,4:9))
{
  train[,i] = (train[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
train = train[,-2]
#train = cbind(train, class.ind(train$agree))[,-2]
#colnames(train)[8:9] = c("notagree","agree")

#''' Tuning parameter '''



formula.bpn <- agree + notagree ~ CAPITAL +EX15 +attendevent +date
bpn <- neuralnet(formula = formula.bpn, 
                  data = train,
                  hidden = c(2,2),       # 留旅糷2node
                  learningrate = 0.001, # learning rate
                  threshold = 0.01,    # partial derivatives of the error function, a stopping criteria
                  stepmax = 5e8        # 程ieration计 = 500000(5*10^5)

                  )

plot(bpn)
pred <- compute(bpn, test[,c("CAPITAL","EX15","attendevent","date")])
pred.result <- round(pred$net.result)[,2]
sum(testtarget==pred.result)/length(testtarget)
print( table(pred.result,testtarget) )

#'''mxnet'''


expand.grid(c(1:i),c(0:j))
for
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=10)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=0)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=2)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

devices <- mx.cpu()

mx.set.seed(2009)
model <- mx.model.FeedForward.create(softmax, X=data.matrix(train[,-2]), y=train[,2],
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))
print("fuck")
preds = predict(model, data.matrix(test))
pred.label = max.col(t(preds))-1
sum(testtarget==pred.label)/length(testtarget)
table(pred.label, testtarget)
		mx.set.seed(2017)
		model <- mx.mlp(data.matrix(train[,-2]), train[,2], hidden_node=c(1,), out_node=2, out_activation="softmax",
	            num.round=20, array.batch.size=15, learning.rate=0.01, momentum=0.9, 
	            eval.metric=mx.metric.accuracy)

#'''Grid Search MLP'''
l1neural = c(3:20)
l2neural = c(1:20)
gs = c(0)
formula.bpn <- agree ~ CAPITAL +EX15 +attendevent +date

for (i in l1neural)
{
	for (j in l2neural)
	{
		bpn <- neuralnet(formula = formula.bpn, 
                  data = train,
                  hidden = c(i,j),       # 留旅糷2node
                  learningrate = 0.001, # learning rate
                  threshold = 0.01,    # partial derivatives of the error function, a stopping criteria
                  stepmax = 5e8        # 程ieration计 = 500000(5*10^5)
                  )
		pred <- compute(bpn, test[,c("CAPITAL","EX15","attendevent","date")])
		pred.result <- round(pred$net.result)
		gs[length(gs)+1] = sum(testtarget==pred.result)/length(testtarget)
	}
}
gridsearchmatrix = matrix(gs[-1], length(l2neural), length(l1neural))
rownames(gridsearchmatrix) = l2neural
colnames(gridsearchmatrix) = l1neural
print(paste("l2:",rownames(gridsearchmatrix)[which(gridsearchmatrix == max(gridsearchmatrix), arr.ind = T)[1]],
		", l1:",colnames(gridsearchmatrix)[which(gridsearchmatrix == max(gridsearchmatrix), arr.ind = T)[2]]))

print("fuck")















ggplot(data=train, aes(x=cos(CAPITAL*date), y=log(EX15*date),colour=factor(agree))) + 
	geom_point(size=3, alpha=.6)

ggplot(data=DF, aes(x=city,colour=agree)) + 
	geom_histogram()


outer = read.csv("threedata.csv")
real = outer[,c(6,7,18,21:24)]
real = na.exclude(real)
#'''transform date between 0 and 1.'''
real[,1] = as.numeric(substr(formatC(real[,1], width=7, flag="0"),1,3))
colnames(real)[1] = "date"
real = data.frame(real["CAPITAL"],real["attendevent"],
						real["EX14"],real["IM14"],
						real["EX15"],real["IM15"],real["date"])
#'''preprocessing testset'''
for ( i in c("CAPITAL","attendevent","EX14","IM14","EX15","IM15","date"))
{
  real[,i] = (real[,i] - min(na.exclude(train[,i])))/(max(na.exclude(train[,i])) - min(na.exclude(train[,i])))
}
preds = predict(model, data.matrix(real))
pred.label = max.col(t(preds))-1
table(pred.label)
outer = data.frame(pred.label, outer)

write.csv(outer,"BPresult.csv",row.names=F)










