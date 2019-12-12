#define libraries that are required
library(ggplot2)
library(caTools)
library(e1071)

#Read excel file
loans <- read.csv('loan_data.csv')

#Check data summary and Structure
#print(summary(loans))
print(str(loans))

#Convert level based values into factor based
loans$credit.policy <- factor(loans$credit.policy)
loans$inq.last.6mths <- factor(loans$inq.last.6mths)
loans$delinq.2yrs <- factor(loans$delinq.2yrs)
loans$pub.rec <- factor(loans$pub.rec)
loans$not.fully.paid <- factor(loans$not.fully.paid)

#plot Histogram for fico with not.fully.paid as fill aesthetic
print(ggplot(loans,aes(x=fico)) + geom_histogram(aes(fill=not.fully.paid),color='black',bin=40,alpha = 0.5) + scale_fill_manual(values = c('green','red')) + theme_bw())

#Plot barplot for factors of purpose with not.fully.paid as fill aesthetic and position dodge 
print(ggplot(loans, aes(x=factor(purpose))) + geom_bar(aes(fill=not.fully.paid), position = 'dodge') + theme_bw())

#Plot scatterplot of fico vs int.rate with not.fully.paid as fill aesthetic
print(ggplot(loans,aes(int.rate, fico)) + geom_point(aes(color=not.fully.paid), alpha=0.5) + theme_grey())

#set seed to fix value to get same set of outcome
set.seed(101)

#Splitting data into train and test data
spl <- sample.split(loans$not.fully.paid, SplitRatio = 0.7)
train <- subset(loans,spl == T)
test <- subset(loans,spl== F)

#Build the SVM model and predict on test
model <- svm(not.fully.paid ~ ., data = train)
print(summary(model))
pred.values <- predict(model,test[1:13])
print(table(pred.values,test$not.fully.paid))

#tune SVM model on train data with different ranges of cost and gamma (May take few minutes)
tune.result <- tune(svm, train.x = not.fully.paid ~ ., data = train, kernel = 'radial',ranges = list(cost = c(1,10), gamma = c(0.1,1)))
print(tune.result)

#Build SVM model for obtained values of cost and gamma
model <- svm(not.fully.paid ~ ., data = train, cost = 1, gamma = 0.1)
pred.values <- predict(model, test[1:13])

#draw table of predictions and test data
print(table(pred.values, test$not.fully.paid))

write.csv(table(pred.values, test$not.fully.paid), file = 'conf_mat.csv',sep=',')