#defining libraries needed for program
library(caTools)
library(ISLR)
library(class)
library(ggplot2)

#set certain fix value to get same set of results
set.seed(101)

#Check Data
print(head(iris))
print(var(iris[,1]))
print(var(iris[,2]))

#Standardize the data to get same variance throughout data-set
std.data <- scale(iris[1:4])
final.data <- cbind(std.data,iris[5])
print(head(final.data))

#Split data into train and test data
samp <- sample.split(final.data$Species, SplitRatio = 0.70)
train <- subset(final.data, samp == T)
test <- subset(final.data, samp == F)

#make KNN model and predictions based on test data
predicted.species <- knn(train[1:4],test[1:4],train$Species,k=1)
print(predicted.species)

#Check Mis-classification rate for model
print(mean(predicted.species != test$Species))

#Check Mis-classfication rate for different values of K
predicted.species <- NULL
error.rate <- NULL

for (i in 1:10) {
  set.seed(101)
  predicted.species <- knn(train[1:4],test[1:4],train$Species,k=i)
  error.rate[i] <- mean(predicted.species != test$Species)
}

print(error.rate)

#Plot K values versus Error rate to get better idea
k.values <- 1:10
error.df <- data.frame(k.values,error.rate)
print(error.df)
print(ggplot(error.df,aes(x=k.values,y=error.rate)) + geom_point() + geom_line(lty='dotted',color='red'))
