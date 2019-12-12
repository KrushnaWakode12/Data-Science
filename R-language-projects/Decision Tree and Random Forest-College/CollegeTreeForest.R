#Call libraries that will be required for project
library(ISLR)
library(ggplot2)
library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)


#Assign Data to Data Frame
df <- College

#Plot Room.Board Versus Grad.Rate as Scatterplot with colored by Private Coulmn
plot(ggplot(df,aes(Room.Board,Grad.Rate)) + geom_point(aes(color=Private)))

#Plot Histogram of Full Time Undergrad Students with COlor by Private
plot(ggplot(df,aes(F.Undergrad)) + geom_histogram(aes(fill=Private),color='black',bins=50))

#Plot Histogram of Graduation Rate of Students with color by Private. Observe This One Carefully.
plot(ggplot(df,aes(Grad.Rate)) + geom_histogram(aes(fill=Private),color='black',bins=50))

#Find College for which Grad Rate is More Than 100%. Replace it with 100%.
print(subset(df,Grad.Rate >100))
df['Cazenovia College','Grad.Rate'] <- 100

#Set Seed Value to obtain same set of outcome
set.seed(101)

#Divide the dataset into tarining data and test data
samp <- sample.split(df$Private, SplitRatio = 0.70)
train <- subset(df, samp == T)
test <- subset(df, samp == F)

#Build Decision Tree Model and predict outcome for test data
tree <- rpart(Private ~., method = 'class', data=train)
tree.pred <- predict(tree,test)

#Check Head of prediction for reference/observation
print(head(tree.pred))

#Divide Predictions to Yes/No Factor and attach it as new coulmn
tree.pred <- as.data.frame(tree.pred)
joint <- function(x)
{
  if(x >= 0.5){
    return('Yes')
  }else{
   return('No') 
  }
}
tree.pred$Private <- sapply(tree.pred$Yes,joint)
print(head(tree.pred))

#Print Table to understand efficiency of Model
print(table(tree.pred$Private,test$Private))

#Plot Tree Model
print(prp(tree))

#Build Random Forest Model and Predict for test data
rf.model <- randomForest(Private ~ ., data =train, importance = T)
print(rf.model$confusion)
print(rf.model$importance)
p <- predict(rf.model,test)

#Print Table to Understand efficiency of Random Forest Model
print(table(p,test$Private))

write.table(table(p,test$Private), file = 'conf_table.csv', sep = ',')
