#Define libraries required for Program
library(caTools)
library(neuralnet)
library(ggplot2)

set.seed(101)

#Read input file into data frame
df <- read.csv('bank_note_data.csv')

#Split data into training and test data
spl <- sample.split(df$Class, SplitRatio = 0.70)
train <- subset(df, spl == T)
test <- subset(df, spl == F)

#Build Neural Network with 10 hidden layers and find predictions for test data
nn <- neuralnet(Class ~ Image.Var + Image.Skew + Image.Curt + Entropy, data = train, hidden = 10, linear.output = F)
pred.values <- compute(nn, test[,1:4])

#ROund off the predicted values to get level of classes
preds <- sapply(pred.values$net.result, round)

#Draw table to compare predicted and real data
print(table(preds,test$Class))

write.table(x = table(preds,test$Class), file = 'conf_mat.csv', sep=',')