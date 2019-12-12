#Load libraries required
library(caTools)
library(ggplot2)
library(dplyr)

#Set seed value to get same set of outcomes
set.seed(101)

#read data file
adult <- read.csv('adult_sal.csv')

adult <- select(adult,-X)

#get table of adult data-set based on type of employee
print(table(adult$type_employer))

#DATA CLEANING
#Merge job-factors to get more clean data
unemp <- function(job){
  job <- as.character(job)
  if (job =='Without-pay' | job == 'Never-worked'){
    return('Unemployed')
  }else{
    return(job)
  }
}
adult$type_employer <- sapply(adult$type_employer,unemp)

#Combine More factors to get cleaner data
group_emp <- function(job){
  job <- as.character(job)
  if (job=='Local-gov' | job == 'State-gov'){
    return('SL-job')
  }else if (job == 'Self-emp-inc' | job == 'Self-emp-not-inc'){
    return('Self-employed')
  }else{
    return(job)
  }
}
adult$type_employer <- sapply(adult$type_employer,group_emp)
print(table(adult$type_employer))

#Clean data based on marital factor
print( table(adult$marital))
group_marital <- function(mar){
  mar <- as.character(mar)
  if(mar == 'Never-married' ){
    return(mar)
  }else if(mar == 'Divorced' | mar == 'Separated' | mar == 'Widowed'){
      return('Not-married')
  }else{
      return('Married')
    }
}

adult$marital <- sapply(adult$marital,group_marital)
print(table(adult$marital))

#Clean data based on type of country
print(table(adult$country))
Asia <- c('China','Hong','India','Iran','Cambodia','Japan', 'Laos' ,
          'Philippines' ,'Vietnam' ,'Taiwan', 'Thailand')

North.America <- c('Canada','United-States','Puerto-Rico' )

Europe <- c('England' ,'France', 'Germany' ,'Greece','Holand-Netherlands','Hungary',
            'Ireland','Italy','Poland','Portugal','Scotland','Yugoslavia')

Latin.and.South.America <- c('Columbia','Cuba','Dominican-Republic','Ecuador',
                             'El-Salvador','Guatemala','Haiti','Honduras',
                             'Mexico','Nicaragua','Outlying-US(Guam-USVI-etc)','Peru',
                             'Jamaica','Trinadad&Tobago')
Other <- c('South')

group_country <- function(coun){
  if(coun %in% Asia){
    return('Asia')
  }else if(coun %in% North.America){
    return('North.America')
  }else if(coun %in% Europe){
    return('Europe')
  }else if(coun %in% Latin.and.South.America){
    return('Latin.and.South.America')
  }else{
    return('Other')
  }
}

adult$country <- sapply(adult$country,group_country)
print(table(adult$country))

#Convert character type to Factor Levels for better analysis
adult$type_employer <- sapply(adult$type_employer,factor)
adult$country <- sapply(adult$country,factor)
adult$marital <- sapply(adult$marital,factor)

#Substitute '?' values with NA
adult[adult == '?'] <- NA

#Omit NA values from adult dataset
adult <- na.omit(adult)

#Plot Histogram based on Age vs Count with filled color base on income
print(ggplot(adult,aes(age)) + geom_histogram(aes(fill=income),color='black',binwidth = 1))

#Plot Histogram based on Hr per week vs count
print(ggplot(adult,aes(hr_per_week)) + geom_histogram() + theme_bw())

#Plot Bar plot based on country in which employee belongs
print(ggplot(adult,aes(country)) + geom_bar(aes(fill=income),color='black') + theme_bw())

#Split the data into training and test data
samp <- sample.split(adult$income,SplitRatio = 0.70)
train <- subset(adult,samp == T)
test <- subset(adult,samp == F)

#Build Logistic Regression Model
model <- glm(income ~ ., family = binomial(logit), data = train)
print(summary(model))

#Make Step Model 
step.model <- step(model)
print(summary(step.model))

#Predict income based on test dataset
test$predicted_income <- predict(model, newdata = test, type = 'response')

#Table Based on predictions
a <- table(test$income, test$predicted_income > 0.5)
print(a)

print('Accuracy of Model is:')
print((a[1] + a[4])/(a[1] + a[2] + a[3] +a[4]))

print('precision of Model is:')
print(a[1] / (a[1] + a[2]))

write.table(a, file = 'conf_table.csv', sep = ',')
