#Building Linear Regression without test data on Kaggles' Bike Share Project

#Load add-on packages
library(ggplot2)
library(dplyr)

#read data file in wd
bike <- read.csv('bikeshare.csv')

#plotting count vs temp as scatterplot to understand relationship between them
a1 <- ggplot(data = bike,aes(temp,count)) + geom_point(alpha = 0.2,aes(color=temp)) + theme_bw()
plot(a1)

#plotting count vs datetime as scatterplot to understand relationship them
bike$datetime <- as.POSIXct(bike$datetime)
a2 <- ggplot(bike,aes(datetime,count)) + geom_point(aes(color = temp),alpha = 0.5) + scale_color_continuous(low='#55D8CE',high='#FF6E2E') + theme_bw()
plot(a2)

#obtain correlation between temp and count
x <- cor(bike[,c('temp','count')])

#boxplot count vs seasons 
a3 <- ggplot(bike,aes(factor(season),count)) + geom_boxplot(aes(color=factor(season))) + theme_bw()
plot(a3)

#add another column of hour to df of bike from dattime column
bike$hour <- sapply(bike$datetime,function(x){format(x,'%H')})
bike$hour <- sapply(bike$hour,as.numeric)

#scatterplot counts vs hour only with workingday equal to 1 i.e, on working day
x <- ggplot(filter(bike,workingday==1),aes(hour,count)) + geom_point(position = position_jitter(w=1,h=0),aes(color=temp),alpha=0.5)
a4 <- x + scale_color_gradientn(colors = c('dark blue','blue','light blue','light green','yellow','orange','red')) + theme_bw()
plot(a4)

#scatterplot counts vs hour only with workingday equal to 0 i.e, on non-working day
x <- ggplot(filter(bike,workingday==0),aes(hour,count)) + geom_point(position = position_jitter(w=1,h=0),aes(color=temp),alpha=0.5)
a5 <- x + scale_color_gradientn(colors = c('dark blue','blue','light blue','light green','yellow','orange','red')) + theme_bw()
plot(a5)

#linear regression model based on tempreture for count
temp.model <- lm(count ~ temp,bike)

#linear regression model based on all factors except casual,registered,datetime,atemp to get count predictions
model <- lm(count ~ . -casual -registered -datetime -atemp,bike)

print(summary(model))
