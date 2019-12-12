library(ggplot2)
library(data.table)
library(dplyr)

#Reading CSV files as Data
sal <- read.csv('Salaries.csv')
batting <- read.csv('batting.csv')

#Base Avg
batting$BA <- batting$H / batting$AB
#On Base Percentage
batting$OBP <- (batting$H + batting$BB + batting$HBP)/(batting$AB + batting$BB + batting$HBP + batting$SF)
#Building X1B (Singles)
batting$X1B <- batting$H - batting$X2B - batting$X3B- batting$HR
#Creating Slugging Avg
batting$SLG <- ((batting$X1B)+(2*batting$X2B)+(3*batting$X3B)+(4*batting$HR))/(batting$AB)


#before merging salaries and stats, year coulmn should coincide
batting <- subset(batting, yearID >= 1985)

#combining/merging salary and stat data
combo <- merge(batting,sal, by = c('playerID','yearID'))

#data frame of lost players
lp <- c('damonjo01','giambja01','saenzol01')
lost_players <- subset(combo, playerID %in% lp)
lost_players <- subset(lost_players, yearID ==2001)
lost_players <- lost_players[,c('playerID','H','X2B','X3B','HR','OBP','SLG','BA','AB')]

#now to find replacement to this players and filter things out
avail_players <- subset(combo, yearID ==2001)
avail_players <- filter(avail_players,salary<8000000,OBP>0,AB>=500)
possible <- head(arrange(avail_players,desc(OBP)),10)
possible <- possible[,c('playerID','OBP','AB','salary')]
grphclcmpr <- ggplot(possible,aes(x=OBP,y=salary)) + geom_point()
print(grphclcmpr)

#showing results
print("The lost players are:")
print(select(lost_players,'playerID'))
sprintf('\n')
print('And through program, three replacement for the players are(along with their data):')
print(possible[2:4,])

write.csv(possible[2:4,],file = 'outcome.csv',sep = ',')
