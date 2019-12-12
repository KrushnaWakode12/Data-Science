library(ggplot2)
library(data.table)

df <- fread('Economist_Assignment_Data.csv',drop=1)

pl <- ggplot(df,aes(x=CPI,y=HDI)) + geom_point(aes(color=Region),shape='o',size=4) + geom_smooth(aes(group=1),method='lm',formula = y~log(x),se=F,color='red')

pointsToLabel <- c("Russia", "Venezuela", "Iraq", "Myanmar", "Sudan",
                   "Afghanistan", "Congo", "Greece", "Argentina", "Brazil",
                   "India", "Italy", "China", "South Africa", "Spane",
                   "Botswana", "Cape Verde", "Bhutan", "Rwanda", "France",
                   "United States", "Germany", "Britain", "Barbados", "Norway", "Japan",
                   "New Zealand", "Singapore")

pl2 <- pl + geom_text(aes(label = Country), color = "gray20", 
                       data = subset(df, Country %in% pointsToLabel),check_overlap = TRUE)

jpeg(pl3, filename = 'asout.jpeg',width = 720, height = 400,units = 'px')
pl3 <- pl2 + theme_bw() + scale_x_continuous(name = 'Corruption Perception Index,2011',limits = c(0,10),breaks = 1:10) + scale_y_continuous(name = 'Human Deveplopmwnt Index,2011',limits = c(0.2,1),breaks = 0.2:1) + ggtitle('Corruption and Human Development')

print(pl3)

dev.off()