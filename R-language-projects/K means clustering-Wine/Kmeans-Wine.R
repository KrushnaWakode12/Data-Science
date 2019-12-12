#define libraries required for program
library(ggplot2)
library(cluster)

#Read input files
df1 <- read.csv('winequality-red.csv', sep = ';')
df2 <- read.csv('winequality-white.csv', sep = ';')

#add wine labels
df1$label <- sapply(df1$pH, function(x){'red'})
df2$label <- sapply(df2$pH, function(x){'white'})

#COmbine both datasets
wine <- rbind(df1,df2)

#Plot Histograms and Scatterplot between different parameters to understand co-relation between them
print(ggplot(wine,aes(x=residual.sugar)) + geom_histogram(aes(fill=label),color='black',bins = 50) + scale_fill_manual(values = c('#ae4554','white')) + theme_gray())
print(ggplot(wine,aes(x=citric.acid)) + geom_histogram(aes(fill=label),color='black',bins = 50) + scale_fill_manual(values = c('#ae4554','white')) + theme_gray())
print(ggplot(wine,aes(x=alcohol)) + geom_histogram(aes(fill=label),color='black',bins = 50) + scale_fill_manual(values = c('#ae4554','white')) + theme_gray())
print(ggplot(wine,aes(x=citric.acid,y=residual.sugar)) + geom_point(aes(color=label), alpha = 0.3) + scale_color_manual(values = c('red','white')) + theme_dark())

#Build K-means model and observe outcome 
wine.cluster <- kmeans(wine[1:12],2)
print(wine.cluster)

#Compare Predicted clusters with given data outcome to find effeciency of model 
print(table(wine$label,wine.cluster$cluster))

write.table(table(wine$label,wine.cluster$cluster), file='conf_table.csv', sep=',',col.names = c('white','red'))
