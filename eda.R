library(ggplot2)
library(dplyr)
train <- read.csv2("train.csv",header = TRUE,sep =",")
colnames(train) <- c("id","month_last","donation_count","total_volume","month_first","result")
qplot(month_last,donation_count,data = train, geom = c("point","smooth"),
      method = "lm",color = factor(result))+facet_grid(result~.)
qplot(month_first,donation_count,data = train, geom = c("point","smooth"),
      method = "lm",color = factor(result),alpha = total_volume)+facet_grid(result~.)
ggsave("firstplot.pdf")
qplot(donation_count,total_volume,data = train, alpha= 0.2) + facet_grid(result~.)
# donation_count and total_volume are perfectly correlated

qplot(month_first,month_last,data = train) + facet_grid(result~.)

train2 <- train %>% mutate(age = month_first - month_last)
qplot(age,donation_count,data = train2,geom = c("point","smooth"),
      method ="lm",color = factor(result))
qplot(age,total_volume,data = train2,geom = c("point","smooth"),
      method ="lm")+ facet_grid(result~.)
ggsave("agevscount.pdf")
# The people who are interested in donation will have more frequent donations
train3 <- train2 %>% mutate(freq =  age/ donation_count)
qplot(factor(result),freq,data = train3,geom = "boxplot")
