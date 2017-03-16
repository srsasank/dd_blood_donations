library(ggplot2)
library(dplyr)
library(caret)
train <- read.csv2("../train.csv",header = TRUE,sep =",")
test <- read.csv2("../test.csv",header = TRUE,sep =",")
colnames(train) <- c("id","month_last","donation_count","total_volume","month_first","result")
colnames(test) <- c("id","month_last","donation_count","total_volume","month_first")
test <-  test %>% mutate(result = 0.5)

checker <- function(x,y){
  ifelse(x == y,1,0)
}

train <- train %>% mutate(age = month_first - month_last,
                          freq = age / donation_count,
                          log_freq = log(freq+1),
                          sqrt_freq = sqrt(freq),
                          donation_score = age * donation_count,
                          log_score = log(donation_score+1),
                          first_time = checker(month_first,month_last),
                          log_recency = log(month_last +1),
                          sqrt_recency = sqrt(month_last)
)
test <- test %>%  mutate(age = month_first - month_last,
                         freq = age / donation_count,
                         log_freq = log(freq+1),
                         sqrt_freq = sqrt(freq),
                         donation_score = age * donation_count,
                         log_score = log(donation_score+1),
                         first_time = checker(month_first,month_last),
                         log_recency = log(month_last +1),
                         sqrt_recency = sqrt(month_last))

train_new <- select(train, -id)
test_new <- select(test, -id)
train_new_data <- select(train_new, - result)
train_new_label <- select(train_new, result)
test_new_data <- select(test_new, - result)
test_new_label <- select(test_new, result)

train_new_data_pp <- preProcess(train_new_data, 
                                method = c("center", "scale", "YeoJohnson", "nzv"))
train_new_data_pp
train_new_data_transformed <- predict(train_new_data_pp, newdata = train_new_data)
test_new_data_pp <- preProcess(test_new_data, 
                               method = c("center", "scale", "YeoJohnson", "nzv"))

test_new_data_transformed <- predict(test_new_data_pp, newdata = test_new_data)

library(data.table)
trainDT <- setDT(train_new_data_transformed)
trainDT2 <- select(trainDT,total_volume,freq,first_time,sqrt_recency)
trainlbl <- setDT(train_new_label)
testDT <- setDT(test_new_data_transformed)
testDT2 <- select(testDT,total_volume,freq,first_time,sqrt_recency)
testlbl <- setDT(test_new_label)

LogLoss<-function(act, pred)
{
  eps = 1e-15;
  nr = length(pred)
  pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr) 
  pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
  ll = sum(act*log(pred) + (1-act)*log(1-pred))
  ll = ll * -1/(length(act)) 
  return(ll);
}



plot(c(0,1),c(0,1), col="grey",type="l",xlab = "Mean Prediction",ylab="Observed Fraction")
reliability.plot <- function(obs, pred, bins=10, scale=T) {
  # Plots a reliability chart and histogram of a set of predicitons from a classifier
  #
  # Args:
  # obs: Vector of true labels. Should be binary (0 or 1)
  # pred: Vector of predictions of each observation from the classifier. Should be real
  # number
  # bins: The number of bins to use in the reliability plot
  # scale: Scale the pred to be between 0 and 1 before creating reliability plot
  require(plyr)
  library(Hmisc)
  min.pred <- min(pred)
  max.pred <- max(pred)
  min.max.diff <- max.pred - min.pred
  if (scale) {
    pred <- (pred - min.pred) / min.max.diff 
  }
  bin.pred <- cut(pred, bins)
  k <- ldply(levels(bin.pred), function(x) {
    idx <- x == bin.pred
    c(sum(obs[idx]) / length(obs[idx]), mean(pred[idx]))
  })
  is.nan.idx <- !is.nan(k$V2)
  k <- k[is.nan.idx,]
  return(k)
}

# reading the train dataset
train_final <-  cbind(trainDT2,trainlbl)
train_final$result <- as.factor(train_final$result)
test_final <- testDT2
set.seed(221)
sub <- sample(nrow(train_final), floor(nrow(train_final) * 0.85))
training<-train_final[sub,]
cv<-train_final[-sub,]

# training a random forest model without any feature engineering or pre-processing
library(randomForest)
model_rf<-randomForest(result~.,data = training,keep.forest=TRUE,importance=TRUE)

#predicting on the cross validation dataset
result_cv<-as.data.frame(predict(model_rf,cv,type="prob"))   

#calculating Log Loss without Platt Scaling
LogLoss(as.numeric(as.character(cv$result)),result_cv$`1`)   

# performing platt scaling on the dataset
dataframe<-data.frame(result_cv$`1`,cv$result)
colnames(dataframe)<-c("x","y")

# training a logistic regression model on the cross validation dataset
model_log<-glm(y~x,data = dataframe,family = binomial)

#predicting on the cross validation after platt scaling
dataframe[-2]
result_cv_platt<-predict(model_log,dataframe[-2],type = "response")
LogLoss(as.numeric(as.character(cv$result)),result_cv_platt)




