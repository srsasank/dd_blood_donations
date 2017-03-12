library(ggplot2)
library(dplyr)
library(ROSE)
train <- read.csv2("train.csv",header = TRUE,sep =",")
test <- read.csv2("test.csv",header = TRUE,sep =",")
colnames(train) <- c("id","month_last","donation_count","total_volume","month_first","result")
colnames(test) <- c("id","month_last","donation_count","total_volume","month_first")
test <-  test %>% mutate(result = 0.5)

checker <- function(x,y){
  ifelse(x == y,1,0)
}


train <- train %>% mutate(age = month_first - month_last,
                          freq = age / donation_count,
                          donation_score = age * donation_count,
                          first_time = checker(month_first,month_last))
test <- test %>%  mutate(age = month_first - month_last,
                         freq = age / donation_count,
                         donation_score = age * donation_count,
                         first_time = checker(month_first,month_last))

train_new <- select(train, -id)
test_new <- select(test, -id)
write.csv(train_new, "train_new.csv",row.names = FALSE)
write.csv(test_new, "test_new.csv",row.names = FALSE)
trainrose <- ROSE(result ~ ., data = train_new, seed = 1)$data
write.csv(trainrose,"train_rose.csv",row.names = FALSE)
library(data.table)
trainDT <- fread("train_new.csv")
trainDT2 <- select(trainDT, - result)
trainlbl <- select(trainDT, result)
testDT <- fread("test_new.csv")
testDT2 <- select(testDT, - result)
testlbl <- select(testDT, result)


library(xgboost)
dtrain <- xgb.DMatrix(data = data.matrix(trainDT2),label = data.matrix(trainlbl))
dtest <- xgb.DMatrix(data = data.matrix(testDT2),label=data.matrix(testlbl))
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eta=0.3,
  gamma=5,
  max_depth=6,
  min_child_weight=1,
  subsample=1,
  colsample_bytree=1
)

xgbcv <- xgb.cv(params = params
                ,data = dtrain
                ,nrounds = 200
                ,nfold = 10
                ,showsd = T
                ,stratified = T
                ,print_every_n = 10
                ,early_stopping_rounds = 20
                ,maximize = F,
                eval_metric = "logloss"
)
min(xgbcv$test.error.mean)
xgb1 <- xgb.train(
  params = params
  ,data = dtrain
  ,nrounds = 48
  ,watchlist = list(val=dtest,train=dtrain)
  ,print_every_n = 10
  ,early_stopping_rounds = 10
  ,maximize = F
  ,eval_metric = "logloss"
)
test$xgbpred <- predict(xgb1,dtest)
write.csv(test,"final_pred2.csv")

trainroseDT <- fread("train_rose.csv")
trainroseDT2 <- select(trainroseDT, - result)
trainroselbl <- select(trainroseDT, result)
dtrainrose <- xgb.DMatrix(data = data.matrix(trainroseDT2),label = data.matrix(trainroselbl))

params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eta=0.1,
  gamma=5,
  max_depth=6,
  min_child_weight=1,
  subsample=1,
  colsample_bytree=1
)

xgbcv <- xgb.cv(params = params
                ,data = dtrainrose
                ,nrounds = 200
                ,nfold = 10
                ,showsd = T
                ,stratified = T
                ,print_every_n = 10
                ,early_stopping_rounds = 20
                ,maximize = F,
                eval_metric = "logloss"
)

xgb1 <- xgb.train(
  params = params
  ,data = dtrainrose
  ,nrounds = 25
  ,watchlist = list(val=dtest,train=dtrain)
  ,print_every_n = 10
  ,early_stopping_rounds = 10
  ,maximize = F
  ,eval_metric = "logloss"
)

test$xgbpredrose <- predict(xgb1,dtest)
write.csv(test,"final_predrose.csv")
