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


ga_ctrl <- gafsControl(functions = rfGA,
                       method = "repeatedcv",
                       repeats = 1,
                       number = 5,
                       
                       verbose = TRUE)
set.seed(10)
train_ex <- as.factor(train_new_label)
rf_ga <- gafs(x = data.matrix(train_new_data_transformed), y = data.matrix(train_new_label),
              iters = 30,
              gafsControl = ga_ctrl)
rf_ga

library(data.table)
trainDT <- setDT(train_new_data_transformed)
trainDT2 <- select(trainDT,total_volume,freq,first_time,sqrt_recency)
trainlbl <- setDT(train_new_label)
testDT <- setDT(test_new_data_transformed)
testDT2 <- select(testDT,total_volume,freq,first_time,sqrt_recency)
testlbl <- setDT(test_new_label)


library(xgboost)
dtrain <- xgb.DMatrix(data = data.matrix(trainDT2),label = data.matrix(trainlbl))
dtest <- xgb.DMatrix(data = data.matrix(testDT2),label=data.matrix(testlbl))
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eta=0.1,
  gamma=5,
  max_depth=15,
  min_child_weight=1,
  subsample=1,
  colsample_bytree=1
)

xgbcv <- xgb.cv(params = params
                ,data = dtrain
                ,nrounds = 5000
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
  ,nrounds = 103
  ,watchlist = list(val=dtest,train=dtrain)
  ,print_every_n = 10
  ,early_stopping_rounds = 10
  ,maximize = F
  ,eval_metric = "logloss"
)
test$xgbpred <- predict(xgb1,dtest)
write.csv(test,"final_pred2.csv")
library(e1071)
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3, 
                        #summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        allowParallel=T)

xgb.grid <- expand.grid(nrounds = 1000,
                        eta = c(0.01,0.05,0.1),
                        max_depth = c(2,4,6,8,10,14),
                        gamma = c(1,2,3,4,5),
                        min_child_weight=1,
                        subsample=1,
                        colsample_bytree=1
)
set.seed(45)
train_final <- cbind(train_new_data_transformed,train_new_label)
library(magrittr)
train_final$result %<>% factor
levels(train_final$result) <- c("Low","High")
xgb_tune <-train(result~total_volume+freq+first_time+sqrt_recency,
                 data=train_final,
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid,
                 verbose=T,
                 metric="AUC",
                 nthread =3
)
xgbpred <- predict(xgb_tune,test_new_data_transformed, type = "prob")
test$high <- xgbpred$High
write.csv(test,"final_pred2.csv")

LogLosSummary <- function (data, lev = NULL, model = NULL) {
  LogLos <- function(actual, pred, eps = 1e-15) {
    stopifnot(all(dim(actual) == dim(pred)))
    pred[pred < eps] <- eps
    pred[pred > 1 - eps] <- 1 - eps
    -sum(actual * log(pred)) / nrow(pred) 
  }
  if (is.character(data$obs)) data$obs <- factor(data$obs, levels = lev)
  pred <- data[, "pred"]
  obs <- data[, "obs"]
  isNA <- is.na(pred)
  pred <- pred[!isNA]
  obs <- obs[!isNA]
  data <- data[!isNA, ]
  cls <- levels(obs)
  
  if (length(obs) + length(pred) == 0) {
    out <- rep(NA, 2)
  } else {
    pred <- factor(pred, levels = levels(obs))
    require("e1071")
    out <- unlist(e1071::classAgreement(table(obs, pred)))[c("diag",                                                                                                                                                             "kappa")]
    
    probs <- data[, cls]
    actual <- model.matrix(~ obs - 1)
    out2 <- LogLos(actual = actual, pred = probs)
  }
  out <- c(out, out2)
  names(out) <- c("Accuracy", "Kappa", "LogLoss")
  
  if (any(is.nan(out))) out[is.nan(out)] <- NA 
  
  out
}


fitControl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 10, 
                           ## Estimate class probabilities 		   	  
                           classProbs = TRUE, 
                           ## Evaluate performance using  
                           ## the following function 	
                           summaryFunction = LogLosSummary) 

set.seed(825) 

gbmGrid <- expand.grid(interaction.depth = c(1, 5, 9), 	
                       n.trees = (1:30)*50, shrinkage = 0.1, 
                       n.minobsinnode = 20)

gbmFit3 <- train(result~total_volume+freq+first_time+sqrt_recency, data = train_final, method = "gbm", 
                 trControl = fitControl, verbose = FALSE,
                 tuneGrid = gbmGrid, 
                 ## Specify which metric to optimize 
                 metric = "LogLos")
gbmFit3


ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE,summaryFunction = LogLosSummary,classProbs = TRUE)

mod_fit <- train(result~total_volume+freq+first_time+sqrt_recency,  data=train_final, method="glm", family="binomial",
                 trControl = ctrl, tuneLength = 5)
mod_fit
pred = predict(mod_fit, newdata=test_new_data_transformed, type = "prob")
test$lgr <- pred$High
write.csv(test,"xgboost/final_pred2.csv")
