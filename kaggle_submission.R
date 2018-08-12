#import libraries
library(kknn)
library(randomForest)
library(dplyr)
library(vcdExtra)
library(ISLR)
library(rpart)
library(party)
library(partykit)
library(rattle)
library(caret)
library(ggplot2)

#trainControl definition
TRCONTROL = trainControl(
  method = "cv",
  number = 1,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE
)

#read in data
trained_data <- read.csv("data/application_train.csv", stringsAsFactors = FALSE, na.strings = c("NA", ""))
tested_data <- read.csv("data/application_test.csv", stringsAsFactors = FALSE, na.strings = c("NA", ""))

# test ---
#set target
target <- trained_data$TARGET
trained_data$TARGET <- factor(target, labels = c("no", "yes"))
trained_data$income <- trained_data$AMT_INCOME_TOTAL
trained_data$annuity <- trained_data$AMT_ANNUITY
trained_data$credit <- trained_data$AMT_CREDIT
trained_data$goods <- trained_data$AMT_GOODS_PRICE
# ----

#fill in NA cells
trained_data$AMT_GOODS_PRICE[is.na(trained_data$AMT_GOODS_PRICE)] <- 
  with(trained_data, ave(AMT_GOODS_PRICE, FUN = function(x) median(x, na.rm = TRUE)))[is.na(trained_data$AMT_GOODS_PRICE)]
trained_data$AMT_ANNUITY[is.na(trained_data$AMT_ANNUITY)] <- 
  with(trained_data, ave(AMT_ANNUITY, FUN = function(x) median(x, na.rm = TRUE)))[is.na(trained_data$AMT_ANNUITY)]

#fill in missing values to test new variables
trained_data$CNT_CHILDREN[is.na(trained_data$CNT_CHILDREN)] <- 
  with(trained_data, ave(CNT_CHILDREN, FUN = function(x) median(x, na.rm = TRUE)))[is.na(trained_data$CNT_CHILDREN)]
trained_data$FLAG_OWN_CAR[is.na(trained_data$FLAG_OWN_CAR)] <- 
  with(trained_data, ave(FLAG_OWN_CAR, FUN = function(x) median(x, na.rm = TRUE)))[is.na(trained_data$FLAG_OWN_CAR)]
trained_data$FLAG_OWN_REALTY[is.na(trained_data$FLAG_OWN_REALTY)] <- 
  with(trained_data, ave(FLAG_OWN_REALTY, FUN = function(x) median(x, na.rm = TRUE)))[is.na(trained_data$FLAG_OWN_REALTY)]
trained_data$NAME_EDUCATION_TYPE[is.na(trained_data$NAME_EDUCATION_TYPE)] <- 
  with(trained_data, ave(NAME_EDUCATION_TYPE, FUN = function(x) median(x, na.rm = TRUE)))[is.na(trained_data$NAME_EDUCATION_TYPE)]

#remove low variance variables
nz <- nearZeroVar(trained_data, freqCut = 2000, uniqueCut = 10)
trained_data <- trained_data[,-nz]

# convert certain varibales to factors
trained_data$NAME_FAMILY_STATUS = as.factor(trained_data$NAME_FAMILY_STATUS)
trained_data$FLAG_OWN_CAR = as.factor(trained_data$FLAG_OWN_CAR)
trained_data$NAME_EDUCATION_TYPE = as.factor(trained_data$NAME_EDUCATION_TYPE)

#importance of all the variables
importance <- varImp(fit_glm, scale = FALSE)
importance

# making a glm for testing variables 
fit_glm <- glm(TARGET ~ NAME_FAMILY_STATUS + AMT_CREDIT + AMT_GOODS_PRICE + CNT_CHILDREN + FLAG_OWN_CAR 
               + NAME_EDUCATION_TYPE + AMT_ANNUITY, trained_data, family = binomial)

print(fit_glm)
summary(fit_glm)

#resampling methods to test model
fit_boot = train(TARGET ~ NAME_FAMILY_STATUS + AMT_CREDIT + AMT_GOODS_PRICE + CNT_CHILDREN + FLAG_OWN_CAR 
                 + NAME_EDUCATION_TYPE + AMT_ANNUITY, data = trained_data, method = "nnet", metric = "Sens",
                 trControl = trainControl(classProbs = TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE, method = "boot"))

fit_cv = train(TARGET ~ AMT_INCOME_TOTAL + CNT_CHILDREN + FLAG_OWN_REALTY, data = trained_data, method = "nnet", metric = "Sens", 
               trControl = trainControl(classProbs = TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE, method = "cv", number = 5))

# rewrite the file
tested_data$AMT_GOODS_PRICE[is.na(tested_data$AMT_GOODS_PRICE)] <- 
  with(tested_data, ave(AMT_GOODS_PRICE, FUN = function(x) median(x, na.rm = TRUE)))[is.na(tested_data$AMT_GOODS_PRICE)]
tested_data$AMT_ANNUITY[is.na(tested_data$AMT_ANNUITY)] <- 
  with(tested_data, ave(AMT_ANNUITY, FUN = function(x) median(x, na.rm = TRUE)))[is.na(tested_data$AMT_ANNUITY)]

#filling in missing values for variables in testing set
tested_data$CNT_CHILDREN[is.na(tested_data$CNT_CHILDREN)] <- 
  with(tested_data, ave(CNT_CHILDREN, FUN = function(x) median(x, na.rm = TRUE)))[is.na(tested_data$CNT_CHILDREN)]
tested_data$FLAG_OWN_CAR[is.na(tested_data$FLAG_OWN_CAR)] <- 
  with(tested_data, ave(FLAG_OWN_CAR, FUN = function(x) median(x, na.rm = TRUE)))[is.na(tested_data$FLAG_OWN_CAR)]
tested_data$FLAG_OWN_REALTY[is.na(tested_data$FLAG_OWN_REALTY)] <- 
  with(tested_data, ave(FLAG_OWN_REALTY, FUN = function(x) median(x, na.rm = TRUE)))[is.na(tested_data$FLAG_OWN_REALTY)]
tested_data$NAME_EDUCATION_TYPE[is.na(tested_data$NAME_EDUCATION_TYPE)] <- 
  with(tested_data, ave(NAME_EDUCATION_TYPE, FUN = function(x) median(x, na.rm = TRUE)))[is.na(tested_data$NAME_EDUCATION_TYPE)]

#remove low variance variables
nz <- nearZeroVar(tested_data, freqCut = 2000, uniqueCut = 10)
tested_data <- tested_data[,-nz]

#convert certain varibales to factors
tested_data$NAME_FAMILY_STATUS = as.factor(tested_data$NAME_FAMILY_STATUS)
tested_data$FLAG_OWN_CAR = as.factor(tested_data$FLAG_OWN_CAR)
tested_data$NAME_EDUCATION_TYPE = as.factor(tested_data$NAME_EDUCATION_TYPE)

numbers <- data.frame(tested_data$SK_ID_CURR)
prediction <- predict(fit_glm, tested_data, type = "response")
submission <- data.frame(numbers, prediction)

colnames(submission)[1] = ("SK_ID_CURR")
colnames(submission)[2] = ("TARGET")

submit <- data.frame(submission[1], submission[2])

write.csv(submit, file = "prediction.csv", row.names = FALSE, quote = FALSE)



