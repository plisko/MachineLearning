library(caret)
setwd("./data/")
# dataset for the homework (nearly 20000 observations)
homeworkSet <- read.csv("pml-training.csv")
# 20 observations we are tested upon
final20Set <- read.csv("pml-testing.csv")

# first of all we "take a look" at the data
summary(homeworkSet)
str(homeworkSet)

# from the inspection of the set, we could see that some variables are not significant for our analysis
# i.e. skewness_yaw_belt is most of the times NA or #DIV/0!
# many variables are non-numeric

## Feature selection
# we remove the factor variables (also "classe")
colFactors <- (sapply(homeworkSet, class) == "factor")
colNAs <- colSums(is.na(homeworkSet))/dim(homeworkSet)[1]
selectedColumns <- colnames(homeworkSet)[(colFactors == FALSE) & (colNAs < 0.9)]
selectedColumns
# the first 4 columns are incemental and timestamp data, they can be removed
selectedColumns <- selectedColumns[-c(1,2,3,4)]

# Principal Component Analysis
homeworkSetReduced <- homeworkSet[c("classe", selectedColumns)]
final20SetReduced <- final20Set[selectedColumns]
# We choose a simple 70% 30% decomposition in training/test sets
inTrain <- createDataPartition(y = homeworkSetReduced$classe, p = 0.7, list = FALSE)
trainingSet <- homeworkSetReduced[inTrain,]
testingSet <- homeworkSetReduced[-inTrain,]


preProc <- preProcess(trainingSet[,-1], method = "pca")
trainingSetPreProcessed <- predict(preProc, trainingSet[-1])
trainingSetPreProcessed <- cbind(trainingSet$classe, trainingSetPreProcessed)
colnames(trainingSetPreProcessed)[1] <- "classe"
library(randomForest)
head(trainingSetPreProcessed)
modelRF <- randomForest(classe ~ ., data = trainingSetPreProcessed)
modelRF

testingSetPreProcessed <- predict(preProc, testingSet[-1])
testClassePredicted <- predict(modelRF, testingSetPreProcessed)

table(testClassePredicted, testingSet$classe)
sum(testClassePredicted != testingSet$classe)/dim (testingSetPreProcessed)[1]

final20SetPreprocessed <- predict(preProc, final20SetReduced)
final20ClassePredicted <- predict(modelRF, final20SetPreprocessed)
final20ClassePredicted

## same analysis without PCA
modelRF2 <- randomForest(classe ~ ., data = trainingSet)
modelRF2

testClassePredicted2 <- predict(modelRF2, testingSet)

table(testClassePredicted2, testingSet$classe)
sum(testClassePredicted2 != testingSet$classe)/nrow(testingSet)

final20ClassePredicted2 <- predict(modelRF2, final20SetReduced)
final20ClassePredicted2

## tree model with PCA
modelRF3 <- train(classe ~ ., data = trainingSetPreProcessed, method = "rpart")
modelRF3

testClassePredicted <- predict(modelRF3, testingSetPreProcessed)

table(testClassePredicted, testingSet$classe)
sum(testClassePredicted != testingSet$classe)/dim (testingSetPreProcessed)[1]

final20SetPreprocessed3 <- predict(preProc, final20SetReduced)
final20ClassePredicted3 <- predict(modelRF3, final20SetPreprocessed)
final20ClassePredicted3



