# Practical-machine-learning-project
The following packages are required for this analysis.

library(caret)
library(dplyr)
library(randomForest)
Reading the data
Note: assuming that the test and training csv data sets are contained in a directory called “Data”.

Read the training data into R, identifying “NA”, “” and “#DIV/0!” as NA strings

pmltrain <- read.csv("./Data/pml-training.csv",na.strings=c("NA","","#DIV/0!"))
Spliting the data into training and test sets
Split the plmtrain into a training set (for model training) and a test set (for predicting the out of sample error), splitting on the classe variable (this is the variable of interest) with a 70-30 split

set.seed(555)

# Taking 70% for the training data and 30% for the test data
inTrain <- createDataPartition(y = pmltrain$classe, list = FALSE, p=0.7)
trainData <- pmltrain[inTrain,]
testData <- pmltrain[-inTrain,]
The analysis is now conducted purely on the trainData until the model is build and an out of sample error is needed.

Identify variables that are mostly NAs
There are a number of na’s in the dataset

table(is.na(trainData))
## 
##   FALSE    TRUE 
##  851290 1346630
Find which variables (if any) that are mostly na values

naprops <- colSums(is.na(trainData))/nrow(trainData)
mostlyNAs <- names(naprops[naprops > 0.75]) # mostly being 75%
mostlyNACols <- which(naprops > 0.75) # there's about 100 of them
Take a random (small) sample from the training data
Take a small sample of the training data to work with

set.seed(1256)
smalltrain <- trainData %>% tbl_df %>% sample_n(size=1000)
Remove the variables that are made up of mostly NAs

smalltrain <- smalltrain[,-mostlyNACols]
Remove row number and user name as candidate predictors
Remove the row number (X) and user_name column

smalltrain <- smalltrain[,-grep("X|user_name",names(smalltrain))]
Remove the cvtd_timestamp variable as a candidate predictor
This factor variable makes prediction of the test set difficult and is reduandant when raw time data is available in the data set.

smalltrain <- smalltrain[,-grep("cvtd_timestamp",names(smalltrain))]
Remove candidate predictors that have near zero variance
smalltrain <- smalltrain[,-nearZeroVar(smalltrain)]
List of candidate predictors
modelVars <- names(smalltrain)
modelVars1 <- modelVars[-grep("classe",modelVars)] # remove the classe var
The predictors for the machine learning are

modelVars1
##  [1] "raw_timestamp_part_1" "raw_timestamp_part_2" "num_window"          
##  [4] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [7] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
## [10] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [13] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [16] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [19] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [22] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [25] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [28] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [31] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [34] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [37] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [40] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [43] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [46] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [49] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [52] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [55] "magnet_forearm_z"
Build a random forest model
Using a random forest with the predictors in modelVars1 to predict the classe variable.

set.seed(57)
cleanedTrainData <- trainData[,modelVars]
modelFit <- randomForest(classe ~., data=cleanedTrainData, type="class")
Get Error Estimates
Begin with an insample error estimate (from trainData - which is 70% of pml-training.csv)

## Get the values predicted by the model
predTrain <- predict(modelFit,newdata=trainData)

## Use a confusion matrix to get the insample error
confusionMatrix(predTrain,trainData$classe)$table
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
The in sample error is unrealistically high.

Now getting an out of sample error estimate (from testData - which is 30% of pml-training.csv)

classe_col <- grep("classe",names(testData))
predTest <- predict(modelFit, newdata = testData[,-classe_col], type="class")

confusionMatrix(predTest,testData$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    2    0    0    0
##          B    0 1137    3    0    0
##          C    0    0 1023    2    0
##          D    0    0    0  961    1
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                         
##                Accuracy : 0.998         
##                  95% CI : (0.997, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.998         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.998    0.997    0.997    0.999
## Specificity             1.000    0.999    1.000    1.000    1.000
## Pos Pred Value          0.999    0.997    0.998    0.999    0.999
## Neg Pred Value          1.000    1.000    0.999    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.163    0.184
## Detection Prevalence    0.285    0.194    0.174    0.163    0.184
## Balanced Accuracy       1.000    0.999    0.998    0.998    0.999
The model has an out of sample accuracy of: 0.998.

Prediciting exercise activity using the model
Load the pml-test data

pmltest <- read.csv("./Data/pml-testing.csv",na.strings=c("NA","","#DIV/0!"))
Perform the prediction

# plmtest predicition
predplmtest <- predict(modelFit, newdata = pmltest, type="class")
The final outcome is suppressed from presentation in keeping with the terms of the Coursera Honor Code.

print(predplmtest)
