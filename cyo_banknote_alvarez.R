

############################    1  INTRODUCTION    ###########################


# Data Science "Choose Your Own" Project
# BANK NOTE AUTHENTICATION PROJECT
# Author: Rocío Alvarez Mercé

# Git Repo available at: https://github.com/rocioam/Capstone-CYO


# Please follow the code as presented :)
# If not, after reading the dataset rush to section 4 to see an important
# message, or code won't run after


#######################    2 PACKAGES AND LIBRARIES    #######################


# Here are all the packages and libraries needed to run this code.
# If you have problems installing the "AppliedPredictiveModeling"
# package, install it from RStudio > Tools > Install Packages

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", 
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", 
                                     repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", 
                                     repos = "http://cran.us.r-project.org")
if(!require(AppliedPredictiveModeling)) install.packages("AppliedPredictiveModeling",
                                                         repos="http://R-Forge.R-project.org")
if(!require(rpart)) install.packages("rpart", 
                                     repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", 
                                            repos = "http://cran.us.r-project.org")

# Libraries

library(tidyverse)
library(caret)
library(dplyr)
library(AppliedPredictiveModeling)
library(rpart)
library(randomForest)





#############################    3 DATASET    ###########################


# There are three options to download/load the dataset:

# Option 1: The dataset can be downloaded directly from the website 
# of the UCI Machine Learning Repository, using this link: 
# https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt 
# and save it as a *.csv* file    **VERY IMPORTANT**


# Option 2: The *.csv* file can be downloaded directly from my
# Git Repo mentioned at the start of this file, and then use
# this code to read it:


banknote_auth <- read.csv('data_banknote_authentication.csv', header=FALSE)
colnames(banknote_auth)<- c("variance_wt","skewness_wt",
                            "curtosis_wt","entropy","authentic")



# Option 3: Reading the dataset directly from the UCI website
# using this code  **you need an active internet connection**

banknote_auth <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"),header=FALSE)

colnames(banknote_auth)<- c("variance_wt","skewness_wt",
                            "curtosis_wt","entropy","authentic")





###########################    4 DATA ANALYSIS    #########################


# Check if the dataframe was loaded correctly, if not, go back to 
# section 3.

class(banknote_auth)


# Getting to know the dataset

dim(banknote_auth)
head(banknote_auth)
sapply(banknote_auth, class)
summary(banknote_auth$authentic)


# Let's see the classification column more in depth
summary(banknote_auth$authentic)


# Barplot for the classification variable
library(ggplot2)
banknote_auth %>% ggplot(aes(factor(authentic), 
                             fill = authentic )) + guides(fill = F) +
  geom_bar() + labs(y = "Count", x = "", title = "Banknotes Count") + 
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_x_discrete(labels=c("Fake", "Authentic"))

# Summarize information
percentage <- prop.table(table(banknote_auth$authentic)) * 100
cbind(Count = table(banknote_auth$authentic), Percentage = percentage)


# Change data class to factor for last column
# **IT'S VERY IMPORTANT TO RUN THIS TO GET THE CODE TO WORK AFTER**
banknote_auth$authentic <- as.factor(banknote_auth$authentic)
                                            

# Now we have another class type
class(banknote_auth$authentic)
levels(banknote_auth$authentic)


# Attribute analysis: what do we have to work with?
summary(banknote_auth)

  
#########################    Data visualization    ##########################

# Predictors Boxplot

x <- banknote_auth[,1:4]
y <- banknote_auth[,5]

par(mfrow = c(1,4))
for(i in 1:4) {
  boxplot(x[,i], col = 'powderblue', main = names(banknote_auth)[i]) 
}
  

# Separate boxplots for binary classification
featurePlot(x = x, y = y, "box")
  
 
# Density Plots 
transparentTheme(trans = .9)
featurePlot(x = x, y = y, plot = "density", 
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(2, 2), 
            auto.key = list(columns = 2))


# Plot all parameters with respect to all parameters
transparentTheme(set = TRUE, pchSize = 0.4, trans = .6)
featurePlot(x = banknote_auth[,1:4], y = banknote_auth[,5], "ellipse",
            auto.key = list(columns = 2))




########################    5 MACHINE LEARNING    #######################


# Create Train and Test sets
set.seed(60) 
test_index <- createDataPartition(banknote_auth$authentic, times = 1, p = 0.2, list = F)
train_set <- banknote_auth[-test_index,]
test_set <- banknote_auth[test_index,]


# Sets dimension
dimtrain <- dim(train_set)
dimtest <- dim(test_set)
dimtrain
dimtest
     

# Quick check to see that the sets are balanced
checktrain <- table(train_set$authentic)/nrow(train_set)
checktest <- table(test_set$authentic)/nrow(test_set)

checktable <- matrix(c(checktrain[1],checktest[1],checktrain[2],checktest[2]), 
                     ncol=2)
colnames(checktable) <- c('Fake (= 0)', 'Authentic (=1 )')
rownames(checktable) <- c('Train Set', ' Test Set')
as.table(checktable)


##### Baseline Model: Logistic Regression #####

# Traning using all 4 predictors
glm_fit <- train_set %>% glm(authentic ~ variance_wt + skewness_wt + 
                               curtosis_wt + entropy, 
                             data =., family = "binomial")

# Get predictions and metrics
p_hat_logit <- predict(glm_fit, newdata = test_set, type = "response")
y_hat_logit <- ifelse(p_hat_logit > 0.5, 1, 0) %>% factor(levels = c(0, 1))
confusionMatrix(y_hat_logit, test_set$authentic)
acc_baseline <- confusionMatrix(y_hat_logit, test_set$authentic)$overall[1]


# Create a little df with the accuracy results
accuracy_results <- data_frame(Method = "Baseline Model", 
                               Accuracy = acc_baseline)
accuracy_results



# Traning with only 3 predictors: all but *entropy*
glm_fit2 <- train_set %>% glm(authentic ~ variance_wt + skewness_wt + 
                                curtosis_wt, data =., family = "binomial")

# Predictions with 3 parameters
p_hat_logit2 <- predict(glm_fit2, newdata = test_set, type = "response")
y_hat_logit2 <- ifelse(p_hat_logit2 > 0.5, 1, 0) %>% factor(levels = c(0, 1))
confusionMatrix(y_hat_logit2, test_set$authentic)





##### Second Model: Decision Tree #####
# Tree fitting
fit_tree <- rpart(banknote_auth$authentic~., data = banknote_auth)
fit_tree

# Tree plotting
plot(fit_tree)
text(fit_tree, cex = 0.7)


# Training the algorithm
# Finding the best cp using cross-validation
xrpart <- train_set[, 1:4]
yrpart <- train_set$authentic
train_rpart <- train(xrpart, yrpart, method = "rpart", 
                     tuneGrid = data.frame(cp = seq (0,0.8,len = 35)))

plot(train_rpart)

# Predictions and metrics for Decision Treee
acc_tree <- confusionMatrix(predict(train_rpart, test_set), 
                            test_set$authentic)$overall["Accuracy"]
confusionMatrix(predict(train_rpart, test_set), test_set$authentic)
acc_tree

accuracy_results <- bind_rows(accuracy_results,
                              data_frame(Method = "Decision Tree", 
                                         Accuracy = acc_tree))
accuracy_results



##### Third Model: Random Forest #####

# Training the model
train_rf <- randomForest(train_set$authentic~., data = train_set)

# Predictions and metrics
confusionMatrix(predict(train_rf, test_set), test_set$authentic)
acc_rf <- confusionMatrix(predict(train_rf, test_set), 
                          test_set$authentic)$overall[1]

accuracy_results <- bind_rows(accuracy_results,
                              data_frame(Method = "Random Forest", 
                                         Accuracy = acc_rf))
accuracy_results



##### Fourth Model: K-Nearest Neighbours #####

# Training the model
knn_fit <- knn3(train_set$authentic ~., data = train_set, k = 5)

# Predictions and metrics
y_hat_knn <- predict(knn_fit, test_set, type = "class")
confusionMatrix(data = y_hat_knn, reference = test_set$authentic)

acc_knn <- confusionMatrix(y_hat_knn, test_set$authentic)$overall[1]
accuracy_results <- bind_rows(accuracy_results,
                              data_frame(Method = "KNN", 
                                         Accuracy = acc_knn))
accuracy_results


# Try KNN model with only 3 predictors: all but *entropy*
knn_fit2 <- train_set %>% knn3(authentic ~ variance_wt + 
                                 skewness_wt + curtosis_wt, data =.)
y_hat_knn2 <- predict(knn_fit2, test_set, type = "class")
confusionMatrix(data = y_hat_knn2, reference = test_set$authentic)



# Final accuracy for all 4 models using 4 predictors
accuracy_results



                                            