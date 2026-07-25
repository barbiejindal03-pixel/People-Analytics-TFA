# Goal: Predict which applicants are likely to withdraw from TFA's 6-stage
# admissions pipeline, using classifiers trained on engineered behavioral,
# academic, and sentiment features.
#
# This script covers data cleaning, feature engineering, and the KNN + ANN
# modeling pipeline. The team additionally built and compared Naive Bayes,
# SVM, and Decision Tree models on the same cleaned dataset (org_model) --
# see TFA_Final_Presentation.pdf for the full 5-model comparison. Final
# result: KNN was selected as the best-performing model (highest accuracy,
# highest Kappa, and the best sensitivity/specificity balance).

# ---- Libraries
library(caret)
library(tidyverse)
library(dplyr)
library(lubridate)
library(ggplot2)
library(nnet)          # for ANN
library(C50)           # for Decision Tree
library(e1071)
install.packages("fastDummies")
library(fastDummies)
library(NeuralNetTools)

set.seed(123)

# ---- Load Dataset 
org <- read.csv("~/Downloads/excel data set csv.csv", stringsAsFactors = FALSE)
head(org)
str(org)
summary(org)

# ---- Convert Dates + Create Early Submission Feature 
org$Application.Deadline <- mdy(org$Application.Deadline)
org$Submitted.Date <- mdy(org$Submitted.Date)
org$earlysubmission <- as.numeric(
  difftime(org$Application.Deadline, org$Submitted.Date, units = "days")
)

# ---- Remove Irrelevant / Redundant Columns 
# (post-decision leakage vars, high-cardinality fields, operational timestamps)
org <- subset(org, select = -c(
  Person.Id, Major.2, Major.1...Cleaned, Minor...Cleaned, Undergraduate.University,
  Major.2...Cleaned, Application.Deadline, Submitted.Date,
  Undergraduate.University...Cleaned, Attended.Event, Met, Invited, Responded,
  Essays.Topic.0, Essays.Topic.1, Essays.Topic.2, Essays.Topic.3, Essays.Topic.4,
  Essays.Topic.5, Essays.Topic.6, Essays.Topic.7, Essays.Topic.8, Essays.Topic.9,
  Sign.up.Date, Started.Date, Confirmed.TFA.Offer
))

# ---- Convert Categorical Predictors to Factors 
org$App.Started.Year..RTAT. <- factor(org$App.Started.Year..RTAT.)
org$Is.Math..Sci..or.Eng.Major.Minor <- factor(org$Is.Math..Sci..or.Eng.Major.Minor)
org$Region.Preference.Level <- factor(org$Region.Preference.Level)
org$Completed.Admissions.Process <- factor(org$Completed.Admissions.Process)
org$School.Selectivity <- factor(org$School.Selectivity)

# ---- GPA Cleaning, Clustering & Imputation 
org$Cumulative.GPA[org$Cumulative.GPA == 0] <- NA
gpa_data <- na.omit(org$Cumulative.GPA)
gpa_matrix <- matrix(gpa_data, ncol = 1)

set.seed(123)
gpa_clusters <- kmeans(gpa_matrix, centers = 3)
print(gpa_clusters$centers)  # ~2.7 / 3.2 / 3.7 -> Low / Medium / High

org$GPA_Group <- cut(
  org$Cumulative.GPA,
  breaks = c(0, 3, 3.5, 4),
  labels = c("Low", "Medium", "High")
)
org$Cumulative.GPA[is.na(org$Cumulative.GPA)] <- mean(org$Cumulative.GPA, na.rm = TRUE)

# ---- Dummy Variables for Low-Cardinality Factors 
dummy_obj <- dummyVars(
  ~ School.Selectivity + Region.Preference.Level +
    Is.Math..Sci..or.Eng.Major.Minor + App.Started.Year..RTAT.,
  data = org,
  fullRank = TRUE
)
dummy_data <- data.frame(predict(dummy_obj, newdata = org))

org_model <- cbind(
  org %>% select(
    -School.Selectivity, -Region.Preference.Level,
    -Is.Math..Sci..or.Eng.Major.Minor, -App.Started.Year..RTAT.
  ),
  dummy_data
)
str(org_model)

# Remove remaining high-cardinality fields before modeling
org_model <- org_model %>% select(-Major.1, -Minor)

# ---- Train-Test Split (80/20) 
set.seed(123)
train_index <- createDataPartition(org_model$Completed.Admissions.Process, p = 0.8, list = FALSE)
train_data <- org_model[train_index, ]
test_data <- org_model[-train_index, ]

# ---- Standardize Numeric Predictors 
num_vars <- sapply(train_data, is.numeric)
preproc <- preProcess(train_data[, num_vars], method = c("center", "scale"))
train_scaled <- train_data
train_scaled[, num_vars] <- predict(preproc, train_data[, num_vars])
test_scaled <- test_data
test_scaled[, num_vars] <- predict(preproc, test_data[, num_vars])

train_scaled <- na.omit(train_scaled)
test_scaled <- na.omit(test_scaled)

table(train_scaled$Completed.Admissions.Process)
table(test_scaled$Completed.Admissions.Process)

# MODEL 1: K-Nearest Neighbors (KNN)

set.seed(123)
sample_index <- createDataPartition(
  train_scaled$Completed.Admissions.Process,
  p = 15000 / nrow(train_scaled),
  list = FALSE
)
train_small <- train_scaled[sample_index, ]

knn_model <- train(
  Completed.Admissions.Process ~ .,
  data = train_small,
  method = "knn",
  tuneLength = 5,
  trControl = trainControl(method = "cv", number = 5, allowParallel = FALSE)
)
knn_model

knn_pred <- predict(knn_model, newdata = test_scaled)
confusionMatrix(knn_pred, test_scaled$Completed.Admissions.Process)
# Note: an early iteration of this pipeline (smaller/differently-sampled
# training data) showed poor KNN sensitivity. After refinement, KNN was the
# team's top-performing model in the final comparison -- see
# TFA_Final_Presentation.pdf for the final 5-model results.

# MODEL 2: Artificial Neural Network (ANN)

set.seed(123)
tfa_data <- org_model
tfa_data$GPA_Group <- cut(
  tfa_data$Cumulative.GPA,
  breaks = c(-Inf, 3, 3.5, Inf),
  labels = c("Low", "Medium", "High")
)
tfa_data <- dummy_cols(
  tfa_data,
  select_columns = "GPA_Group",
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

set.seed(123)
train_index <- createDataPartition(tfa_data$Completed.Admissions.Process, p = 0.8, list = FALSE)
train_data <- tfa_data[train_index, ]
test_data <- tfa_data[-train_index, ]

ann_grid <- expand.grid(
  size = seq(5, 15, 5),
  decay = seq(0.1, 0.7, 0.2)
)

ann_model <- train(
  Completed.Admissions.Process ~
    Cumulative.GPA + Essay.1.Length + Essay.2.Length + Essay.3.Length +
    Essays.Unique.Words + Essays.Sentiment + earlysubmission +
    School.Selectivity.less_selective + School.Selectivity.more_selective +
    School.Selectivity.most_selective + School.Selectivity.selective +
    School.Selectivity.unknown + Region.Preference.Level.2 +
    Region.Preference.Level.3 + Is.Math..Sci..or.Eng.Major.Minor.1 +
    App.Started.Year..RTAT..2016 + GPA_Group_Medium + GPA_Group_High,
  data = train_data,
  method = "nnet",
  metric = "Kappa",
  maxit = 500,
  preProcess = c("center", "scale"),
  trControl = trainControl(method = "cv", number = 2),
  tuneGrid = ann_grid,
  trace = FALSE
)

ann_pred <- predict(ann_model, newdata = test_data)
confusionMatrix(ann_pred, test_data$Completed.Admissions.Process)
# Result: 80.9% accuracy, Kappa 0.0545 in this run -- in the final team
# comparison across all 5 models, KNN outperformed ANN on Kappa and
# sensitivity/specificity balance and was selected as the final model.

# ---- Visualize Network 
net <- ann_model$finalModel
plotnet(net)

# Note: SVM, Decision Tree, and Naive Bayes can be added using the same
# cleaned dataset (org_model / tfa_data) as a next step.
