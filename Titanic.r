train <- read.csv(file="data/train.csv")
test <- read.csv(file="data/test.csv")

summary(train)

train$Age[is.na(train$Age)] <- median(train$Age, na.rm = TRUE)
test$Age[is.na(test$Age)] <- median(train$Age, na.rm = TRUE)

summary(train)
summary(test)

library(randomForest)

my_forest <- randomForest(formula = as.factor(Survived) ~ Pclass + Sex + Age, importance=TRUE, ntree=100, data=train)

my_prediction <- predict(my_forest, train)
my_prediction <- predict(my_forest, test)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = my_prediction)

# Write your solution away to a csv file with the name my_solution.csv
write.csv(my_solution, file="my_solution.csv")
