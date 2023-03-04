library(tidyverse)
library(caret)

# Load data
df <- read.csv('customer_churn.csv')

# Visualize data
ggplot(df, aes(x=Churn, fill=Churn)) +
  geom_bar()

# Preprocess data
df <- df %>% 
  mutate(Gender = ifelse(Gender == 'Male', 1, 0),
         PhoneService = ifelse(PhoneService == 'Yes', 1, 0),
         MultipleLines = ifelse(MultipleLines == 'Yes', 1, 0),
         InternetService = ifelse(InternetService == 'Fiber optic', 1, 0),
         Contract = ifelse(Contract == 'One year', 1, 0),
         PaymentMethod = ifelse(PaymentMethod == 'Electronic check', 1, 0)) %>% 
  select(-customerID)
X <- df %>% select(-Churn)
y <- df$Churn
trainIndex <- createDataPartition(y, p=0.8, list=FALSE)
X_train <- X[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X[-trainIndex, ]
y_test <- y[-trainIndex]

# Train model
model <- train(x=X_train, y=y_train, method='glm', family='binomial')

# Evaluate model
y_pred <- predict(model, X_test)
accuracy <- mean(y_pred == y_test)
confusion <- confusionMatrix(y_pred, y_test)
print(paste('Accuracy:', accuracy))
print(confusion$table)
