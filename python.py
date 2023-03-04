import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
df = pd.read_csv('customer_churn.csv')

# Visualize data
sns.countplot(x='Churn', data=df)

# Preprocess data
df = pd.get_dummies(df, columns=['Gender', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract', 'PaymentMethod'], drop_first=True)
X = df.drop(['Churn'], axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy)
print('Confusion matrix:\n', confusion)
