# PRODIGY_DS_03
Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. Use a dataset such as the Bank Marketing dataset from the UCI Machine Learning Repository.
Additionally data visualiztion using PowerBI is done to check model's performance.


Dataset link:https://archive.ics.uci.edu/dataset/222/bank+marketing

#### Table of Contents
1. [Files](#files)
2. [Requirements](#requirements)
3. [Setup Instructions](#setup-instructions)
4. [Usage](#usage)
   - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
   - [Model Training and Evaluation](#model-training-and-evaluation)
   - [Saving Results](#saving-results)
   - [Printing Results](#printing-results)
5. [Jupyter Notebook Code](#jupyter-notebook-code)
6. [Contact](#contact)

---

#### 1. Files
- `decision_tree_classifier.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, evaluation, and result visualization.
- `predictions.csv`: CSV file containing the actual and predicted values for the test dataset.
- `results.csv`: CSV file containing the evaluation metrics of the model.

#### 2. Requirements
- Python 3.x
- pandas
- scikit-learn
- Jupyter Notebook

#### 3. Setup Instructions
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/vaidehii203/PRODIGY_DS_03.git
   cd decision-tree-classifier
   ```

2. **Install Dependencies:**
   ```sh
   pip install pandas scikit-learn
   ```

#### 4. Usage

##### Data Loading and Preprocessing
- The notebook loads the Bank Marketing dataset (`bank-additional-full.csv` for training and `bank-additional.csv` for testing) using pandas.
- Categorical variables in the dataset are encoded using `LabelEncoder`.

##### Model Training and Evaluation
- The Decision Tree Classifier from scikit-learn is used to build the model.
- Evaluation metrics such as accuracy, precision, recall, F1 score, and confusion matrix are computed using scikit-learn's metrics functions.

##### Saving Results
- Predictions and evaluation metrics are saved to CSV files (`predictions.csv` and `results.csv`).

##### Printing Results
- The notebook prints the evaluation metrics and displays the predictions DataFrame to the console for verification.

#### 5. Jupyter Notebook Code

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the dataset
train_data = pd.read_csv("D:\\intership tasks\\bank-additional\\bank-additional-full.csv", delimiter=';')
test_data = pd.read_csv("D:\\intership tasks\\bank-additional\\bank-additional.csv", delimiter=';')

# Display the first few rows of the dataset
train_data.head()

# Encode categorical variables in the training dataset
label_encoders = {}
for column in train_data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    train_data[column] = label_encoders[column].fit_transform(train_data[column])

# Encode categorical variables in the testing dataset using the same encoders
for column in test_data.select_dtypes(include=['object']).columns:
    if column in label_encoders:
        test_data[column] = label_encoders[column].transform(test_data[column])

# Train-test split
x_train = train_data.drop(columns=['y'])
y_train = train_data['y']

x_test = test_data.drop(columns=['y'])
y_test = test_data['y']

# Build and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(x_train, y_train)

# Predict
y_pred = clf.predict(x_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
recall = recall_score(y_test, y_pred, pos_label=1, average='binary')
f1 = f1_score(y_test, y_pred, pos_label=1, average='binary')
conf_matrix = confusion_matrix(y_test, y_pred)

# Save predictions and evaluation metrics to CSV
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('predictions.csv', index=False)

results = pd.DataFrame([{
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1
}])
results.to_csv('results.csv', index=False)

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nPredictions DataFrame:")
print(predictions_df)
```

#### 6. Contact
For more information or to get in touch, please visit my LinkedIn profile.https://www.linkedin.com/in/vaidehi-kale-b635b7264/

---

