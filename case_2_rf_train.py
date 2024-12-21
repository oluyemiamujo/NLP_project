#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skmultilearn.problem_transform import LabelPowerset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np


# In[2]:


# 1. Data Preparation
df = pd.read_csv('combined_dataset.csv')

# Clean and preprocess text data (example)
df['incident'] = df['incident'].str.replace('[^a-zA-Z0-9\s]', '', regex=True)

# Convert target columns to lists for multi-label binarization
df['targets'] = df.apply(lambda row: [row['goal'], row['assets'], row['solutions']], axis=1)


# In[3]:


# Binarize the multi-label targets
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['targets'])

# Generate embeddings for the incident descriptions
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
X = np.array(embedding_model.encode(df['incident'].tolist()))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[4]:


# Initialize Label Powerset with RandomForestClassifier
lp = LabelPowerset(RandomForestClassifier(random_state=42))

# Train the model
lp.fit(X_train, y_train)

# Predict on the test set
y_pred_lp = lp.predict(X_test)

# Evaluate
# print("Label Powerset Classification Report:")
# print(classification_report(y_test, y_pred_lp, target_names=mlb.classes_))


# In[5]:


# Train models
import joblib 

joblib.dump(lp, './models/labelPowerset_RFC.pkl')


# In[7]:


from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score

def performance(y_test, y_pred):
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'accuracy': accuracy, 
        'precision': precision, 
        'f1': f1,
        'recall': recall, 
    }

goal_perf = performance(y_test, y_pred_lp)
print(goal_perf)


# # Inference

# In[23]:




