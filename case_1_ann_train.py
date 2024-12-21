#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


# Custom functions for precision, recall, and F1 score
def precision_score_custom(y_true, y_pred):
    TP = ((y_true == 1) & (y_pred == 1)).sum().item()
    FP = ((y_true == 0) & (y_pred == 1)).sum().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision

def recall_score(y_true, y_pred):
    TP = ((y_true == 1) & (y_pred == 1)).sum().item()
    FN = ((y_true == 1) & (y_pred == 0)).sum().item()
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recall

def f1_score_custom(y_true, y_pred):
    precision = precision_score_custom(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


# In[4]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump, load
# from sklearn.metrics import precision_score, f1_score

# Load your data
data = pd.read_csv('combined_dataset.csv')
# Handle missing values
data = data.dropna(subset=["incident"])  # Drop rows with missing 'incident'

# Assuming the dataset has columns 'incident', 'goal', 'assets', 'solutions'
feature = 'incident'
targets = ['goal', 'assets', 'solutions']

# Prepare the TF-IDF vectorizer
# tfidf = TfidfVectorizer()
# tfidf_vectorizer = TfidfVectorizer()

class AmujoANN(nn.Module):
    def __init__(self, input_size):
        super(AmujoANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Split the data for each target and train a model
for target in targets:
    # Prepare features and labels
    X = data[feature]
    y = data[target]
    tfidf_vectorizer = TfidfVectorizer()
    # Transform the text data to TF-IDF features
    X_tfidf = tfidf_vectorizer.fit_transform(X).toarray()
    # tfidf_vectorizer.fit_transform(X).toarray()
    dump(tfidf_vectorizer, './models/tfidf_vectorizer.joblib')
    # Encode the target labels and convert to float
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y).astype(float)
    
    # Ensure target values are in range [0, 1]
    y_encoded = y_encoded / y_encoded.max()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize the model, loss function, and optimizer
    model = AmujoANN(input_size=X_tfidf.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    
    # Train the model
    epochs = 20
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Save the trained model 
    torch.save(model.state_dict(), f'./models/model_{target}.pth')
    # dump(X_tfidf, './models/tfidf_vectorizer.joblib')

    
    # Evaluate the model
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs >= 0.4).float()
        accuracy = (predictions == y_test_tensor).sum().item() / y_test_tensor.size(0)
        precision = precision_score_custom(y_test_tensor, predictions)
        recall = recall_score(y_test_tensor, predictions)
        f1 = f1_score_custom(y_test_tensor, predictions)
        print(f'Accuracy for target {target}: {accuracy}')
        print(f'Precision for target {target}: {precision}')
        print(f'Recall for target {target}: {recall}')
        print(f'F1 Score for target {target}: {f1}')
        print("===================================================================")


# In[5]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
import torch

class AmujoANNInf(nn.Module):
    def __init__(self, input_size):
        super(AmujoANNInf, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def load_model(input_size, model_path):
    model = AmujoANNInf(input_size=input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def make_inference(model, incident_text, tfidf_vectorizer):
    X_tfidf = tfidf_vectorizer.transform([incident_text])
    X_tensor = torch.tensor(X_tfidf.toarray(), dtype=torch.float32)
    with torch.no_grad():
        output = model(X_tensor)
        prediction = (output >= 0.5).float().item()
    return prediction

# Load the fitted TF-IDF vectorizer
tfidf = load('./models/tfidf_vectorizer.joblib')

# Check if the loaded object is a TfidfVectorizer
if isinstance(tfidf, TfidfVectorizer):
    # Calculate input size
    input_size = len(tfidf.get_feature_names_out())
else:
    raise ValueError(f"Loaded object is {type(tfidf)} not a TfidfVectorizer")

# Load the label encoders
label_encoders = {
    target: load(f'./models/label_encoder_{target}.joblib') for target in ['goal', 'assets', 'solutions']
}

# Load the trained models
goal_model = load_model(input_size=input_size, model_path='./models/model_goal.pth')
assets_model = load_model(input_size=input_size, model_path='./models/model_assets.pth')
solutions_model = load_model(input_size=input_size, model_path='./models/model_solutions.pth')

# Example of making inferences
incident_text = "incident: Silent Librarian has exfiltrated entire mailboxes from compromised accounts"
goal_prediction = make_inference(goal_model, incident_text, tfidf)
assets_prediction = make_inference(assets_model, incident_text, tfidf)
solutions_prediction = make_inference(solutions_model, incident_text, tfidf)

# Convert predictions to class labels
goal_label = label_encoders['goal'].inverse_transform([int(goal_prediction)])[0]
assets_label = label_encoders['assets'].inverse_transform([int(assets_prediction)])[0]
solutions_label = label_encoders['solutions'].inverse_transform([int(solutions_prediction)])[0]

print(f'Predicted goal: {goal_label}')
print(f'Predicted assets: {assets_label}')
print(f'Predicted solutions: {solutions_label}')


# In[ ]:




