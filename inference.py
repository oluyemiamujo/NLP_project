import argparse
import os
import joblib 
from sentence_transformers import SentenceTransformer
# import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore")

loaded_model = joblib.load('./models/labelPowerset_RFC.pkl')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def mapper(problem, result):
    # print(problem)
    x = dict(zip(result, problem))
    mapp = [key for key, value in x.items() if value == 1]

    return mapp

def RF_fn(text):
    goal_lst = ['availability', 'confidentiality', 'integrity']
    assets_lst = ['application software', 'authentication data', 'email accounts', 'networks', 'operating system', 'running processes', 'users data']
    solution_lst = ['access controls', 'application whitelisting', 'data loss prevention', 'disable macros', 'endpoint detection and response', 'firewall rules harden', 'multifactor authentication', 'network segmentation and filtering', 'stronger key exchange algorithms', 'update security patches', 'user account control']

    incident_embeddings = np.array(embedding_model.encode(text)).reshape(1, -1) #.encode(incident, show_progress_bar=True).reshape(1, -1)
    predicted = loaded_model.predict(incident_embeddings)    
    pred = predicted.toarray().tolist()[0]
    g = mapper(pred[0:3], goal_lst)
    a = mapper(pred[3:7], assets_lst)
    s = mapper(pred[7:], solution_lst)
    print("\n=============RF Dependent GAS==================================")
    gg = ''.join(g) if g else 'Unknown'
    aa = ''.join(a) if a else 'Unknown'
    ss = ''.join(s) if s else 'Unknown'

    print(f'Predicted goal: {gg}')
    print(f'Predicted assets: {aa}')
    print(f'Predicted solutions: {ss}')



# text = "Adversaries may gain access and continuously communicate with victims by injecting malicious content into systems through online network traffic. Rather than luring victims to malicious payloads hosted on a compromised website (i.e., Drive-by Target followed by Drive-by Compromise), adversaries may initially access victims through compromised data-transfer channels where they can manipulate traffic and/or inject their own content. These compromised online network channels may also be used to deliver additional payloads (i.e., Ingress Tool Transfer) and other data to already compromised systems."



import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
import torch
import torch.nn as nn


class AmujoANN(nn.Module):
    def __init__(self, input_size):
        super(AmujoANN, self).__init__()
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
    model = AmujoANN(input_size=input_size)
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

def ANN_fn(incident_text):
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
    # incident_text = "incident: Silent Librarian has exfiltrated entire mailboxes from compromised accounts"
    goal_prediction = make_inference(goal_model, incident_text, tfidf)
    assets_prediction = make_inference(assets_model, incident_text, tfidf)
    solutions_prediction = make_inference(solutions_model, incident_text, tfidf)
    
    # Convert predictions to class labels
    goal_label = label_encoders['goal'].inverse_transform([int(goal_prediction)])[0]
    assets_label = label_encoders['assets'].inverse_transform([int(assets_prediction)])[0]
    solutions_label = label_encoders['solutions'].inverse_transform([int(solutions_prediction)])[0]
    print("\n=============ANN Independent GAS==================================")
    
    print(f'Predicted goal: {goal_label}')
    print(f'Predicted assets: {assets_label}')
    print(f'Predicted solutions: {solutions_label}')







def inputIncident():
    incident = ''
    incident = input("Describe the incident: ")
    return incident



def main(input_value):
    # print(f"Received input: {input_value}")
    if input_value == "model1":
        incident = inputIncident()
        if incident:
            ANN_fn(incident)
        else:
            print("incident is empty!")
    elif input_value == "model2":
        incident = inputIncident()
        if incident:
            RF_fn(incident)
        else:
            print("incident is empty!")
    else:
        print("Wrong model!")

            
if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Process input for the script.")
    
    # Add an argument for input
    parser.add_argument("input_value", type=str, help="Input value to process")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Pass the input to the main function
    main(args.input_value)
