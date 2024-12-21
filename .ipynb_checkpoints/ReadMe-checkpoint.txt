ReadME
=================================

Note: 
1. Make sure you run the requiremnts file
2. Navigate to the Final_Project_submission directory
3. Ensure all the accompanied files are present in the directory above
4. The entire code was developed and run in Narnia environment.
the training of the two cases each is <5min




Training and Testing
=====================
Case 1: ANN for No dependency among the targets (GAS)

 - At the CLI run code below

python case_1_ann_train.py

 - The program should run, train the model (< 7 min) using the dataset located in the working dircetory, generate performance results, and save the trained model for inferencing anytime


Case 2: RandomForest with LabelPowerset for creating dependency among the targets (GAS)

 - At the CLI run code below

python case_2_rf_train.py

 - The program should run, train the model (< 7 min) using the dataset located in the working dircetory, generate performance results, and save the trained model for inferencing anytime




Inference (in a single cli)
=============================

To make inference on model for Case 1, Run code below

python inference.py model1 


To make inference on model for Case 2, Run code below

python inference.py model2 


In all the cases above, you will be prompted to enter a security incident description. You may describe any cybersecurity incident you are aware of or copy paste a sample from sample_incident.txt located in the working directory.

 - Press Enter


Regards
