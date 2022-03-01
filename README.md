# Predictive_modeling_pipeline


Step1: Run for necessary modules installation <br /><br />
pip install requirements.txt

Step 2: Data preparation <br /><br />
Preparing dataset with all the features and index column (example - id of patient). Also specify the the index in both independent (explanatory variable) and dependent variable (response variable)

<br />

Step 3: Run the command in the command prompt window - <br /> <br />
python main.py --input_data dataset/dataset_trial_1 --data_list patient_id,gender,age,smoker,family_degree,cancer_type --target_list patient_id,mutation --classes ALK,others --sampling yes --important_features 15 --cv_folds 5
<br />

Description of command line:<br />

--input_data = path of dataset <br />
--data_list = Independent variable for models <br />
--target_list = Dependent variable for models <br />
--classes = Name of classes desired for binary classification <br />
--sampling = If data is imbalanced then yes/no <br />
--important features = give features size that needs to be considered most important <br />
--cv_folds = provide folds for stratifiedshuffle split <br />
