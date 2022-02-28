# Predictive_modeling_pipeline


Step1: Modules to be installed

1) pip install imblearn
2) pip install lime
3) pip install pydotplus
4) pip install svglib
5) pip install colour
6) pip install graphviz
7) pip install dtreeviz

Step 2: Data preparation


Step 3: Run the command - 
python3 main.py --input_data dataset/dataset_trial_1 --data_list patient_id,gender,age,smoker,family_degree,cancer_type --target_list patient_id,mutation 
--classes ALK,others --sampling yes --important_features 15 --cv_folds 5

Description of command line:

--input_data = path of dataset
--data_list = Independent variable for models
--target_list = Dependent variable for models
--classes = Name of class
--sampling = If data is imbalanced then yes/no
--important features = give features size that needs to be considered most important
--cv_folds = provide folds for stratifiedshuffle split
