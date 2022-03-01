# Predictive_modeling_pipeline


Step1: Modules to be installed

1) pip install imblearn
2) pip install lime
3) pip install pydotplus
4) pip install svglib
5) pip install colour
6) pip install graphviz
7) pip install dtreeviz

Step 2: Data preparation <br /><br />
Preparing dataset with 


Step 3: Run the command - <br /> <br />
python main.py --input_data dataset/dataset_trial_1 --data_list patient_id,gender,age,smoker,family_degree,cancer_type --target_list patient_id,mutation --classes ALK,others --sampling yes --important_features 15 --cv_folds 5
<br /><br />
Description of command line:<br /><br />

--input_data = path of dataset <br />
--data_list = Independent variable for models <br />
--target_list = Dependent variable for models <br />
--classes = Name of classes desired for binary classification <br />
--sampling = If data is imbalanced then yes/no <br />
--important features = give features size that needs to be considered most important <br />
--cv_folds = provide folds for stratifiedshuffle split <br />
