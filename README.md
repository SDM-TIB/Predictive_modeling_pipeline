# Predictive_modeling_pipeline


Step1: Modules to be installed

1) pip install imblearn
2) pip install lime
3) pip install pydotplus
4) pip install svglib


Step 2: Data preparation


Step 3: Run the command
python3 main.py --input_data 'dataset/dataset_trial_1' --data_list patient_id,gender,age,smoker,family_degree,cancer_type --target_list patient_id,mutation 
--classes ALK,others --sampling yes --important_features 15 --cv_folds 5

--input_data = 
