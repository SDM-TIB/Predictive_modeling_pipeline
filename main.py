import os, argparse
import pandas as pd
import preprocessing_data
import sampling_strategy
import classification

def load(args):
    df = pd.read_csv(args.input_data+".csv")
    lst = args.data_list
    lst = lst.split(",")
    lst_target = args.target_list
    lst_target = lst_target.split(",")
    data = df[lst]
    target = df[lst_target]
    classes = args.classes
    class0 = classes.split(",")[0]
    class1 = classes.split(",")[1]

    return data,target, class0, class1,classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictive pipeline")
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--data_list", type=str , default=[],help="Independent variable for models (Features column names)")
    parser.add_argument("--target_list", type=str , default=[],help="Dependent variable for models (Output column name)")
    parser.add_argument('--classes', type=str, default=[], help="Classes for binary classification (from dependent variable)")
    parser.add_argument('--sampling', type=str, help="Is the data imbalanced? (yes/no)")
    parser.add_argument('--important_features', type=int, default=15 ,help="Number of important features")
    parser.add_argument('--cv_folds', type=int, default=5, help="Cross validation folds for stratified shuffle split of model to generate important features")

    args = parser.parse_args()

    data,target,class0,class1,classes = load(args)

    encode_data, encode_target = preprocessing_data.load_data(data, target, class0)

    samp = args.sampling
    if samp == "yes":
        print("Data is Imbalanced so Sampling Required")
        sampled_data, sampled_target = sampling_strategy.sampling(encode_data, encode_target)
    else:
        print("Data is Balanced so sampling is not required")
        sampled_data, sampled_target = encode_data, encode_target

    imp_features = args.important_features
    cross_validation = args.cv_folds
    result = classification.binary_classification(sampled_data, sampled_target,imp_features,cross_validation, classes)

















