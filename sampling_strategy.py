import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

def sampling(encode_data,encode_target):
    print("-------- Undersampling strategy -----------")
    sampling_strategy = "not minority"
    autopct = "%.2f"
    #print(encode_data)
    X = encode_data
    Y = encode_target['class']
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=123)
    X_res, y_res = rus.fit_resample(X, Y)
    y_res.value_counts().plot.pie(autopct=autopct)
    plt.title("Under-sampling")
    plt.savefig('output/plots/Under-sampling.png')
    print("******** Undersampling Plot saved in output folder ***********")
    y_res = pd.DataFrame(data=y_res)
    #print(y_res)
    y_res.to_csv('dataset/target_under_sampling.csv')
    X_res.to_csv('dataset/data_under_sampling.csv', index=None)
    return X_res, y_res
