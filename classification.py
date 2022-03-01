import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, make_scorer
import lime
import lime.lime_tabular
from sklearn import tree
import seaborn as sns
import dtreeviz_lib
#from dtreeviz.trees import *
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def plot_feature_importance(importance, names, model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(15, 15))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + '_FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    print("****************** Plot for important features is saved in output folder ******************")
    plt.savefig('output/plots/Random_Forest_Feature_importance.png')


def binary_classification(sampled_data, sampled_target, imp_features, cross_validation, classes):
    sampled_data = sampled_data.drop(columns="first_col")
    sampled_target['class'] = sampled_target['class'].astype(int)
    # print(sampled_data)
    # print(sampled_target)
    X = sampled_data
    y = sampled_target['class']

    X_imput, y_imput = X.values, y.values
    print("---------------- Random Forest Classification with Stratified shuffle split -----------------------")
    rf_estimator = RandomForestClassifier(max_depth=4, random_state=0)
    cv = StratifiedShuffleSplit(n_splits=cross_validation, test_size=0.3, random_state=123)
    important_features = set()
    important_features_size = imp_features
    print("************** Classification report for every iteration ************************************")
    for i, (train, test) in enumerate(cv.split(X_imput, y_imput)):
        rf_estimator.fit(X_imput[train], y_imput[train])
        y_predicted = rf_estimator.predict(X_imput[test])

        print(classification_report(y_imput[test], y_predicted))

        fea_importance = rf_estimator.feature_importances_
        indices = np.argsort(fea_importance)[::-1]
        for f in range(important_features_size):
            important_features.add(X.columns.values[indices[f]])

    plot_feature_importance(rf_estimator.feature_importances_, X.columns, 'RANDOM FOREST')

    # Taking important features
    new_sampled_data = sampled_data[list(important_features)]
    X_train, X_test, y_train, y_test = train_test_split(new_sampled_data.values, sampled_target['class'].values,
                                                        random_state=123)

    feature_names = new_sampled_data.columns
    parameters = {"max_depth": range(4, 6)}

    # Defining Decision tree Classifier
    clf = tree.DecisionTreeClassifier()

    # GrdiSearchCV to select best hyperparameters
    grid = GridSearchCV(estimator=clf, param_grid=parameters)
    grid_res = grid.fit(X_train, y_train)
    best_clf = grid_res.best_estimator_

    # predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_clf.score(X_test, y_test)
    y_pred = best_clf.predict(X_test)

    # lime interpretability

    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),
                                                       feature_names=new_sampled_data.columns.values,
                                                       class_names=best_clf.classes_, discretize_continuous=True,
                                                       random_state=123)

    # exp = explainer.explain_instance(X_test[5], best_clf.predict_proba, num_features=10)

    [explainer.explain_instance(i, best_clf.predict_proba, num_features=10).save_to_file('output/Lime results/Lime' + str(j) + '.html') for j, i in enumerate(X_test)]
    print(
        "***************************** Lime Interpretability results saved in output folder ****************************")

    target_names = classes.split(",")
    print("****************** Classification report saved in output folder *************************")
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    classificationreport = pd.DataFrame(report).transpose()
    # print(classificationreport)
    classificationreport.to_csv("output/Final Classiciation report.csv", index=True)

    bool_feature = []
    for feature in new_sampled_data.columns:
        values = new_sampled_data[feature].unique()
        if len(values) == 2:
            values = sorted(values)
            if values[0] == 0 and values[1] == 1:
                bool_feature.append(feature)
    # print(bool_feature)

    viz = dtreeviz_lib.dtreeviz(best_clf, new_sampled_data, sampled_target['class'], target_name='class',
                                feature_names=feature_names, class_names=target_names, fancy=False,
                                show_root_edge_labels=True, bool_feature=bool_feature)
    viz.save('output/plots/RF_undersampling_final_results.svg')
    print("****** Decision tree plot saved in output/plot folder *********")

    return classificationreport
