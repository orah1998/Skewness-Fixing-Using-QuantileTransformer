from scipy.stats import spearmanr, kendalltau, gamma
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample
from sklearn.utils import compute_sample_weight
import pandas as pd
import numpy as np
from scipy.stats import skew, iqr, jarque_bera
from sklearn.preprocessing import QuantileTransformer
from sklearn import preprocessing
from dataprep.eda import plot, plot_correlation, create_report, plot_missing



'''
This function is our unique part of the project.
The function splits the data into the protected variable sub-groups using
quantiles in order to split the data while ignoring outliers, as quantile would enforce that
the groups have similar probabilty of a random sample to fall into.

We than proceed to measuring skewness using the skew() function, and for completeness of the check, we also
utilized common methods - IQR as well as JB_test
'''
def bin_splitting(data, column_to_split, target_variable):
    print("---- Bin Splitting started ----")
    data_copy = data.copy()
    threshold = 0.5     
    skewness = data[column_to_split].skew()

    # Here we will check for skewness using IQR to measure a heavy tailed distibution
    quantiles = data[column_to_split].quantile([0.25, 0.5, 0.75])
    iqr_condition = quantiles[0.75] - quantiles[0.25] > 2*(quantiles[0.5]-quantiles[0.25])

    jb_test = jarque_bera(data[column_to_split])
    jb_condition = jb_test[0] > jb_test[1]


    if abs(skewness) > threshold and (iqr_condition or jb_condition) :

        # transforming the data into qunatile bins
        bins, split_values = pd.qcut(data[column_to_split], q=3, retbins=True, labels=False, duplicates='drop')

        # Print the skewness value and its direction
        if skewness > 0:
            print(f"The {column_to_split} column is right-skewed")
            minority_data = data[data[column_to_split] >= split_values[-2]]
            majority_data = data[data[column_to_split] < split_values[-2]]
            return fix_distribution(minority_data, majority_data, target_variable, column_to_split)


        elif skewness < 0:
            print(f"The {column_to_split} column is left-skewed")
            minority_data = data[data[column_to_split] <= split_values[1]]
            majority_data = data[data[column_to_split] > split_values[1]]

            return fix_distribution(minority_data, majority_data, target_variable, column_to_split)

    else:
        return data
    


'''
    This function gets the splitted by quantiles data, and afterwards it fits the major group
    into the distribution of the minor group.
    Our approach is to fit the transformer on the minority data and then 
    apply the transform() function on majority data to ensure that the 
    majority data fits to the minority data.
'''
def fix_distribution(minority_data, majority_data, target_variable, protected_attribute):
            print("---- Distribution Tool in process ----")      
            print("--- Starting to handle skewness ---")

            transformer = QuantileTransformer(output_distribution='normal', random_state=42)

            # Saving a copy of the target variable
            temp_protected_column_minority = minority_data[protected_attribute].copy()
            temp_protected_column_majority = majority_data[protected_attribute].copy()
            temp_target_column_minority = minority_data[target_variable].copy()
            temp_target_column_majority = majority_data[target_variable].copy()

            minority_normalized = pd.DataFrame(transformer.fit_transform(minority_data.drop(target_variable, axis=1)),
                                            columns=minority_data.drop(target_variable, axis=1).columns)
            
            majority_normalized = pd.DataFrame(transformer.transform(majority_data.drop(target_variable, axis=1)),
                                               columns=majority_data.drop(target_variable, axis=1).columns)
            
            
            # re entering the target column to make sure it didnt change
            minority_normalized[protected_attribute] = temp_protected_column_minority
            majority_normalized[protected_attribute] = temp_protected_column_majority
            minority_normalized[target_variable] = temp_target_column_minority
            majority_normalized[target_variable] = temp_target_column_majority
            # Combine the normalized data with the majority data
            balanced_data = pd.concat([minority_normalized, majority_data])
            balanced_data = balanced_data.dropna()
            return balanced_data




np.random.seed(123)

'''
 the function will always return majority before minority
 It is a utility function that is used to split the data in our notebook of reweghting and resampling
'''
def get_majority_minority(data, column_to_split, target_variable):
    threshold = 0.5     
    skewness = data[column_to_split].skew()

    # Here we will check for skewness using IQR to measure a heavy tailed distibution
    quantiles = data[column_to_split].quantile([0.25, 0.5, 0.75])
    iqr_condition = quantiles[0.75] - quantiles[0.25] > 2*(quantiles[0.5]-quantiles[0.25])

    jb_test = jarque_bera(data[column_to_split])
    jb_condition = jb_test[0] > jb_test[1]


    if abs(skewness) > threshold and (iqr_condition or jb_condition) :

        # transforming the data into qunatile bins
        bins, split_values = pd.qcut(data[column_to_split], q=3, retbins=True, labels=False, duplicates='drop')

        print(split_values)
        # Print the skewness value and its direction
        if skewness > 0:
            print(f"The {column_to_split} column is right-skewed")
            minority_data = data[data[column_to_split] >= split_values[-2]]
            majority_data = data[data[column_to_split] < split_values[-2]]
            return majority_data, minority_data    


        elif skewness < 0:
            print(f"The {column_to_split} column is left-skewed")
            minority_data = data[data[column_to_split] <= split_values[1]]
            majority_data = data[data[column_to_split] > split_values[1]]
            return majority_data, minority_data

    return data[data[column_to_split] >= np.median(data[column_to_split])] , data[data[column_to_split] < np.median(data[column_to_split])]
        




'''
    Simply calculates spearman correlation for our project
'''
def spearman(data, protected_attribte, target_attribute):
    # Compute Spearman's rank correlation coefficient
    spearman_corr, spearman_p = spearmanr(data[protected_attribte], data[target_attribute])
    print("Spearman's correlation coefficient:", spearman_corr)
    print("p-value:", spearman_p)




'''
    calculates our metric of separation that we proposed at the paper proposal part of
    the course.
'''
def compute_fairness_matrics(data, protected_variable, target_variable):

  
  majority_data, minority_data = get_majority_minority(data, protected_variable, target_variable)
  
  X_train_major, X_test_major, y_train_major, y_test_major, pa_train_major, pa_test_major = train_test_split(majority_data.drop(columns=[protected_variable, target_variable]),
                                                                         majority_data[target_variable], majority_data[protected_variable], test_size=0.2, random_state=42)
  X_train_minor, X_test_minor, y_train_minor, y_test_minor, pa_train_minor, pa_test_minor = train_test_split(minority_data.drop(columns=[protected_variable, target_variable]),
                                                                         minority_data[target_variable], minority_data[protected_variable], test_size=0.2, random_state=42)

  # Fit a logistic regression classifier
  clf = LogisticRegression(random_state=123, max_iter=1200)
  clf.fit(pd.concat([X_train_major, X_train_minor]), pd.concat([y_train_major, y_train_minor]))

  # Define the threshold for classification
  threshold = 0.5


  # Method: Separation
  # Compute the proportion of positive outcomes for each group
  pos_rate_protected = np.sum((clf.predict_proba(X_test_minor)[:, 1] >= threshold) * (y_test_minor == 1)) / np.sum((y_test_minor == 1))
  pos_rate_non_protected = np.sum((clf.predict_proba(X_test_major)[:, 1] >= threshold) * (y_test_major == 1)) / np.sum((y_test_major == 1))
  acc = accuracy_score(pd.concat([y_test_major, y_test_minor]), clf.predict(pd.concat([X_test_major, X_test_minor])))

  return abs(pos_rate_protected - pos_rate_non_protected), acc
 