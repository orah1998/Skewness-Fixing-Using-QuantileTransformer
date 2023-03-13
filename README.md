# Skewness-Fixing-Using-QuantileTransformer


Our project is quite simple to use.


<b> First and foremost </b> we will state that the protected attribute that choose to handle is a continuous one, as well as a target variable that is a binary type.
This is because we aimed to address skewness on the protected variable which forced it to be continuous.
Target was decided to be binary as other forms of it were quite complex to calculate fairness on, as well as our formula of fairness which was prevented.

In order to run our code on your dataset, all you need to do is simply:

```python

#### Bin Splitting - QuantileTransformer ####

fixed_data = bin_splitting(data, protected_attribute, target_variable)

X = fixed_data.drop(columns=[target_variable, protected_attribute])
y = fixed_data[target_variable]
Z = fixed_data[protected_attribute]
```

Quite simple right?

We also included a utility function of spearman correlation that consists of checking the correlations of the target and the protected variables.

Another feature is a a test for measuring fairness. the test function is called `compute_fairness_matrics` and can be used as following:
```python

fixed_data = bin_splitting(data, protected_attribute, target_variable)

X = fixed_data.drop(columns=[target_variable, protected_attribute])
y = fixed_data[target_variable]
Z = fixed_data[protected_attribute]


compute_fairness_matrics(fixed_data, protected_attribute, target_variable)
# the function return the difference in the separation value of the two groups of the protected attribute that was splitted using quantiles, as well as the overall<br>
accuracy of the model with the given data
```
