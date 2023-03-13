# Skewness-Fixing-Using-QuantileTransformer


Our project is quite simple to use.


<b> First and foremost </b> we will state that the protected attribute that choose to handle is a continuous one. 
In order to run our code on your dataset, all you need to do is simply:

```python

#### Bin Splitting - QuantileTransformer ####

fixed_data = bin_splitting(data, protected_attribute, target_variable)

X = fixed_data.drop(columns=[target_variable, protected_attribute])
y = fixed_data[target_variable]
Z = fixed_data[protected_attribute]
```

Quite simple right?