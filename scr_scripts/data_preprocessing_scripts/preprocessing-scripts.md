# Encoding and Scaling Automation

## Overview
This repository automates the preprocessing steps for **feature scaling** and **categorical encoding**, reducing the need for repetitive manual preprocessing. The automation ensures flexibility by allowing users to specify both the dataset format (**NumPy or Pandas**) and the exact features to preprocess. 

## Features
The encoding and scaling scripts come pre-equipped with multiple preprocessing techniques, each categorized into their respective modules. 

### Feature Scaling Automation
The **scaling script** includes various scalers to standardize or normalize numerical data. These have, but are not limited to:

- **Normalizer**: Scales input features to unit norm (L1 or L2 normalization).
- **StandardScaler**: Centers the data by removing the mean and scaling it to unit variance.
- **MinMaxScaler**: Scales each feature to a fixed range (default is [0,1]).
- **MaxAbsScaler**: Scales features by dividing by the maximum absolute value.

### Encoding Automation
The **encoding script** manages categorical variables by converting them into numerical representations. The script supports, but is not limited to:

- **OneHotEncoder**: Converts categorical variables into binary indicator variables.
- **OrdinalEncoder**: Encodes categories with integer values in a specified order.
- **Target Encoding**: Replaces categorical variables with the mean of the target variable.
- **LabelBinarizer**: Converts a categorical feature into a binary matrix.

## How It Works
1. **Select Preprocessing Type**: Choose whether to scale or encode.
2. **Specify Output Format**: Decide if you want the transformed data in **NumPy** or **Pandas** format.
3. **Choose Features for Processing**:
   - If using **NumPy**, provide feature indices.
   - If using **Pandas**, specify column names.
4. **Apply Preprocessing**: The script will apply the chosen transformations only to the selected features while leaving the rest of the dataset unchanged.

## Example Usage
### Scaling
```python
from scaling_script import ScalingPreprocessor

scaler = ScalingPreprocessor(method='StandardScaler', output_format='pandas')
df_scaled = scaler.fit_transform(df, columns=['feature1', 'feature2'])
```

### Encoding
```python
from encoding_script import EncodingPreprocessor

encoder = EncodingPreprocessor(method='OneHotEncoder', output_format='numpy')
encoded_array = encoder.fit_transform(data, indices=[0, 2])
```

## Why Use This?
- Eliminates the need for writing redundant scaling and encoding code.
- Ensures consistency across different datasets and models.
- Allows users to apply transformations selectively, avoiding unnecessary preprocessing.

## Contributing
Feel free to fork this repository and improve the scripts. Contributions are welcome!

---

Happy preprocessing! 🚀

