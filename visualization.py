import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Choose a categorical or continuous variable (e.g., petal length)
variable = "petal length (cm)"

# Create a bar chart (for categorical) or histogram (for continuous)
if pd.api.types.is_categorical_dtype(df[variable]):
    df[variable].value_counts().plot(kind="bar")
else:
    df[variable].plot.hist(bins=10)

plt.xlabel(variable)
plt.ylabel("Frequency")
plt.title(f"Distribution of {variable} in Iris Dataset")
plt.show()

#NAME-SWASTIK ROY CHOUDHURY
#TASK=01_DATA SCIENCE