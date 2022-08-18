import pandas as pd

from sklearn.datasets import load_iris
# df = pd.read_csv("./iris.csv")
data = load_iris()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['species'] = pd.Categorical.from_codes(data.target, data.target_names)


def slice_iris(df, feature):
    """ Function for calculating descriptive stats on slices of the Iris dataset."""
    for cls in df["species"].unique():
        df_temp = df[df["species"] == cls]
        mean = df_temp[feature].mean()
        stddev = df_temp[feature].std()
        print(f"Class: {cls}")
        print(f"{feature} mean: {mean:.4f}")
        print(f"{feature} stddev: {stddev:.4f}")
    print()

'''
Features:
`sepal length (cm)`
`sepal width (cm)`
`petal length (cm)`
`petal width (cm)`
'''
slice_iris(df, "sepal length (cm)")
slice_iris(df, "sepal width (cm)")
slice_iris(df, "petal length (cm)")
slice_iris(df, "petal width (cm)")