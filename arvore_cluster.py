
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv("features_cluster_name.csv", sep=";")

# %%

features = df.columns.tolist()[:-1]
features

target = 'cluster_name'

# %%

clf = tree.DecisionTreeClassifier()

clf.fit(df[features], df[target])

# %%

plt.figure(dpi=400)
tree.plot_tree(clf,feature_names=features,  
                filled=True)