from sklearn.datasets import load_iris
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
os.makedirs('data', exist_ok=True)
iris_df.to_csv('data/iris_data.csv', index=False)
iris_df['target'] = iris.target

x=iris_df.drop('target',axis=1)
y=iris_df['target']

ls=LogisticRegression()

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=42)

ls.fit(train_x,train_y)


pre=ls.predict(test_x)

cm=confusion_matrix(pre,test_y)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
plt.savefig('confusion_matrix_plot.png')