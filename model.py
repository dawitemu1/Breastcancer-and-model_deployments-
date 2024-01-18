import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# random seed
seed = 42

# Read original dataset
iris_df = pd.read_csv("C:/Users/Dawit Shibabaw/Desktop/Iris_stremalit - Copy/data.csv.csv")
iris_df.sample(frac=1, random_state=seed)

# selecting features and target data
X = iris_df[['radius_mean', 'perimeter_mean', 'area_mean',
       'concave points_mean', 'fractal_dimension_mean', 'radius_se',
       'texture_se', 'area_se', 'concavity_se', 'concave points_se',
       'texture_worst', 'perimeter_worst', 'area_worst', 'compactness_worst']]
y = iris_df[['diagnosis']]

# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y)

# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# train the classifier on the training data
clf.fit(X_train, y_train)

# predict on the test set
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")  # Accuracy: 0.91

# save the model to disk
joblib.dump(clf, "Breastcnacer_rf_model.sav")
