import pandas as pd

# Load data
iris_data = pd.read_csv("Iris.csv")
iris_data.head()

# Check nulls
iris_data.isnull().sum()

# Splitting the data
X = iris_data.drop("Species", axis=1)
y = iris_data["Species"]

# Train-test split (done ONCE)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# Metrics
from sklearn.metrics import precision_score, accuracy_score, recall_score

# -------------------------------
# 1. KNN (Pipeline + GridSearch)
# -------------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

pipeline_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

param_knn = {
    "knn__n_neighbors": [3, 5, 7, 9]
}

grid_knn = GridSearchCV(pipeline_knn, param_knn, cv=5)
grid_knn.fit(X_train, y_train)

y_pred_knn = grid_knn.predict(X_test)

print("KNN Results")
print("Recall Score:", recall_score(y_test, y_pred_knn, average='macro'))
print("Accuracy Score:", accuracy_score(y_test, y_pred_knn))
print("Precision Score:", precision_score(y_test, y_pred_knn, average='macro'))
print("Best Params:", grid_knn.best_params_)
print()

# -------------------------------
# 2. Logistic Regression (Pipeline + GridSearch)
# -------------------------------
from sklearn.linear_model import LogisticRegression

pipeline_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000))
])

param_lr = {
    "lr__C": [0.1, 1, 10]
}

grid_lr = GridSearchCV(pipeline_lr, param_lr, cv=5)
grid_lr.fit(X_train, y_train)

y_pred_lr = grid_lr.predict(X_test)

print("Logistic Regression Results")
print("Recall Score:", recall_score(y_test, y_pred_lr, average='macro'))
print("Accuracy Score:", accuracy_score(y_test, y_pred_lr))
print("Precision Score:", precision_score(y_test, y_pred_lr, average='macro'))
print("Best Params:", grid_lr.best_params_)
print()

# -------------------------------
# 3. Naive Bayes (Pipeline)
# -------------------------------
from sklearn.naive_bayes import GaussianNB

pipeline_nb = Pipeline([
    ("scaler", StandardScaler()),
    ("nb", GaussianNB())
])

pipeline_nb.fit(X_train, y_train)

y_pred_nb = pipeline_nb.predict(X_test)

print("Naive Bayes Results")
print("Recall Score:", recall_score(y_test, y_pred_nb, average='macro'))
print("Accuracy Score:", accuracy_score(y_test, y_pred_nb))
print("Precision Score:", precision_score(y_test, y_pred_nb, average='macro'))
