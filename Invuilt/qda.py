from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

class_names = iris.target_names

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create the QDA model
model = QuadraticDiscriminantAnalysis(store_covariance=True)

# Train the model
model.fit(X_train, y_train)

# Predict class labels
y_pred = model.predict(X_test)

# Get probability of each class
y_prob = model.predict_proba(X_test)

# Calculate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Display predicted probabilities
print("\nPredicted probabilities:")
print(y_prob[:5])

# Display learned parameters
print("\nClass Priors:")
print(model.priors_)

print("\nClass Means:")
print(model.means_)

print("\nClass Covariance Matrices:")
for i in range(len(class_names)):
    print("\n", class_names[i])
    print(model.covariance_[i])
