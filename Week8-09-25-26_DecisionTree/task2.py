import numpy as np
import pandas as pd

def entropy(y):

    if len(y) == 0:
        return 0

    classes, counts = np.unique(y, return_counts=True)

    total = len(y)
    result = 0

    for count in counts:

        p = count / total

        result -= p * np.log2(p)

    return result




def informationgain(parenty, lefty, righty):

    parent_entropy = entropy(parenty)

    total = len(parenty)

    left_weight = len(lefty) / total
    right_weight = len(righty) / total

    child_entropy = (
        left_weight * entropy(lefty)
        + right_weight * entropy(righty)
    )

    gain = parent_entropy - child_entropy

    return gain



def findbestsplit(X, y):

    bestfeature = None
    bestthreshold = None
    bestgain = -1

    # Check every feature
    for feature in X.columns:

        values = np.sort(X[feature].unique())

        # Try every possible threshold
        for i in range(len(values) - 1):

            # Threshold between two consecutive values
            threshold = (values[i] + values[i + 1]) / 2

            leftmask = X[feature] <= threshold
            rightmask = X[feature] > threshold

            lefty = y[leftmask]
            righty = y[rightmask]

            # Ignore empty splits
            if len(lefty) == 0 or len(righty) == 0:
                continue

            gain = informationgain(
                y,
                lefty,
                righty
            )

            # Store the split with maximum gain
            if gain > bestgain:

                bestgain = gain
                bestfeature = feature
                bestthreshold = threshold

    return bestfeature, bestthreshold, bestgain




def majorityclass(y):

    return y.value_counts().idxmax()



def build_tree(X, y, depth=0, max_depth=4, min_samples_split=2):

    # Stop if node is pure
    if len(y.unique()) == 1:

        return {
            "leaf": True,
            "prediction": y.iloc[0]
        }


    # Stop if maximum depth is reached
    if depth >= max_depth:

        return {
            "leaf": True,
            "prediction": majorityclass(y)
        }


    # Stop if there are too few samples
    if len(y) < min_samples_split:

        return {
            "leaf": True,
            "prediction": majorityclass(y)
        }


    # Find the best split
    feature, threshold, gain = findbestsplit(X, y)


    # If no useful split exists
    if feature is None or gain <= 0:

        return {
            "leaf": True,
            "prediction": majorityclass(y)
        }


    # Split the data
    leftmask = X[feature] <= threshold
    rightmask = X[feature] > threshold

    X_left = X[leftmask]
    X_right = X[rightmask]

    y_left = y[leftmask]
    y_right = y[rightmask]


    # Build left subtree
    lefttree = build_tree(
        X_left,
        y_left,
        depth + 1,
        max_depth,
        min_samples_split
    )


    # Build right subtree
    righttree = build_tree(
        X_right,
        y_right,
        depth + 1,
        max_depth,
        min_samples_split
    )


    # Store this decision node
    return {
        "leaf": False,
        "feature": feature,
        "threshold": threshold,
        "gain": gain,
        "left": lefttree,
        "right": righttree
    }




def predict_one(row, tree):

    # If leaf is reached, return its prediction
    if tree["leaf"]:

        return tree["prediction"]


    feature = tree["feature"]
    threshold = tree["threshold"]


    # Follow left or right branch
    if row[feature] <= threshold:

        return predict_one(
            row,
            tree["left"]
        )

    else:

        return predict_one(
            row,
            tree["right"]
        )



def predict(X, tree):

    predictions = []

    for _, row in X.iterrows():

        predictions.append(
            predict_one(row, tree)
        )

    return predictions




def print_tree(tree, depth=0):

    spaces = "    " * depth

    if tree["leaf"]:

        print(
            spaces,
            "Leaf ->",
            tree["prediction"]
        )

        return


    print(
        spaces,
        tree["feature"],
        "<=",
        tree["threshold"]
    )

    print(spaces, "Left:")
    print_tree(tree["left"], depth + 1)

    print(spaces, "Right:")
    print_tree(tree["right"], depth + 1)




df = pd.read_csv("AAPL.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Remove ** if they are present in the column names
df.columns = df.columns.str.replace("*", "", regex=False)


# Convert Date to date format 
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df = df.reset_index(drop=True)


# Return = (Today's price - Yesterday's price) / Yesterday's price

df["Return"] = df["Adj_Close"].pct_change()




df["Price_Range"] = (df["Adj_High"] - df["Adj_Low"])






# Target = 1 -> next day's price is higher
# Target = 0 -> next day's price is not higher

df["Target"] = ( df["Adj_Close"].shift(-1) > df["Adj_Close"]).astype(int)



df = df.dropna()
df = df.reset_index(drop=True)


X = df[
    [
        "Adj_Close",
        "Adj_Volume",
        "Return",
        "Price_Range"
    ]
]

y = df["Target"]

split = int(0.67 * len(df))


X_train = X.iloc[:split]
y_train = y.iloc[:split]

X_test = X.iloc[split:]
y_test = y.iloc[split:]



print("Total rows:", len(df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nFeatures used:")
print(X.columns.tolist())





tree = build_tree(
    X_train,
    y_train,
    max_depth=4,
    min_samples_split=2
)






print_tree(tree)



predictions = predict(
    X_test,
    tree
)



correct = 0

for actual, predicted in zip(y_test, predictions):

    if actual == predicted:
        correct += 1


total = len(y_test)

accuracy = (correct / total) * 100



print("Correct predictions:", correct)
print("Total test samples:", total)

print(
    "Directional Prediction Accuracy:",
    round(accuracy, 2),
    "%"
)



results = pd.DataFrame({

    "Date": df.iloc[split:]["Date"].values,

    "Actual": y_test.values,

    "Predicted": predictions

})
print("Actual vs preidcted")
print(results.to_string(index=False))