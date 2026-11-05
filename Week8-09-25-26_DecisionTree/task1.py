import numpy as np
import pandas as pd

def entropy(y):
    if(len(y)==0):
        return 0
    
    #find different classes how many times a class occurs
    
    uniqueclasses,counts=np.unique(y,return_counts=True)
    total=len(y)
    
    entropyvalue=0
    
    for count in counts:
        probability=count/total
        
        entropyvalue-= (probability*np.log2(probability))
        
    return entropyvalue

def informationgain(parenty,lefty,righty):
    parent_entropy = entropy(parenty)

    # Number of samples in parent.
    total_samples = len(parenty)

    # Number of samples in each child.
    left_samples = len(lefty)
    right_samples = len(righty)

    # Calculate weighted entropy of the children.
    weighted_child_entropy = (
        (left_samples / total_samples) * entropy(lefty) +
        (right_samples / total_samples) * entropy(righty)
    )

    # Information Gain.
    gain = parent_entropy - weighted_child_entropy

    return gain

def findbestsplit(X,y):
    bestfeature=None
    bestthreshold=None
    bestgain=-1
    bestiscategorical=False
    
    for feature in X.columns:
        column=X[feature]
        
        #Categorical feature
        
        if column.dtype=="object":
            
            uniquevalues=column.unique()
            
            for value in uniquevalues:
                leftmask=column==value
                rightmask=column!=value
                
                #target values for both the groups
                
                lefty=y[leftmask]
                righty=y[rightmask]
                
                if len(lefty) == 0 or len(righty) == 0:
                    continue

                # Calculate Information Gain.
                gain = informationgain(y,lefty,righty)
                
                if gain > bestgain:

                    bestgain = gain
                    bestfeature = feature
                    bestthreshold = value
                    bestiscategorical = True
        else:
            values=np.sort(column.unique())
            
            if(len(values)<2):
                continue
            
            for i in range(len(values)-1):
                
                threshold = (values[i] + values[i + 1]) / 2


                # Left:
                # feature <= threshold
                leftmask = column <= threshold

                # Right:
                # feature > threshold
                rightmask = column > threshold


                # Get target values in both groups.
                lefty = y[leftmask]
                righty = y[rightmask]


                # Ignore invalid splits.
                if len(lefty) == 0 or len(righty) == 0:
                    continue


                # Calculate Information Gain.
                gain = informationgain(y,lefty,righty)


                # Check whether this is the best
                # split found so far.
                if gain > bestgain:

                    bestgain = gain
                    bestfeature = feature
                    bestthreshold = threshold
                    bestiscategorical = False
                    
    
    return (
        bestfeature,
        bestthreshold,
        bestgain,
        bestiscategorical
    )

def majorityclass(y):
    # value_counts() counts every class.
    # idxmax() returns the class with the highest count.
    return y.value_counts().idxmax()


def build_tree(
    X,
    y,
    depth=0,
    max_depth=4,
    min_samples_split=2
):
    

    if len(y.unique()) == 1:

        return {
            "leaf": True,
            "prediction": y.iloc[0]
        }


    # Stop if maximum depth has been reached.
   

    if depth >= max_depth:

        # Predict the majority class at this node.
        return {
            "leaf": True,
            "prediction": majorityclass(y)
        }


    # if there are too few samples to split.

    if len(y) < min_samples_split:

        # Predict the majority class.
        return {
            "leaf": True,   
            "prediction": majorityclass(y)
        }

    # find best split
    (bestfeature, bestthreshold,bestgain,bestiscategorical) = findbestsplit(X, y)



    # IF NO USEFUL SPLIT EXISTS
  

    if bestfeature is None or bestgain <= 0:
        # Make this node a leaf.
        return {
            "leaf": True,
            "prediction": majorityclass(y)
        }


    # CREATE LEFT AND RIGHT DATASETS
  

    if bestiscategorical:
        # For categorical features:
        
        # LEFT:
        # feature == category
        
        # RIGHT:
        # feature != category

        left_mask = (X[bestfeature] == bestthreshold)

        right_mask = (X[bestfeature] != bestthreshold)

    else:

        # For numerical features:
        


        left_mask = (X[bestfeature] <= bestthreshold)

        right_mask = (X[bestfeature] > bestthreshold)


    # SPLIT THE DATA
   

    X_left = X[left_mask]
    X_right = X[right_mask]

    y_left = y[left_mask]
    y_right = y[right_mask]


    # RECURSIVELY BUILD LEFT SUBTREE
 
    left_subtree = build_tree(
        X_left,
        y_left,
        depth=depth + 1,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )


    # RECURSIVELY BUILD RIGHT SUBTREE
  

    right_subtree = build_tree(
        X_right,
        y_right,
        depth=depth + 1,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )


    # RETURN CURRENT INTERNAL NODE


    return {
        "leaf": False,
        "feature": bestfeature,
        "threshold": bestthreshold,

        # True → categorical
        # False → numerical
        "categorical": bestiscategorical,

        # Store the Information Gain.
        "information_gain": bestgain,

        "left": left_subtree,
        "right": right_subtree
    }



def print_tree(tree, depth=0, data=None):

    space = "    " * depth

    if tree["leaf"]:

        print(space + "Leaf")
        print(space + "Prediction:", tree["prediction"])

        if data is not None:
            print(space + "Data points:")

            for i in range(len(data)):
                print(space, data.iloc[i].to_dict())

            print(space + "Class counts:")
            print(space, data["Drug"].value_counts().to_dict())

        print()
        return

    print(space + tree["feature"], end=" ")

    if tree["categorical"]:
        print("==", tree["threshold"])
    else:
        print("<=", tree["threshold"])

    if tree["categorical"]:
        leftdata = data[data[tree["feature"]] == tree["threshold"]]
        rightdata = data[data[tree["feature"]] != tree["threshold"]]
    else:
        leftdata = data[data[tree["feature"]] <= tree["threshold"]]
        rightdata = data[data[tree["feature"]] > tree["threshold"]]

    print(space + "Left:")
    print_tree(tree["left"], depth + 1, leftdata)

    print(space + "Right:")
    print_tree(tree["right"], depth + 1, rightdata)

# PREDICT ONE PATIENT

def predict_one(row, tree):

    # IF WE REACHED A LEAF

    if tree["leaf"]:

        # Return the drug stored in the leaf.
        return tree["prediction"]


    # Get the feature used at this node.
    feature = tree["feature"]

    # Get the threshold/category.
    threshold = tree["threshold"]


    # CATEGORICAL SPLIT


    if tree["categorical"]:

        # Example:
        #
        # BP == HIGH
        #
        # If true → left
        # Otherwise → right

        if row[feature] == threshold:

            return predict_one(row,tree["left"])

        else:

            return predict_one(row,tree["right"])


    # NUMERICAL SPLIT

    else:

        

        if row[feature] <= threshold:

            return predict_one(row,tree["left"])

        else:

            return predict_one(row,tree["right"])

# PREDICT ALL PATIENTS


def predict(X, tree):
 
    # Store all predictions.
    predictions = []

    # Go through every patient.
    for _, row in X.iterrows():

        # Predict this patient.
        prediction = predict_one(row,tree)

        # Add prediction to the list.
        predictions.append(prediction)

    return predictions


# LOAD DATASET

df = pd.read_csv("drug200.csv")






# X contains the input features.
X = df.drop("Drug", axis=1)

y = df["Drug"]


# Calculate the number of training samples.
split_index = int(0.67 * len(df))

X_train = X.iloc[:split_index]

y_train = y.iloc[:split_index]




X_test = X.iloc[split_index:]

y_test = y.iloc[split_index:]


print("\nTraining samples:", len(X_train))
print("Verification samples:", len(X_test))



tree = build_tree( X_train,y_train,max_depth=10,min_samples_split=2)


print("\nDecision Tree:")
print_tree(tree, data=df.iloc[:split_index])




predictions = predict(X_test, tree)





correct = 0

for actual, predicted in zip(y_test,predictions):

    if actual == predicted:
        correct += 1


# Total number of verification samples.
total = len(y_test)


# Calculate accuracy percentage.
accuracy = (correct / total) * 100




print("\nCorrect predictions:", correct)
print("Total verification samples:", total)

print(
    f"Accuracy: {accuracy:.2f}%"
)