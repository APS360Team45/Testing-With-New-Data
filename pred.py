from pprint import pprint

# Define the labels for each fruit and numerical label
fruit_labels = {0: 'banana', 1: 'tomato', 2: 'mango'}
num_labels = {0: 'zero', 1: 'one', 2: 'two', 3: 'three'}

# Initialize counters for each fruit and numerical label
fruit_counts = {label: 0 for label in fruit_labels.values()}
num_counts = {label: 0 for label in num_labels.values()}
correct_counts = {f"{fruit}-{num}": 0 for fruit in fruit_labels.values() for num in num_labels.values()}

# Parse the predictions.txt file and update the counters
with open('predictions.txt', 'r') as f:
    for line in f:
        # Parse the prediction, actual label, and file path from the line
        prediction, actual, file_path = line.strip().split('|')
        prediction = prediction.strip().split()[-1]
        actual = actual.strip().split()[-1]
        
        # Check that the actual label is valid
        if int(actual) not in fruit_labels:
            continue
        
        # Update the fruit and numerical label counters
        fruit_counts[fruit_labels[int(actual)]] += 1
        num_counts[num_labels[int(actual)]] += 1
        
        # Update the correct prediction counter if the prediction was correct
        if prediction == actual:
            correct_counts[f"{fruit_labels[int(actual)]}-{num_labels[int(actual)]}"] += 1

# Compute the accuracy for each fruit and numerical label
fruit_accuracy = {fruit: correct_counts[f"{fruit}-{num}"] / fruit_counts[fruit] * 100 if fruit_counts[fruit] != 0 else 0 for fruit in fruit_labels.values() for num in num_labels.values() if f"{fruit}-{num}" in correct_counts}
num_accuracy = {num: correct_counts[f"{fruit}-{num}"] / num_counts[num] * 100 if num_counts[num] != 0 else 0 for num in num_labels.values() for fruit in fruit_labels.values() if f"{fruit}-{num}" in correct_counts}

# Pretty print the results
results = {"Fruit counts": fruit_counts, "Numerical label counts": num_counts, "Correct predictions": correct_counts, "Fruit accuracy": {fruit: f"{accuracy:.1f}%" for fruit, accuracy in fruit_accuracy.items()}, "Numerical label accuracy": {num: f"{accuracy:.1f}%" for num, accuracy in num_accuracy.items()}}

# Print each key-value pair on a separate line
for key, value in results.items():
    print(key + ":")
    for subkey, subvalue in value.items():
        print(f"\t{subkey}: {subvalue}")