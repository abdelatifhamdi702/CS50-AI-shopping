import csv
import sys
import calendar
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.4  # Define the test size for train-test split

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    predictions = train_and_predict(X_train, y_train, X_test)

    # Evaluate the model and print results
    sensitivity, specificity, f1_score = evaluate(y_test, predictions)
    correct_count = sum(1 for true_label, predicted_label in zip(y_test, predictions) if true_label == predicted_label)
    incorrect_count = len(predictions) - correct_count
    print(f"Correct: {correct_count}")  # Print the number of correct predictions
    print(f"Incorrect: {incorrect_count}")  # Print the number of incorrect predictions
    print(f"True Positive Rate (Sensitivity): {100 * sensitivity:.2f}%")  # Print sensitivity as a percentage
    print(f"True Negative Rate (Specificity): {100 * specificity:.2f}%")  # Print specificity as a percentage
    print(f"F1 Score: {f1_score:.2f}")  # Print F1 score

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).
    """
    # Create a mapping of month abbreviations to numerical values
    months = {month: index - 1 for index, month in enumerate(calendar.month_abbr) if index}
    months['June'] = months.pop('Jun')  # Handle the abbreviation for June

    evidence = []  # List to store the evidence data
    labels = []    # List to store the labels

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)  # Create a CSV reader
        for row in reader:
            # Parse each row and convert it into a list of features
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                months[row['Month']],  # Map month abbreviation to numerical value
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,  # Convert VisitorType to binary
                1 if row['Weekend'] == 'TRUE' else 0  # Convert Weekend to binary
            ])
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)  # Convert Revenue to binary

    return evidence, labels

def train_and_predict(X_train, y_train, X_test):
    """
    Train a nearest neighbor classifier (k=1) on the training data and make predictions on the test data.
    """
    predictions = []

    for test_point in X_test:
        nearest_neighbor = find_nearest_neighbor(test_point, X_train, y_train)
        predictions.append(nearest_neighbor)

    return predictions

def find_nearest_neighbor(test_point, X_train, y_train):
    """
    Find the nearest neighbor in X_train to the test_point and return its label.
    """
    min_distance = float('inf')
    nearest_neighbor = None

    for i, train_point in enumerate(X_train):
        distance = euclidean_distance(test_point, train_point)
        if distance < min_distance:
            min_distance = distance
            nearest_neighbor = y_train[i]

    return nearest_neighbor

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points represented as lists.
    """
    squared_distance = sum((x - y) ** 2 for x, y in zip(point1, point2))
    return squared_distance ** 0.5

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity, f1_score).
    """
    true_positives = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 1)
    true_negatives = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 0)
    false_positives = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 1)
    false_negatives = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 0)
    
    sensitivity = true_positives / (true_positives + false_negatives)  # Calculate sensitivity
    specificity = true_negatives / (true_negatives + false_positives)  # Calculate specificity
    precision = true_positives / (true_positives + false_positives)  # Calculate precision
    recall = sensitivity  # Recall is the same as sensitivity
    f1_score = 2 * (precision * recall) / (precision + recall)  # Calculate F1 score

    return sensitivity, specificity, f1_score

if __name__ == "__main__":
    main()
