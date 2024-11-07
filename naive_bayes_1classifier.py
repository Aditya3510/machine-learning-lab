
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class NaiveBayes:
    """
    Naive Bayes classifier for continuous and discrete features using pandas.
    """
    def __init__(self, continuous=None):
        """
        :param continuous: List containing a bool for each feature column to be analyzed.
                           True if the feature column contains a continuous feature, False if discrete.
        """
        self.continuous = continuous or []
        self.priors = {}
        self.likelihoods = defaultdict(dict)
        self.classes = None

    def fit(self, data: pd.DataFrame, target_name: str):
        """
        Fitting the training data by saving all relevant conditional probabilities for discrete or continuous features.
        :param data: pd.DataFrame containing training data (including the label column)
        :param target_name: Name of the label column in data.
        """
        self.classes = data[target_name].unique()
        features = data.drop(columns=[target_name])
        # Calculate priors for each class
        for cls in self.classes:
            cls_data = data[data[target_name] == cls]
            self.priors[cls] = len(cls_data) / len(data)
            # Calculate conditional probabilities
            for i, col in enumerate(features.columns):
                if self.continuous[i]:
                    # Store mean and std for continuous features
                    mean, std = cls_data[col].mean(), cls_data[col].std()
                    self.likelihoods[cls][col] = (mean, std)
                else:
                    # Store probability distribution for discrete features
                    value_counts = cls_data[col].value_counts(normalize=True).to_dict()
                    self.likelihoods[cls][col] = value_counts

    def _calculate_likelihood(self, value, mean, std):
        """
        Calculates the Gaussian probability density for continuous features.
        """
        if std == 0:  # Avoid division by zero
            return 1 if value == mean else 0
        exponent = math.exp(-((value - mean) ** 2) / (2 * std ** 2))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

    def predict_probability(self, data: pd.DataFrame):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted.
        :return: pd.DataFrame containing probabilities for all categories as well as the classification result.
        """
        results = []
        for _, row in data.iterrows():
            row_probs = {}
            for cls in self.classes:
                total_log_prob = math.log(self.priors[cls])
                for i, col in enumerate(data.columns):
                    if self.continuous[i]:
                        mean, std = self.likelihoods[cls][col]
                        prob = self._calculate_likelihood(row[col], mean, std)
                    else:
                        prob = self.likelihoods[cls][col].get(row[col], 1e-6)  # Smoothing for unseen values
                    total_log_prob += math.log(prob)
                row_probs[cls] = total_log_prob
            results.append(row_probs)
        probs_df = pd.DataFrame(results)
        probs_df['Prediction'] = probs_df.idxmax(axis=1)
        return probs_df

    def evaluate_on_data(self, data: pd.DataFrame, test_labels):
        """
        Predicts a test DataFrame and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data.
        :param test_labels: True labels for the test data.
        :return: Tuple of overall accuracy and confusion matrix values.
        """
        predictions = self.predict_probability(data)['Prediction']
        accuracy = (predictions == test_labels).mean()
        # Create confusion matrix
        confusion_matrix = pd.crosstab(test_labels, predictions, rownames=['Actual'], colnames=['Predicted'])
        return accuracy, confusion_matrix
