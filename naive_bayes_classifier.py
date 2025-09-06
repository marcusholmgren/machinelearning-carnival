"""
A simple text classification library using a Naive Bayes classifier.

This module provides a `Classifier` base class and a `NaiveBayes` implementation
for categorizing text documents. It is designed to be straightforward and
demonstrates key concepts like feature extraction, probability smoothing, and
log-space calculations for numerical stability.

Key Concepts:
- Naive Bayes: A probabilistic classifier based on applying Bayes' theorem with a
  "naive" assumption of conditional independence between every pair of features.
- Bag-of-Words: A simple text representation model where a text is represented
  as an unordered collection (or "bag") of its words, disregarding grammar and
  word order but keeping multiplicity. This implementation uses a binary version
  (set-of-words) where only the presence or absence of a word matters.
- Smoothing: A technique (here, Bayesian averaging) used to handle the "zero-
  frequency problem," where a feature encountered during classification was not
  seen in a particular category during training.
"""

import re
import math
from typing import Dict, List, Callable, Any, Set

# A simple set of English stopwords. In a real-world application,
# you would use a more comprehensive list from a library like NLTK or SpaCy.
DEFAULT_STOPWORDS: Set[str] = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
    'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was',
    'were', 'will', 'with'
}

def extract_features(
    doc: str,
    stopwords: Set[str] = DEFAULT_STOPWORDS
) -> Set[str]:
    """
    Extracts unique words from a document to serve as features.

    This function tokenizes a document, converts words to lowercase, filters
    out stopwords and very short/long words, and returns a unique set of words.

    Args:
        doc: The text document (a string) to process.
        stopwords: A set of words to ignore during feature extraction.

    Returns:
        A set of unique words (features) from the document.
    """
    # Use regex to split the document by any non-alphanumeric character.
    splitter = re.compile(r'\W+')

    # Process words: lowercase, filter by length, and exclude stopwords.
    words = {
        s.lower() for s in splitter.split(doc)
        if 2 < len(s) < 20 and s.lower() not in stopwords
    }
    return words


class Classifier:
    """
    A base class for a text classifier.

    This class manages the counting of features per category and documents
    per category, providing the foundational data structures and methods
    for probabilistic classification.

    Attributes:
        feature_counts (Dict[str, Dict[str, int]]): Stores counts of each
            feature within each category. Format: {feature: {category: count}}.
        category_counts (Dict[str, int]): Stores the number of documents
            trained for each category. Format: {category: count}.
        thresholds (Dict[str, float]): Stores classification thresholds for each
            category.
        get_features (Callable[[Any], Set[str]]): A function to extract features
            from an item.
    """

    def __init__(self, get_features: Callable[[Any], Set[str]]):
        """Initializes the classifier with a feature extraction function."""
        # Counts of feature/category combinations, e.g., {'word': {'good': 4, 'bad': 1}}
        self.feature_counts: Dict[str, Dict[str, int]] = {}
        # Counts of documents in each category, e.g., {'good': 10, 'bad': 8}
        self.category_counts: Dict[str, int] = {}
        # Thresholds for classification confidence
        self.thresholds: Dict[str, float] = {}
        self.get_features = get_features

    def _increment_feature_count(self, feature: str, category: str) -> None:
        """Increases the count of a feature for a given category."""
        self.feature_counts.setdefault(feature, {})
        self.feature_counts[feature].setdefault(category, 0)
        self.feature_counts[feature][category] += 1

    def _increment_category_count(self, category: str) -> None:
        """Increases the count of a given category."""
        self.category_counts.setdefault(category, 0)
        self.category_counts[category] += 1

    def get_feature_count(self, feature: str, category: str) -> int:
        """Returns the number of times a feature has appeared in a category."""
        return self.feature_counts.get(feature, {}).get(category, 0)

    def get_category_count(self, category: str) -> int:
        """Returns the total number of documents in a category."""
        return self.category_counts.get(category, 0)

    def get_total_count(self) -> int:
        """Returns the total number of documents trained."""
        return sum(self.category_counts.values())

    def get_categories(self) -> List[str]:
        """Returns a list of all trained categories."""
        return list(self.category_counts.keys())

    def train(self, item: Any, category: str) -> None:
        """Trains the classifier on an item and its category."""
        features = self.get_features(item)
        for f in features:
            self._increment_feature_count(f, category)
        self._increment_category_count(category)

    def feature_prob(self, feature: str, category: str) -> float:
        """
        Calculates P(Feature | Category), the conditional probability of a
        feature given a category.
        """
        cat_count = self.get_category_count(category)
        if cat_count == 0:
            return 0.0
        return self.get_feature_count(feature, category) / cat_count

    def weighted_prob(
        self,
        feature: str,
        category: str,
        weight: float = 1.0,
        assumed_prob: float = 0.5
    ) -> float:
        """
        Calculates a smoothed probability using a weighted average.

        This prevents probabilities from becoming zero for features not seen
        during training in a particular category (the zero-frequency problem).

        Args:
            feature: The feature to calculate the probability for.
            category: The category in the context of which to calculate.
            weight: The strength of the assumed probability. A higher weight
                    means we rely more on the prior assumption.
            assumed_prob: The prior probability to assume for the feature.
        """
        # Calculate the basic P(Feature | Category)
        basic_prob = self.feature_prob(feature, category)

        # Count the number of times this feature has appeared in all categories
        totals = sum(self.get_feature_count(feature, c) for c in self.get_categories())

        # Calculate the weighted average, which pulls the basic probability
        # towards the assumed probability, moderated by the total feature count.
        return ((weight * assumed_prob) + (totals * basic_prob)) / (weight + totals)

    def set_threshold(self, category: str, t: float) -> None:
        """Sets the classification threshold for a category."""
        self.thresholds[category] = t

    def get_threshold(self, category: str) -> float:
        """Gets the classification threshold for a category."""
        return self.thresholds.get(category, 1.0)


class NaiveBayes(Classifier):
    """
    A Naive Bayes classifier implementation.

    This class extends the base Classifier to implement the Naive Bayes algorithm.
    It uses log probabilities to prevent numerical underflow and provides a
    robust classification method.
    """

    def _doc_log_prob(self, item: Any, category: str) -> float:
        """
        Calculates the log probability of a document given a category.

        Instead of multiplying probabilities (which can cause underflow), we
        sum their logarithms. This is numerically stable.
        log(P(Doc | Cat)) = sum(log(P(Feature_i | Cat)))
        """
        features = self.get_features(item)
        log_p = 0.0
        for f in features:
            # Use the smoothed probability to avoid log(0)
            prob = self.weighted_prob(f, category)
            if prob > 0:  # Ensure we don't take log of zero
                log_p += math.log(prob)
        return log_p

    def _full_log_prob(self, item: Any, category:str) -> float:
        """
        Calculates the full log probability of a category given a document.
        log(P(Cat | Doc)) is proportional to log(P(Doc | Cat)) + log(P(Cat))
        """
        # Calculate P(Cat), the prior probability of the category
        category_prob = self.get_category_count(category) / self.get_total_count()
        if category_prob == 0:
            return -float('inf') # Log of zero is negative infinity

        # Combine with the document probability
        return self._doc_log_prob(item, category) + math.log(category_prob)

    def classify(self, item: Any, default: Any = None) -> Any:
        """
        Classifies an item into one of the trained categories.

        It calculates the probability for each category and returns the one with
        the highest score, provided it exceeds a confidence threshold compared
        to the next best category.

        Args:
            item: The item to be classified.
            default: The category to return if classification is uncertain.

        Returns:
            The name of the best-matching category or the default value.
        """
        log_probs: Dict[str, float] = {}
        best_category: Any = default
        max_log_prob = -float('inf')

        # Find the category with the highest log probability
        for cat in self.get_categories():
            log_probs[cat] = self._full_log_prob(item, cat)
            if log_probs[cat] > max_log_prob:
                max_log_prob = log_probs[cat]
                best_category = cat

        if best_category is None:
            return default

        # Confidence thresholding: ensure the best probability is significantly
        # better than the runner-up. We do this in log space to stay consistent.
        # Original logic: P_best < P_next * Threshold
        # Log space logic: log(P_best) < log(P_next) + log(Threshold)
        log_threshold = math.log(self.get_threshold(best_category))
        for cat in self.get_categories():
            if cat == best_category:
                continue
            if log_probs[cat] + log_threshold > max_log_prob:
                return default  # The result is not confident enough

        return best_category


if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Create a classifier instance with our feature extractor
    cl = NaiveBayes(extract_features)

    # 2. Train the classifier with sample data
    print("--- Training Classifier ---")
    cl.train('A great and wonderful victory', 'good')
    cl.train('This was a brilliant and fantastic movie', 'good')
    cl.train('I really disliked this terrible and awful film', 'bad')
    cl.train('What a waste of time, a truly dreadful picture', 'bad')
    print("Training complete.")
    print(f"Categories: {cl.get_categories()}")
    print(f"Total documents trained: {cl.get_total_count()}\n")

    # 3. Test the classifier on new documents
    print("--- Classifying New Documents ---")
    doc1 = "This was a wonderful film with brilliant acting"
    doc2 = "I thought this movie was a dreadful waste of my time"
    doc3 = "The acting was neither great nor terrible" # Ambiguous case

    print(f"Document 1: '{doc1}'")
    print(f"  -> Predicted Category: {cl.classify(doc1)}\n")

    print(f"Document 2: '{doc2}'")
    print(f"  -> Predicted Category: {cl.classify(doc2)}\n")

    print(f"Document 3: '{doc3}'")
    print(f"  -> Predicted Category: {cl.classify(doc3, default='unknown')}\n")

    # 4. Demonstrate thresholding for higher confidence
    print("--- Demonstrating Thresholding ---")
    # Set a threshold for 'good' category: its probability must be at least
    # 1.5 times the probability of the next best category.
    cl.set_threshold('good', 1.5)
    print("Threshold for 'good' category set to 1.5")

    print(f"Document 3: '{doc3}' (with threshold)")
    print(f"  -> Predicted Category: {cl.classify(doc3, default='unknown')}")
