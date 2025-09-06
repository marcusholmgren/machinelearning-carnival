# recommendations_updated.py
"""
A library for a basic collaborative filtering recommendation engine.

This script provides functions to calculate user and item similarities based on
ratings data, and then generate personalized recommendations. It includes implementations
for both user-based and item-based collaborative filtering.

Key Concepts:
- Collaborative Filtering: A technique that can filter out items that a user might
  like on the basis of reactions by similar users.
- User-Based Filtering: Recommends items by finding similar users. If user A and B
  have similar tastes, items that A likes will be recommended to B.
- Item-Based Filtering: Recommends items that are similar to items a user has
  previously liked. It calculates an item-item similarity matrix.
"""

from math import sqrt
from typing import Dict, List, Tuple, Callable

# A type alias for our preferences dictionary for better readability.
# Format: { 'UserName': {'ItemName': Rating, ...}, ... }
PrefsDict = Dict[str, Dict[str, float]]
ItemMatch = Dict[str, List[Tuple[float, str]]]


# A sample dictionary of movie critics and their ratings of a small set of movies.
critics: PrefsDict = {
    'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
                  'Just My Luck': 3.0, 'Superman Returns': 3.5,
                  'You, Me and Dupree': 2.5, 'The Night Listener': 3.0},
    'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
                     'Just My Luck': 1.5, 'Superman Returns': 5.0,
                     'The Night Listener': 3.0, 'You, Me and Dupree': 3.5},
    'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
                         'Superman Returns': 3.5, 'The Night Listener': 4.0},
    'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
                     'The Night Listener': 4.5, 'Superman Returns': 4.0,
                     'You, Me and Dupree': 2.5},
    'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                     'Just My Luck': 2.0, 'Superman Returns': 3.0,
                     'The Night Listener': 3.0, 'You, Me and Dupree': 2.0},
    'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                      'The Night Listener': 3.0, 'Superman Returns': 5.0,
                      'You, Me and Dupree': 3.5},
    'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0,
             'Superman Returns': 4.0}
}


# --- Similarity Metrics ---

def sim_distance(prefs: PrefsDict, person1: str, person2: str) -> float:
    """
    Calculates the Euclidean distance-based similarity score for two people.

    This score is between 0 and 1, where 1 means the two people have identical
    ratings for all commonly rated items. The score is calculated by inverting
    the Euclidean distance, so a smaller distance yields a higher similarity score.

    Args:
        prefs: The main dictionary of all user ratings.
        person1: The name of the first person.
        person2: The name of the second person.

    Returns:
        A float representing the similarity score.
    """
    # Find the set of items rated by both people for efficient calculation.
    shared_items = {item for item in prefs[person1] if item in prefs[person2]}

    # If they have no ratings in common, their similarity is 0.
    if not shared_items:
        return 0.0

    # Calculate the sum of the squares of the differences in their ratings.
    sum_of_squares = sum(
        pow(prefs[person1][item] - prefs[person2][item], 2)
        for item in shared_items
    )

    # The formula 1 / (1 + sqrt(sum_of_squares)) normalizes the distance.
    # It ensures the value is between 0 and 1, and inverts it so that
    # a smaller distance (more similar) results in a larger number.
    return 1 / (1 + sqrt(sum_of_squares))


def sim_pearson(prefs: PrefsDict, person1: str, person2: str) -> float:
    """
    Calculates the Pearson correlation coefficient for two people.

    This score measures the linear relationship between two sets of data. It ranges
    from -1 (perfect negative correlation) to +1 (perfect positive correlation),
    with 0 indicating no correlation. It's useful because it corrects for "grade
    inflation" (e.g., one user consistently gives higher ratings than another).

    Args:
        prefs: The main dictionary of all user ratings.
        person1: The name of the first person.
        person2: The name of the second person.

    Returns:
        A float representing the Pearson correlation score.
    """
    # Find the set of items rated by both people.
    shared_items = {item for item in prefs[person1] if item in prefs[person2]}
    num_shared_items = len(shared_items)

    # If they have no ratings in common, their correlation is 0.
    if num_shared_items == 0:
        return 0.0

    # Calculate the sums of ratings, squares of ratings, and products of ratings
    # for the commonly rated items. These are the building blocks of the formula.
    sum1 = sum(prefs[person1][item] for item in shared_items)
    sum2 = sum(prefs[person2][item] for item in shared_items)

    sum1_sq = sum(pow(prefs[person1][item], 2) for item in shared_items)
    sum2_sq = sum(pow(prefs[person2][item], 2) for item in shared_items)

    product_sum = sum(prefs[person1][item] * prefs[person2][item] for item in shared_items)

    # Calculate the Pearson correlation score.
    numerator = product_sum - (sum1 * sum2 / num_shared_items)
    denominator_part1 = sum1_sq - pow(sum1, 2) / num_shared_items
    denominator_part2 = sum2_sq - pow(sum2, 2) / num_shared_items

    # Denominator can be zero if one person's ratings are all the same.
    # Avoid division by zero.
    if denominator_part1 * denominator_part2 == 0:
        return 0.0

    denominator = sqrt(denominator_part1 * denominator_part2)

    return numerator / denominator


def sim_cosine(prefs: PrefsDict, person1: str, person2: str) -> float:
    """
    Calculates the Cosine Similarity between two people.

    Cosine similarity measures the cosine of the angle between two non-zero vectors.
    It is a judgment of orientation rather than magnitude. Two vectors with the
    same orientation have a cosine similarity of 1.

    Args:
        prefs: The main dictionary of all user ratings.
        person1: The name of the first person.
        person2: The name of the second person.

    Returns:
        A float representing the cosine similarity score.
    """
    # Find the set of items rated by both people.
    shared_items = {item for item in prefs[person1] if item in prefs[person2]}

    if not shared_items:
        return 0.0

    # Calculate dot product and magnitudes
    dot_product = sum(prefs[person1][item] * prefs[person2][item] for item in shared_items)
    magnitude1 = sqrt(sum(pow(prefs[person1][item], 2) for item in shared_items))
    magnitude2 = sqrt(sum(pow(prefs[person2][item], 2) for item in shared_items))

    # Avoid division by zero
    if magnitude1 * magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


# --- Recommendation Algorithms ---

def top_matches(prefs: PrefsDict, person: str, n: int = 5,
               similarity: Callable[[PrefsDict, str, str], float] = sim_pearson) -> List[Tuple[float, str]]:
    """
    Returns the best matches for a given person from the prefs dictionary.

    Args:
        prefs: The main dictionary of all user ratings.
        person: The person to find matches for.
        n: The number of matches to return.
        similarity: The similarity function to use (e.g., sim_pearson).

    Returns:
        A sorted list of tuples, each containing (similarity_score, person_name).
    """
    scores = [(similarity(prefs, person, other), other)
              for other in prefs if other != person]

    # Sort the list to have the highest scores at the top.
    scores.sort(reverse=True)
    return scores[0:n]


def get_recommendations(prefs: PrefsDict, person: str,
                       similarity: Callable[[PrefsDict, str, str], float] = sim_pearson) -> List[Tuple[float, str]]:
    """
    Gets recommendations for a person using a weighted average of every other user's rankings.
    This is a classic example of USER-BASED collaborative filtering.

    Args:
        prefs: The main dictionary of all user ratings.
        person: The person to get recommendations for.
        similarity: The similarity function to use.

    Returns:
        A sorted list of recommended items as (predicted_score, item_name) tuples.
    """
    weighted_scores: Dict[str, float] = {}
    similarity_sums: Dict[str, float] = {}

    for other in prefs:
        # Don't compare a person to themselves.
        if other == person:
            continue

        sim = similarity(prefs, person, other)

        # Ignore scores of zero or lower, as they indicate no or negative correlation.
        if sim <= 0:
            continue

        for item in prefs[other]:
            # Only score items the target person hasn't seen yet.
            if item not in prefs[person]:
                # The predicted score for an item is a weighted sum.
                # It's the sum of (similarity_of_other_user * their_rating_for_item).
                weighted_scores.setdefault(item, 0)
                weighted_scores[item] += prefs[other][item] * sim

                # We also need to sum the similarities of all users who rated this item.
                # This is for normalization later.
                similarity_sums.setdefault(item, 0)
                similarity_sums[item] += sim

    # Create the normalized list of recommendations.
    # The final score is the weighted score sum divided by the similarity sum.
    # This gives a weighted average, preventing items rated by many low-similarity
    # users from outranking items rated by a few high-similarity users.
    rankings = [(total / similarity_sums[item], item)
                for item, total in weighted_scores.items() if similarity_sums[item] != 0]

    # Return the sorted list, from highest score to lowest.
    rankings.sort(reverse=True)
    return rankings


# --- Item-Based Collaborative Filtering ---

def transform_prefs(prefs: PrefsDict) -> PrefsDict:
    """
    Inverts the preference matrix from {user: {item: rating}} to {item: {user: rating}}.

    This is a crucial step for item-based collaborative filtering, as it allows us
    to easily calculate similarities between items instead of between users.

    Args:
        prefs: The original user-item ratings dictionary.

    Returns:
        An inverted item-user ratings dictionary.
    """
    result: PrefsDict = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            result[item][person] = prefs[person][item]
    return result


def calculate_similar_items(prefs: PrefsDict, n: int = 10) -> ItemMatch:
    """
    Creates a dictionary of items showing which other items they are most similar to.

    This pre-computes the item-item similarity matrix, which is the core of
    item-based collaborative filtering. This is computationally expensive but only
    needs to be run periodically, not for every recommendation request.

    Args:
        prefs: The main dictionary of all user ratings.
        n: The number of similar items to store for each item.

    Returns:
        A dictionary where keys are item names and values are a list of
        (similarity_score, similar_item_name) tuples.
    """
    result: ItemMatch = {}
    # Invert the preference matrix to be item-centric.
    itemPrefs = transform_prefs(prefs)

    print("Calculating item similarity matrix...")
    for i, item in enumerate(itemPrefs):
        # Provide status updates for large datasets.
        if (i + 1) % 100 == 0:
            print(f"{i + 1} / {len(itemPrefs)}")

        # Find the most similar items to this one using Euclidean distance.
        # Item-based filtering often uses Euclidean or Cosine distance.
        scores = top_matches(itemPrefs, item, n=n, similarity=sim_distance)
        result[item] = scores
    print("...done.")
    return result


def get_recommended_items(prefs: PrefsDict, itemMatch: ItemMatch, user: str) -> List[Tuple[float, str]]:
    """
    Gets recommendations for a user using an item-item similarity matrix.
    This is a classic example of ITEM-BASED collaborative filtering.

    Args:
        prefs: The original user-item ratings dictionary.
        itemMatch: The pre-computed item-item similarity matrix from calculateSimilarItems.
        user: The user to get recommendations for.

    Returns:
        A sorted list of recommended items as (predicted_score, item_name) tuples.
    """
    userRatings = prefs[user]
    scores: Dict[str, float] = {}
    totalSim: Dict[str, float] = {}

    # Loop over items rated by this user.
    for (item, rating) in userRatings.items():
        # Loop over items similar to this one.
        for (similarity, item2) in itemMatch.get(item, []):
            # Ignore if this user has already rated this similar item.
            if item2 in userRatings:
                continue

            # The score for a candidate item is a weighted sum of the user's ratings
            # for similar items. The weight is the similarity between the items.
            scores.setdefault(item2, 0)
            scores[item2] += similarity * rating

            # Sum of all the similarities for normalization.
            totalSim.setdefault(item2, 0)
            totalSim[item2] += similarity

    # Divide each total score by the total weighting to get an average.
    rankings = [(score / totalSim[item], item)
                for item, score in scores.items() if totalSim.get(item, 0) != 0]

    # Return the rankings from highest to lowest.
    rankings.sort(reverse=True)
    return rankings


# --- Data Loading Utility ---

def loadMovieLens(path: str = './ml-100k') -> PrefsDict:
    """
    Loads the MovieLens 100k dataset.

    Assumes the data files (u.item and u.data) are in the specified path.
    Download from: https://grouplens.org/datasets/movielens/100k/

    Args:
        path: The path to the directory containing u.item and u.data.

    Returns:
        A preferences dictionary in the standard format.
    """
    # Get movie titles
    movies: Dict[str, str] = {}
    try:
        with open(f'{path}/u.item', 'r', encoding='ISO-8859-1') as f:
            for line in f:
                (movie_id, title) = line.split('|')[0:2]
                movies[movie_id] = title
    except FileNotFoundError:
        print(f"Error: Could not find u.item at path: {path}/u.item")
        return {}

    # Load data
    prefs: PrefsDict = {}
    try:
        with open(f'{path}/u.data', 'r') as f:
            for line in f:
                (user, movieid, rating, ts) = line.split('\t')
                prefs.setdefault(user, {})
                prefs[user][movies[movieid]] = float(rating)
    except FileNotFoundError:
        print(f"Error: Could not find u.data at path: {path}/u.data")
        return {}

    return prefs

if __name__ == '__main__':
    print("--- User-Based Recommendations for Toby ---")
    # Find movie recommendations for Toby using Pearson correlation
    user_based_recs = get_recommendations(critics, 'Toby', similarity=sim_pearson)
    print("Movies recommended for Toby:")
    for score, movie in user_based_recs:
        print(f"  - {movie} (Predicted Score: {score:.2f})")
    print("\n" + "="*50 + "\n")

    print("--- Item-Based Recommendations for Toby ---")
    # 1. Pre-calculate the item similarity matrix
    item_similarity_matrix = calculate_similar_items(critics)

    # 2. Get recommendations for Toby based on item similarity
    item_based_recs = get_recommended_items(critics, item_similarity_matrix, 'Toby')
    print("Movies recommended for Toby (based on items he liked):")
    for score, movie in item_based_recs:
        print(f"  - {movie} (Predicted Score: {score:.2f})")
    print("\n" + "="*50 + "\n")

    print("--- Finding Users Similar to Lisa Rose ---")
    similar_users = top_matches(critics, 'Lisa Rose', n=3)
    print("Top 3 users similar to Lisa Rose:")
    for score, user in similar_users:
        print(f"  - {user} (Similarity: {score:.2f})")
    print("\n" + "="*50 + "\n")

    print("--- Finding Movies Similar to 'Superman Returns' ---")
    # To find similar items, we first flip the data to be item-centric
    item_prefs = transform_prefs(critics)
    similar_movies = top_matches(item_prefs, 'Superman Returns', n=3, similarity=sim_distance)
    print("Top 3 movies similar to 'Superman Returns':")
    for score, movie in similar_movies:
        print(f"  - {movie} (Similarity: {score:.2f})")
