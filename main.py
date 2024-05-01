import math

import numpy as np
import tensorflow_datasets as tfds

from iblt import IBLT

# Test script is designed to run IBLT in a test environment of given total iterations.

# k of kNN is usually square root of total N as a general rule of thumb
total_iterations = 1000
K = round(math.sqrt(total_iterations))


def test_GoEmotions():
    """
    Function to generate a test set text and stress level, of size = total iterations. Uses GoEmotions.
    :return: List of texts, List of stress level integers
    """
    stress_mapping = {
        'admiration': 2, 'amusement': 3, 'anger': 5, 'annoyance': 4, 'approval': 2,
        'caring': 1, 'confusion': 4, 'curiosity': 3, 'desire': 2, 'disappointment': 5,
        'disapproval': 5, 'disgust': 5, 'embarrassment': 5, 'excitement': 1, 'fear': 5,
        'gratitude': 1, 'grief': 4, 'joy': 1, 'love': 1, 'nervousness': 4, 'neutral': 4,
        'optimism': 1, 'pride': 1, 'realization': 2, 'relief': 1, 'remorse': 4,
        'sadness': 4, 'surprise': 2
    }

    df, _ = tfds.load('goemotions', split=f'test[:{total_iterations}]', with_info=True)
    texts = []
    stress_values = []
    for example in df:
        # Extract the text data
        text = example['comment_text'].numpy().decode('utf-8')
        texts.append(text)

        # Extract the boolean values for each emotion
        emotions = [emotion for emotion in example.keys() if emotion != 'comment_text' and example[emotion].numpy()]

        # If there are multiple positive labels, keep one at random
        if len(emotions) > 1:
            keep_emotion = np.random.choice(emotions)
            emotions = [keep_emotion]
        elif len(emotions) == 0:
            emotions = ['neutral']

        # Assign stress value based on emotion
        stress_value = max([stress_mapping[emotion] for emotion in emotions])
        stress_values.append(stress_value)

    return texts, stress_values


if __name__ == "__main__":
    # Test script is designed to run IBLT in a test environment of given total iterations
    texts, stress_values = test_GoEmotions()

    # Initialize the IBLT, with kNN k=K
    IBLT_test = IBLT(K)

    # Load the solution key stress values into metrics
    IBLT_test.metrics.deterministic = stress_values

    # Train the SVMt and SVMe
    IBLT_test.train_GoEmotions()

    # Track correct guesses and total iterations
    correct = 0
    total = 0
    # Track the accuracy of the judge for analytics purposes
    j_correct = 0
    j_total = 0

    for sentence, true_stress in zip(texts, stress_values):
        # Predict stress level from a sentence
        predicted_stress = IBLT_test.predict(sentence)
        if predicted_stress == true_stress:
            correct += 1
            j_correct += 1
        total += 1
        j_total += 1
        accuracy = correct / total
        # Append accuracy to track over time
        IBLT_test.metrics.accuracy_by_n.append(accuracy)

        IBLT_test.metrics.judge_accuracy.append(j_correct / j_total)
        if total == K:
            # Reset judge increments before K iterations, as judge is not activated then
            j_total = 0
            j_correct = 0

        # Adaptation process, given the probabilistic and deterministic values
        IBLT_test.adapt(true_stress, predicted_stress)

    # First K values don't matter for judge accuracy
    IBLT_test.metrics.judge_accuracy = IBLT_test.metrics.judge_accuracy[:K]
    # Show metrics and graphs
    IBLT_test.metrics.report_metrics()
