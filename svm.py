import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


class SVMStrategy:
    def train(self, data):
        raise NotImplementedError("Train method must be implemented by subclasses")

    def predict(self, input_text):
        raise NotImplementedError("Predict method must be implemented by subclasses")


class EmotionSVM(SVMStrategy):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if not self.__initialized:
            self.__initialized = True
            self.tfidf_vectorizer = TfidfVectorizer()
            # Enable probability estimates by setting probability=True
            self.svm_classifier = SVC(kernel='linear', probability=True)

    def train(self, data):
        # Initialize lists to store vectors
        vectors = []

        # Loop through the dataset to extract vectors
        for example in data:
            # Extract the text data
            text = example['comment_text'].numpy().decode('utf-8')

            # Extract the boolean values for each emotion
            emotions = [emotion for emotion in example.keys() if emotion != 'comment_text' and example[emotion].numpy()]

            # If there are multiple positive labels, keep one at random
            if len(emotions) > 1:
                keep_emotion = np.random.choice(emotions)
                emotions = [keep_emotion]
            elif len(emotions) == 0:
                emotions = ['neutral']

            # Create the vector with text as the first element
            example_vector = [text] + emotions
            vectors.append(example_vector)

        # Convert list of lists to numpy array
        vectors = np.array(vectors)

        # Separate features (text) and labels (emotions)
        text_features = vectors[:, 0]  # First column contains text
        emotions_labels = vectors[:, 1]  # Second column contains emotion labels

        # Convert text features to numerical representations using TF-IDF
        X = self.tfidf_vectorizer.fit_transform(text_features)

        # Train the classifier
        self.svm_classifier.fit(X, emotions_labels)

    def predict(self, input_text):
        # Preprocess the input text
        preprocessed_text = input_text.lower()  # Example: Convert to lowercase

        # Vectorize the preprocessed text using TF-IDF
        input_vector = self.tfidf_vectorizer.transform([preprocessed_text])

        # Predict the emotion labels
        predicted_labels = self.svm_classifier.predict(input_vector)

        # Predict probability estimates
        probability_estimates = self.svm_classifier.predict_proba(input_vector)

        # Create a dictionary to store probabilities of each emotion
        probabilities = {emotion: proba for emotion, proba in
                         zip(self.svm_classifier.classes_, probability_estimates[0])}

        return predicted_labels, list(probabilities.values())


class StressSVM(SVMStrategy):
    def __init__(self, emotion_model):
        self.emotion_model = emotion_model
        self.svm_classifier = SVC(kernel='linear')

    def _map_emotion_to_stress(self, emotion):
        # Map emotion to stress level
        stress_mapping = {
            'admiration': 2, 'amusement': 3, 'anger': 5, 'annoyance': 4, 'approval': 2,
            'caring': 1, 'confusion': 4, 'curiosity': 3, 'desire': 2, 'disappointment': 5,
            'disapproval': 5, 'disgust': 5, 'embarrassment': 5, 'excitement': 1, 'fear': 5,
            'gratitude': 1, 'grief': 4, 'joy': 1, 'love': 1, 'nervousness': 4, 'neutral': 4,
            'optimism': 1, 'pride': 1, 'realization': 2, 'relief': 1, 'remorse': 4,
            'sadness': 4, 'surprise': 2
        }
        return stress_mapping.get(emotion, 3)  # Default to neutral stress level if emotion not found

    def train(self, data):
        # Initialize lists to store vectors
        X = []
        y = []

        # Loop through the dataset to extract vectors
        for example in data:
            # Extract the text data
            text = example['comment_text'].numpy().decode('utf-8')

            # Predict emotions using the emotion model
            predicted_emotion, probability_vector = self.emotion_model.predict(text)

            # Convert emotion to stress level
            stress_level = self._map_emotion_to_stress(predicted_emotion[0])

            # Append data to training set
            X.append(probability_vector)
            y.append(stress_level)

        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)
        # Train the SVM classifier
        self.svm_classifier.fit(X, y)

    def predict(self, probability_vector):
        # Convert emotion to stress level
        stress_level = self.svm_classifier.predict([probability_vector])[0]

        return stress_level


class SVMClassifier:
    def __init__(self, strategy):
        self.strategy = strategy

    def train(self, data):
        self.strategy.train(data)

    def predict(self, input_text):
        return self.strategy.predict(input_text)


