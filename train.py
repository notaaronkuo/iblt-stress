from svm import SVMClassifier, EmotionSVM, StressSVM
import tensorflow_datasets as tfds

if __name__ == "__main__":

    # Create SVM classifier object instance
    emotion_classifier = SVMClassifier(EmotionSVM())

    # Load the GoEmotions dataset
    df, _ = tfds.load('goemotions', split='train[:10000]', with_info=True)

    # Train the emotion classifier
    print("Training SVMt...")
    emotion_classifier.train(df)
    print("SVMt trained!")

    # Test prediction for emotion classifier

    # user_input = input("Enter a text for emotion prediction: ")
    # predicted_emotions, probabilities = emotion_classifier.predict(user_input)
    # print("Predicted emotions:", predicted_emotions)
    # print("Probabilities:", probabilities)

    # Create StressSVM instance and pass the trained EmotionSVM model
    stress_classifier = SVMClassifier(StressSVM(emotion_classifier.strategy))
    df, _ = tfds.load('goemotions', split='train[:10000]', with_info=True)

    print("Training SVMe...")
    stress_classifier.train(df)
    print("SVMe trained!")

    # Test prediction for stress classifier using the trained emotion model
    while True:
        user_input = input("Enter a text for stress prediction: ")
        _, x_prob = emotion_classifier.predict(user_input)
        predicted_stress = stress_classifier.predict(list(x_prob.values()))
        print("Predicted stress: ", predicted_stress)