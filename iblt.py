from svm import SVMClassifier, EmotionSVM, StressSVM
from knn import StressPredictorKNN
from metrics import IBLTMetrics
import tensorflow_datasets as tfds


class IBLT:
    def __init__(self):
        raise RuntimeError("IBLT: Improper initialization, needs K!")

    def __init__(self, K):
        """
        Class containing IBLT process. Needs to train before using.
        :param K: k of kNN, value required
        """
        self.iterations = 0
        self.current_user_emotion = None
        self.current_prob_stress = None
        self.text_svm = None
        self.emotion_svm = None
        self.emotion_knn = StressPredictorKNN(k=K)
        self.metrics = IBLTMetrics()

    def train_GoEmotions(self, train_size=10000):
        """
        A function to train IBLT via GoEmotions.
        :param train_size: size of sample training data, 10000 by default
        :return: None
        """

        # Create SVM classifier object instance
        emotion_classifier = SVMClassifier(EmotionSVM())

        # Load the GoEmotions dataset
        df, _ = tfds.load('goemotions', split=f'train[:{train_size}]', with_info=True)

        # Train the emotion classifier
        print("Training SVMt...")
        emotion_classifier.train(df)
        print("SVMt trained!")

        # Create StressSVM instance and pass the trained EmotionSVM model
        stress_classifier = SVMClassifier(StressSVM(emotion_classifier.strategy))
        df, _ = tfds.load('goemotions', split=f'train[:{train_size}]', with_info=True)

        print("Training SVMe...")
        stress_classifier.train(df)
        print("SVMe trained!")

        self.text_svm = emotion_classifier
        self.emotion_svm = stress_classifier

    def predict(self, user_text):
        """
        Predicts stress level from a given text using IBLT.
        :param user_text: Str
        :return: int 1-5
        """
        if not user_text:
            raise RuntimeError("IBLT: No user input detected.")

        if self.emotion_svm is None or self.text_svm is None:
            raise RuntimeError("IBLT: Model has not been trained yet, cannot predict. ")

        print("IBLT: Running predictions...")

        # Keep track of loop
        self.iterations += 1

        # Convert to emotion probability vector
        emotion, emotion_prob = self.text_svm.predict(user_text)
        self.current_user_emotion = emotion_prob
        svme_stress = self.emotion_svm.predict(emotion_prob)
        self.metrics.svm_calls.append(svme_stress)
        # Evaluate if there are enough samples for a knn prediction
        knn_stress = None
        if self.emotion_knn.can_predict():
            knn_stress = self.emotion_knn.predict(emotion_prob)
            self.metrics.knn_calls.append(knn_stress)
        else:
            print("IBLT: kNN impossible, more samples needed.")
            self.metrics.knn_calls.append(0)

        judge_decision = self.judge(svm_output=svme_stress, knn_output=knn_stress)
        self.metrics.final_calls.append(judge_decision)
        return judge_decision

    def judge(self, svm_output, knn_output):
        """
        Judge uses a rule of thumb approach to conservatively select the probabilistic output.
        Picks kNN if output is within +/- 1 of the SVMe output. Chooses SVMe output otherwise.
        :param svm_output: Output from pretrained model
        :param knn_output: Output from incremental learning model
        :return: Output of model as selected by the judge
        """
        judge_flag = "SVMt"
        if self.emotion_svm is None or self.text_svm is None:
            raise RuntimeError("IBLT: Model has not been trained yet, cannot judge. ")

        if not knn_output:
            self.current_prob_stress = svm_output

        elif abs(svm_output - knn_output) <= 1:
            self.current_prob_stress = knn_output
            judge_flag = "kNN"

        else:
            self.current_prob_stress = svm_output

        print(f"\nSVMe Value: {svm_output}\nkNN Value: {knn_output}\nIBLT: Judge determination: {judge_flag}\n")
        self.metrics.final_verdict.append(judge_flag)
        return self.current_prob_stress

    def adapt(self, deterministic_stress, probabilistic_stress):
        """
        Adapt the system to increase future accuracy, to be used after predictions for continuous learning.
        :param deterministic_stress: Stress from a certain source, ground truth
        :param probabilistic_stress: Stress as determined by the System
        :return: None
        """
        # Adapt the model using new data
        self.emotion_knn.append_sample(X=self.current_user_emotion,
                                       y=deterministic_stress if deterministic_stress else probabilistic_stress)
        if self.iterations >= (self.emotion_knn.k * 2) and self.iterations % self.emotion_knn.k == 0:
            self.emotion_knn.evict_LFU()

        # append sample, label
        # future work: re-label
