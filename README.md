# IBLT-Stress: Predicting Stress From Text

## Introduction
Dynamic Decision Making (DDM) describes how solutions to problems can evolve over time. In this context, drawing from past experiences and retaining current solutions in memory offers a framework for addressing similar issues in the future. This concept underpins Instance-Based Learning Theory (IBLT) (Gonzalez, 2003), which was introduced to emulate the human problem-solving approach. As outlined in the paper, the primary stages of IBLT are recognition, judgment, choice, and feedback.

Stress detection, a prominent domain in natural language processing, becomes more intricate when applied in real-world scenarios. The definition and threshold of stress can vary among individuals, as can their susceptibility to stress. Consequently, a scenario arises where one or even multiple pre-trained models may falter in real-world classification tasks. This predicament mirrors dynamic decision making, suggesting that individualized definitions of stress could enhance classification accuracy beyond that of pre-trained models.

This project’s aim is to develop a stress detection system that utilizes machine learning techniques within an instance-based learning framework to offer users precise and personalized stress assessments based on their textual input. The objective of the stress detection system is personalization, which necessitates two core components: a baseline predictive classifier and a lazy learning classifier. Both classifiers contribute to the recognition stage of IBLT, but only the lazy learning classifier is capable of adaptation based on feedback.

The baseline predictive classifier serves to establish a knowledge base regarding how stress is perceived within a population. For this purpose, a linear support vector machine (SVM) model is employed. SVM is preferred over other models due to its computational efficiency in the field of stress detection in NLP, exhibiting comparable performance to larger neural networks and transformers while being less computationally demanding. For the SVM specific to serving as a baseline model for stress prediction, it is denoted as SVMe. 

The K-nearest neighbors (KNN) algorithm is selected for the lazy learning classifier. Although KNN is commonly used in stress detection, it consistently underperforms as a standalone classifier compared to SVM. However, KNN is chosen for its capacity to learn without explicit training and its ability to encapsulate an individual's past experiences. Euclidean distance is utilized as the distance metric for the KNN algorithm, consistent with standard KNN implementations.

In addition to these two core components, IBLT necessitates components for judgment, choice, and feedback. The judgment step is managed by the judge component, which employs a heuristic approach to select an output from the classifiers. The choice step comes into play when the user opts to provide their stress level. At this stage, a comparison is made between the stress level determined by the system (probabilistic stress, Sp) and the stress level provided by the user (deterministic stress, Sd). The feedback step involves an adaptation process where techniques are employed to reinforce correct predictions and mitigate incorrect ones.

Finally, a final component is used, specific for this problem space. Stress is defined in the field of psychology as a unique blend of emotions. As such, a middle step between corpus and stress prediction is proposed to convert text into a probability vector of emotions (P(E)). P(E) is predicted using SVM as a classifier, denoted as SVMt. KNN and SVMe, the predictive components of stress, use an input of P(E) to predict stress levels.

## Base Knowledge (Pre-trained Model)
![Base knowledge](https://github.com/notaaronkuo/iblt-stress/blob/main/resources/pretrained-model-diagram.png?raw=true)


<sub>Figure 1: The pre-trained moddels, which are the text-converting SVM (SVMt) and the emotion-converting SVM (SVMe), are trained in parallel with the GoEmotions dataset.</sub>


The intent of the pre trained model is to provide a base knowledge for the system to start with. Eventually the lazy learning model, the kNN, will provide personalized information necessary to provide a more informed result.
The pre-trained SVM models utilized in this project leverage the GoEmotions dataset, which associates words with specific emotions. The SVMt model uses this to train to convert textual inputs into emotion probability vectors (Pe). Additionally, a baseline emotion-to-stress mapping of weights is established and utilized to train the SVMe model. By mapping emotions to stress levels, SVMe provides a crucial link between the emotional context inferred from the text and the ultimate stress level prediction.

## Instance Based Learning Loop
![The IBLT loop outlines the overarching agents that work together to reach a prediction.](https://raw.githubusercontent.com/notaaronkuo/iblt-stress/main/resources/IBLT-loop-diagram.png?token=GHSAT0AAAAAACRWSVBIJMHA4LGY4SFMKLL2ZRRWM6A)


<sub>Figure 2: The IBLT loop outlines the overarching agents that work together to reach a prediction. </sub>


The stress detection system begins by receiving user text input, which is then processed to produce Sp. Initially, the text is converted into a vector of Pe through the SVMt model, establishing the emotional context of the input. Subsequently, both the kNN and SVMe models utilize Pe as input to generate their respective stress predictions. In the case where user data is insufficient for outputting Sp from kNN, the SVMe output by default.

## Judge
![Judge diagram](https://github.com/notaaronkuo/iblt-stress/blob/main/resources/judge-diagram.png?raw=true)


<sub>Figure 3: The judge decides what the stress reported by the system is after evaluating the SVMe and kNN.</sub>



The Judge component determines the final stress level output in our system. It evaluates the Sp generated by both the SVMe and kNN models using an approach of heuristics. In the normal case where SVMe and kNN produce identical stress levels, the Judge simply selects and utilizes this shared Sp value. However, in instances where SVMe and kNN outputs diverge, the Judge employs a decision-making process. If the discrepancy is minor and the kNN-generated Sp is relatively close to that of SVMe, the Judge opts for the kNN Sp. Conversely, if the discrepancy is significant, the Judge favors the SVMe. Additionally, when users provide their perceived stress level (Sd), the Judge compares it with the system-generated Sp and stress values obtained from other sources. Based on this comparison, the Judge determines which value is more likely to be accurate.

## Installation
To set up IBLT-Stress, follow these steps:
1. Clone the repository
2. Install the dependencies provided.
3. Run main.py to run mock user tests.
