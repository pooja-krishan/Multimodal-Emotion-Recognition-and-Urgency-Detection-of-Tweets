# Speech-text-based Emotion Recognition and Urgency Detection

## Introduction

This README provides a comprehensive overview of the objectives, methodologies, datasets, feature extraction techniques, models used, and results analysis for both parts of the project.

## Part 1: Speech-text-based Emotion Recognition

### Problem Statement and Proposed System

This section addresses the challenging task of emotion recognition using speech and text data. The project's goal is to distinguish between four primary emotions: Neutral, Angry, Sad, and Happy. The proposed system is shown below:

### Dataset Description

The primary dataset employed is the Interactive Emotional Dyadic Motion Capture (IEMOCAP) dataset, a rich source of multimodal and multi-speaker data. The Interactive Emotional Dyadic Motion Capture (IEMOCAP) dataset offers a versatile platform for studying emotions across various modalities. The dataset includes audiovisual elements like video, speech, facial motion capture, and text transcriptions. Annotations by multiple annotators include categorical labels (e.g., anger, happiness, sadness, neutrality) and dimensional labels (valence, activation, dominance).

### Visualization of DataSet

![1](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/1.PNG)

### Feature Extraction

For text transcription, two distinct approaches are explored:
1. Extraction of 10 features (e.g., average word length, sentence length, number of words, number of verbs) resembling a previous project (with limited accuracy).
2. Leveraging a BERT pre-trained model from PyTorch and Hugging Face to generate embedding vectors for each text.

![2](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/2.PNG)

For speech, a comprehensive set of 47 acoustic features is extracted from WAV files for each sentence. This set includes 13 MFCC parameters, 13 delta coefficients, and 13 acceleration coefficients.

![3](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/3.PNG)

### Models Used

**Speech-based Classification:**
- Logistic Regression
- MultinomialNB
- RandomForestClassifier

![4](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/4.PNG)

**Speech-based Regression:**
- Linear Regression
- Random Forest Regression
- Support Vector Machines

![5](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/5.PNG)

**Text-based Classification:**
- Logistic Regression
- RandomForestClassifier
- Gradient Boosting Classifier
- RandomForestClassifier with BERT

![7](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/7.PNG)

### Results Analysis (Confusion Matrices)

![6](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/6.PNG)

![8](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/8.PNG)

## Part 2: Urgency Detection of Tweets

### Problem Statement and Proposed System

This part of the project shifts the focus to urgency detection within text data, particularly in the context of tweets. The objective is to identify the level of urgency in these messages, with applications in fields like law enforcement, humanitarian crises, and healthcare hotlines.

![9](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/9.PNG)

### Dataset Description

The dataset used is from Figure-Eight, containing post-disaster messages annotated with binary labels (0 or 1) for 36 predefined aid categories. The Figure-Eight disaster-response dataset provides essential insights into urgency detection within social media posts. The word cloud of the dataset and class imbalance inherent in the dataset is shown below:

![10](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/10.PNG)

![11](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/11.PNG)

### Feature Extraction

The following features were extracted based on extensive research and based on an article in [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/):
1. **Word Count**: Count of words in each tweet.
2. **Character Count**: Count of characters in each tweet.
3. **Average Word Length**: Average length of words in each tweet.
4. **Stopword Count**: Count of stopwords (common words like "and", "the", "is", etc.) in each tweet.
5. **Hashtag Count**: Count of hashtags (#) in each tweet.
6. **Mention Count**: Count of user mentions (@) in each tweet.
7. **URL Count**: Count of URLs in each tweet.

These features have been carefully selected based on their relevance and potential impact in the context of the Twitter dataset. The extraction of these features serves as an important step in building effective custom classifiers for text analysis on Twitter data.

### Models Used

**Pre-trained models:**
- BERT
- DistilBERT v1
- DistilBERT v2
- Enrie

![12](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/12.PNG)

**Models built from scratch:**
- Logistic Regression
- k Nearest Neighbors
- Random Forest

![14](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/14.PNG)


### Results Analysis

![13](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/13.PNG)

![15](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/15.PNG)

![16](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/16.PNG)

![17](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/17.PNG)

### Screenshot of Demo

![8](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/18.PNG)

![19](https://github.com/pooja-krishan/Multimodal-Emotion-Recognition-and-Urgency-Detection-of-Tweets/blob/main/fig/19.PNG)

## Conclusion

This README has provided an in-depth look into the Speech-text-based Emotion Recognition and Urgency Detection project. The combination of diverse datasets, innovative feature extraction techniques, and a variety of models showcases the project's breadth and potential applications. This documentation offers valuable insights and serves as a resource for anyone interested in understanding and exploring these crucial aspects of natural language processing and data analysis.