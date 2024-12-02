# language-model-and-poet-classification-
build a language model for persian poets and try to identify their poems



## Key Sections:
Imports and Setup:

Essential libraries such as numpy, pandas, matplotlib, hazm (for Persian text processing), and heapq are imported.
### Tokenizer Functions:

Functions like unigram_line and bigram_line tokenize the input text into unigrams (single words) and bigrams (pairs of consecutive words).
### Data Preparation:

The prepare_data function processes a text file and creates dictionaries for storing unigrams, bigrams, and counts for each poet, as well as overall counts. This function also handles reading the data, tokenizing it, and filling the dictionaries.
### Unigram Language Model:

The simple unigram model computes perplexity using the frequency of words in the dataset. The function Simple_Unigram_Perplexity calculates the perplexity for each line in the test data based on the unigram frequency.
The unigram model with absolute discounting (Unigram_Perplexity) introduces a discount parameter (delta) to smooth the probabilities and calculate the perplexity.
The best delta is determined by testing a range of values and selecting the one that minimizes perplexity.
Perplexity Calculations: Perplexity is calculated as an inverse measure of model confidence in predicting a sentence, with lower values indicating better models.
### Bigram Language Model:

Similarly to the unigram model, the bigram model (Bigram_Perplexity) calculates perplexity using pairs of consecutive words.
The code includes an optimization for determining the best delta for the bigram model and calculates the perplexity for test data.
### Data for Specific Poets:

The prepare2_data function processes the data for individual poets, enabling analysis and evaluation of perplexity for each poet using both unigram and bigram models.
### Information Gain (IG):

Information Gain (IG_calculator) is a measure used to evaluate the importance of words. It computes the information gain of a word based on its distribution across different poets.
The top 10 words with the highest information gain are identified and displayed.
X-Square Test (X_Square) calculates the chi-squared statistic for a word, assessing the dependency between the word's occurrence and different poets. Words with the highest X-squared values are also identified.
### Observations:
Perplexity and Delta Optimization:

In both unigram and bigram models, the script finds the optimal delta value that minimizes the perplexity using a grid search approach. The lower the perplexity, the better the model.
### Poet-Specific Analysis:

The code evaluates the perplexity for individual poets, which can help to understand how well the model performs on specific styles or datasets.
### Statistical Measures:

Information Gain (IG) and Chi-Squared (X-Square) tests are used to determine which words are most indicative of a particular poet or group of poets.
### Use of Hazm Library:

The hazm library is specifically designed for Persian NLP, allowing efficient tokenization of Persian text.
Points to Consider:
There are some functions and variables in the code that seem to be incomplete or have inconsistent naming conventions (e.g., bi_perplexity is referenced instead of Bigram_Perplexity).
It appears that the section with the X2_list is cut off. To complete the code for calculating the top 10 words with the highest X-squared values, you'd need to iterate over the words, compute the X-squared value, and then use nlargest to get the top 10.



# Naive Bayes Bigram Classifier

This project implements a **Naive Bayes** text classification model for classifying poems into different classes. The model uses **Bigram** for text analysis and has the ability to adjust the **delta** parameter to improve results.

## Description

This code implements a **Naive Bayes Bigram Classifier** used for classifying poems into various groups (such as Iranian poets). The following code includes the main steps for training, predicting, evaluating accuracy, and calculating metrics like **Precision**, **Recall**, and **F1-Score**.

### Data Structure

The input data is taken from text files where each line contains information such as the poet's name, verses, and additional details. Each verse is presented as a pair of lines (including the text and its meaning or translation). The data is organized in a dictionary containing unigram and bigram features for each poet.

### Main Steps

1. **Model Training**:
    - Input data is read, and unigram and bigram features for each poet are extracted.
    - The model is trained using these features.
  
2. **Prediction**:
    - For each new verse pair, the model calculates the probability of it belonging to each poet class.
  
3. **Evaluation**:
    - The model is evaluated using various metrics like accuracy, precision, recall, and F1-score.
    - The **Confusion Matrix** is calculated to display the model's performance in classifying the verses.

4. **Delta Tuning**:
    - The delta parameter is used to adjust prediction probabilities to make the model more accurate.
    - The model's accuracy for various delta values is evaluated to choose the best value.

### Usage

To use the code:

1. **Prepare the Data**: Organize your data in text files following the format defined in the code.
2. **Train the Model**: Load the data and train the Naive Bayes Bigram model with the provided features.
3. **Prediction and Evaluation**: After training the model, you can use it to predict categories and evaluate its performance.

### Main Code

#### 1. `Naive_Bayes_Bigram_Classifier` Class

This class is used for training and predicting the Naive Bayes Bigram model. It contains the following functions:

- **`train(data)`**: Trains the model using input data.
- **`predict(beyts)`**: Predicts the category of a given verse pair.
- **`score(test_d)`**: Evaluates the model on test data and calculates accuracy.
- **`cal_measures(avg='macro_averaging')`**: Calculates various metrics like precision, recall, and F1-score.

#### 2. `prepare2_data` Function

This function is used to prepare the data. It reads text files and extracts unigram and bigram features. The data is stored in a dictionary with poet-specific information.

### Example

To train the model with your data and evaluate it, you can use the following code:

```bash
# Load data
train_data = prepare2_data('/path/to/train.txt', ['moulavi', 'sanaee'])
test_data = prepare2_data('/path/to/test.txt', ['moulavi', 'sanaee'])
```
# Train the model
```bash
bigrams_set = list(set(train_data['all']['bigrams']))
bnb = Naive_Bayes_Bigram_Classifier(bigrams_set, delta=0.6)
bnb.train(train_data)
```
```bash
# Evaluate the model
acc, pred = bnb.score(test_data)
print(f'Accuracy = {acc * 100:.2f}%')
print('Confusion Matrix:')
print(bnb.confusion_matrix)
```

### Tuning the Delta Parameter
To find the optimal delta value, you can test different values and calculate the model's accuracy for each one. For example:

```bash
delta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
acc_list = []
for delta in delta_list:
    bnb = Naive_Bayes_Bigram_Classifier(bigrams_set, delta=delta)
    bnb.train(train_data)
    acc, _ = bnb.score(test_data)
    acc_list.append(acc)

# Plot the results
import matplotlib.pyplot as plt
plt.plot(delta_list, [acc * 100 for acc in acc_list])
plt.xlabel('Delta')
plt.ylabel('Accuracy (%)')
plt.title('Optimal Delta for Naive Bayes Classification')
plt.grid(True)
plt.show()
```

## Prerequisites
Python 3.x

Numpy

Matplotlib

nltk

## Conclusion
This project is a simple implementation of a Naive Bayes classifier for text classification, using Bigram and different delta adjustments to improve model accuracy. You can use this code to train your own model and apply it to classify poems or other textual data.













