Natural Language Processing (NLP): fake-and-real-news

By Matthew Scott (2406934)



“Generative AI was not used to support completion of this assessment. If used: The GenAI tool [ChatGPT] was used for the purpose of [ideation/editing]. Where used for the purpose of supporting development, comments have been provided against relevant code cells.”



Links:
Link to GitHub: NLP-Fake-and-Real-News/README.md at main · Matty024/NLP-Fake-and-Real-News
Link to dataset: fake-and-real-news-dataset



Section 1: Dataset
About:
-	The dataset (“fake-and-real-news-dataset”) was found on Kaggle and was uploaded by clmentbisaillon on the 19th of April 2024 and contains 2 CSV files: “Fake.csv”, which contains 23502 fake news articles, and another “True.csv”, which contains 21417 real news articles. Both CSV files have 4 columns: title (name of the article), text (information covered), subject (what it is), and date.
-	This dataset will be used to determine if a news article is real or fake.
Pre-processing: 
-	Combining the datasets: the dataset will need to be combined into 1 CSV (“TFdata”). Both files have the same columns, they will be combined, with a new column (“Real or Fake”) being added to identify if the row is Fake (0) or Real (1).
-	Unbalanced: the dataset is slightly unbalanced, with it leaning towards Fack (52.3 to 47.7). This imbalance is too small and won't skew the outcome.
-	Normalization: the text column will be normalized.
-	Tokenization: the normalized text will be converted into tokens of 1 term.
-	Stop Words: any stop word found in the tokenized text will be removed. 
-	Lemmatization: This will be used over Stemming because Lemmatization is more accurate, turning words into their true dictionary meaning, and is better suited for this assessment.



Section 2: Representation Learning
What it is:
-	Representation Learning in NLP is the process of converting the text data into “numerical vector” that can be used and understood by the algorithms. This is achieved by taking the text and applying one of the vectorisation models to it (such as Word Embedding, Contextualized Embedding, Transfer Learning, and Document Embedding).

Method used: 
-	For this assessment, TF-IDF (Term Frequency-Inverse Document Frequency) will be used. TF-IDF is a “statistical” method used to determine how important a word is to a document. This is achieved by first the TF (how often a word appears, higher the frequency = the more important) and second the IDF (reduces the weight of common words (‘and’) and increases the weights of rare words (‘cat’)). Using both of these steps, TF-IDF can be used in text classification, search rankings, and other tasks.
-	As the dataset is so large, with 44898 rows and contains full news articles, TF-IDF is the best choice as it does not require huge RAM or CPU to run and is very easy to implement into the assessment without sacrificing the accuracy of the models, as TF-IDF is often used for text classification (fake or real news).



Section 3: Algorithms
-	For this Assessment 2, NLP Algorithms will be implemented on the Real or Fake news dataset, the 2 Algorithms will be:
MLP (Multi-Layer Perceptron): 
-	What it is: MLP is a “supervised” learning algorithm that contains connected dense layers that transform input data from one dimension to another (output data).
-	How it works: MLP consists of 3 layers: input layer (each node in this relates to an input feature), hidden layer (this section processes the information from the input layer), and output layer (this generates the final result/production). These layers are all fully connected (e.g., every node in one layer connects to every node in the next). To process the nodes, MLP implementation involves several functions (e.g., Forward Propagation, Loss Function, Backpropagation, and optimization) and weighted sums to produce the output Layer.
-	Why used: MLP will be used as it can learn complex nonlinear relationships between the data and is suited for complex datasets where understanding the context of the data is key.
SVM (Support Vector Machine): 
-	What it is: SVM is also a “supervised” learning algorithm that uses classification and regression tasks to find the hyperplane between 2 or more classes.
-	How it works: The SVM takes the group of objects / data that will be classified (e.g., cat or dog) and creates a line (hyperplane) separating them, this hyperplane is placed in the optimal position to separate the objects. To find the hyperplane, the SVM uses training sets that are already labelled with the correct name (e.g., this is a dog). Any point that falls within a certain distance of the hyperplane (the margin) is called supporting vectors. These support vectors are directly tied to the positioning of the Hyperplane.
-	Why used: SVM will be used as it is easy to use and simple to train on a dataset; they are also useful for tasks such as spam mail detection, which is like fake or real news detection. For these reasons. SVM will be used on the Fake or Real new dataset.



Section 4: Evaluation 
Evaluation Process:
-	To evaluate the performance of the 2 algorithms (MLP/SVM) and ensure the results can be reproduced, the data was split into testing and training sets, with the training set using 80% and the testing set using 20% of the dataset. Random_state was also used. To evaluate the algorithms, a Classification report will be produced (giving information about its accuracy, F1-score, and performance). The algorithms will also be evaluated based on how long they took to compute.
Analyse Results:
-	Based on the Results of the 2 algorithms: MLP has an accuracy of 0.98 (98%) and an F1-score of 0.98, and SVM has an accuracy of 0.99 (99%) and an F1-score of 0.99. Based on this, there is not much of a difference between the 2, with their only being a 0.01 difference between the 2, the only major difference being how long each one takes to compute, with MLP taking 10 or more minutes and SVM taking around 3 minutes. Based on that, I would recommend SVM as it has the faster computing time and slightly higher scores all around.



Reference List: 
GeeksforGeeks (2021). Normalizing Textual Data with Python. [online] GeeksforGeeks. Available at: https://www.geeksforgeeks.org/python/normalizing-textual-data-with-python/.
 GeeksforGeeks (2018). Python | Lemmatization with NLTK. [online] GeeksforGeeks. Available at: https://www.geeksforgeeks.org/python/python-lemmatization-with-nltk/.
 Google.com. (2019). Google Colab. [online] Available at: https://colab.research.google.com/drive/1Oq4ZUC3dW_W15-i-N0cVwAMryjAshrAe?usp=sharing#scrollTo=ESd-lQV9T49j [Accessed 20 Nov. 2025].
 GeeksforGeeks (2024). 5 Simple Ways to Tokenize Text in Python. [online] GeeksforGeeks. Available at: https://www.geeksforgeeks.org/nlp/5-simple-ways-to-tokenize-text-in-python/.
 GeeksforGeeks (2021). Understanding TFIDF (Term FrequencyInverse Document Frequency). [online] GeeksforGeeks. Available at: https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/.
 Python, R. (n.d.). Split Your Dataset With scikit-learn’s train_test_split() – Real Python. [online] realpython.com. Available at: https://realpython.com/train-test-split-python-data/.
 Acharya, A. (2023). Training, Validation, Test Split for Machine Learning Datasets. [online] encord.com. Available at: https://encord.com/blog/train-val-test-split/.
 CodeSignal Learn. (2015). Transforming Text into Insights: An Introduction to TF-IDF Vectorization in Python. [online] Available at: https://codesignal.com/learn/courses/introduction-to-tf-idf-vectorization-in-python/lessons/transforming-text-into-insights-an-introduction-to-tf-idf-vectorization-in-python.
 Navlani, A. (2019). Scikit-learn SVM Tutorial with Python (Support Vector Machines). [online] www.datacamp.com. Available at: https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python.
 scikit-learn (2025). 1.17. Neural Network Models (supervised) — scikit-learn 0.23.1 Documentation. [online] scikit-learn.org. Available at: https://scikit-learn.org/stable/modules/neural_networks_supervised.html.
 GeeksforGeeks (2021). MultiLayer Perceptron Learning in Tensorflow. [online] GeeksforGeeks. Available at: https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/.
 GeeksforGeeks (2018). Python | Lemmatization with NLTK. [online] GeeksforGeeks. Available at: https://www.geeksforgeeks.org/python/python-lemmatization-with-nltk/.
 GeeksforGeeks (2017). Removing stop words with NLTK in Python. [online] GeeksforGeeks. Available at: https://www.geeksforgeeks.org/nlp/removing-stop-words-nltk-python/.
 kaggle.com. (n.d.). Tokenization in NLP. [online] Available at: https://www.kaggle.com/code/satishgunjal/tokenization-in-nlp.
 Stack Overflow. (n.d.). How is the hidden layer size determined for MLPRegressor in SciKitLearn? [online] Available at: https://stackoverflow.com/questions/55786860/how-is-the-hidden-layer-size-determined-for-mlpregressor-in-scikitlearn.
 GeeksforGeeks (2021). MultiLayer Perceptron Learning in Tensorflow. [online] GeeksforGeeks. Available at: https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/.
 GeeksforGeeks (2021). Support Vector Machine (SVM) Algorithm. [online] GeeksforGeeks. Available at: https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/.
 www.youtube.com. (n.d.). Support Vector Machine (SVM) in 2 minutes. [online] Available at: https://www.youtube.com/watch?v=_YPScrckx28.
 GR, V.K. (2024). Hey everyone! 👋Welcome back to our NLP journey! 🎉 Today, we’re going to dive into another essential text-processing technique: Normalization. Imagine you have a bunch of books, but some are in Spanish, some in French, and others in English. [online] Linkedin.com. Available at: https://www.linkedin.com/pulse/day-8-normalization-standardizing-text-nlp-vinod-kumar-g-r-mqdsc [Accessed 20 Nov. 2025].
 Data Professor (2021). How to handle imbalanced datasets in Python. [online] YouTube. Available at: https://www.youtube.com/watch?v=4SivdTLIwHc [Accessed 31 Dec. 2024].
 ChatGPT. (2025). ChatGPT - Code evaluation. [online] Available at: https://chatgpt.com/share/691f7afe-e084-8010-aaa1-d606dfef42df [Accessed 20 Nov. 2025].
 ChatGPT. (2025). ChatGPT - Convert numbers to words. [online] Available at: https://chatgpt.com/share/691f7c29-b67c-8010-8a07-be701fee7f59 [Accessed 20 Nov. 2025].
 ChatGPT. (2025). ChatGPT - NLP assessment breakdown. [online] Available at: https://chatgpt.com/share/691f7e23-c85c-8010-ac74-80a7c0ce3d8e [Accessed 20 Nov. 2025].
 ChatGPT. (2017). ChatGPT - Assessment breakdown NLP. [online] Available at: https://chatgpt.com/share/691f7e7e-f418-8010-86ee-f56ea3136452 [Accessed 20 Nov. 2025].
