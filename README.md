Natural Language Processing (NLP): fake-and-real-news



Dataset:
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
-	Lemmatization: this will be used over Stemming because Lemmatization is more accurate, turning words into their true dictionary meaning, and is better suited for this assessment.



Section 2: Representation Learning
What it is:
-	Representation Learning in NLP is the process of converting the text data into “numerical vector” that can be used and understood by the algorithms. This is achieved by taking the text and applying one of the vectorisation models to it (such as Word Embedding, Contextualized Embedding, Transfer Learning, and Document Embedding).

Method used: 
-	For this assessment, a Transformer-based model will be used because it uses an “attention mechanism” to understand the context and nuance between a word and the surrounding text (both directions). This allows for more accurate NLP tasks to be performed on the Real or Fake dataset, as the model can understand the context behind each word. 
-	As the dataset is so large, with 44898 rows and contains full news articles, RoBERTa will be used, as it’s a “smarter” and better trained model compared to BERT, and is better suited for large dataset like the Real or Fake dataset. RoBERTa being a Transformer-based model also makes it better suited for this dataset compared to other models (such as, Word2Vec and TF-IDF) as it can understand the nuance and context of the of the surrounding words making it ideal for this dataset.



Section 3: Algorithms
-	For this Assessment 2 NLP Algorithms will be implemented on the Real or Fake news dataset, the 2 Algorithms will be:
MLP (Multi-Layer Perceptron): 
-	What it is: MLP is a “supervised” learning algorithm that contains connected dense layers that are transforms input data from one dimension to another (output data).
-	How it works: MLP consist of 3 layers: input layer (each node in this relate to an input feature), hidden layer (this section processes the information from the input layer), and output layer (this generates the final result / production). These layers are all fully connected (e.g., every node in one layer connects to every node in the next). To process the nodes, MLP implementation several function (e.g., Forward Propagation, Loss Function, Backpropagation, and optimization) and weighted sums to produce the output Layer.
-	Why used: MLP will be used as it is can learn complex nonlinear relationships between the data and is suited for complex dataset where understanding the context of the data is key.
SVM (Support Vector Machine): 
-	What it is: SVM is also a “supervised” learning algorithm that uses classification and regression tasks to find the hyperplane between 2 or more classes.
-	How it works: The SVM takes the group of objects / data that will be classified (e.g., cat or dog) and creates a line (hyperplane) separating them, this hyperplane is placed in the optimal position to separate the objects. To find the hyperplane the SVM uses training set that are already ladled with the correct name (e.g., this is a dog). Any point that falls within a certain distance of the hyperplane (the margin) are called supporting vectors. These support vectors are directly tied to the positioning of the Hyperplane.
-	Why used: SVM will be used as it is easy to use and simple to train on a dataset, they are also useful for tasks such as spasm mail detection which is similar to fake or real new detection. For these reasons. SVM will be used of the Fake or Real new dataset.



Section 4: Evaluation 
-	M 



Reference List: 
