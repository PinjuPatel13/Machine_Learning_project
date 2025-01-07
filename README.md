# Machine Learning Projects

This repository contains various machine learning projects aimed at solving real-world problems using different techniques. The projects showcase different applications such as customer churn prediction, email spam classification, sales forecasting, personalized recommendations, and more. Each project applies different algorithms and methods in the domain of machine learning, and each folder contains the complete code and data necessary to replicate the results.

## Table of Contents
- [Brand Reputation Management](#brand-reputation-management)
- [Customer Churn Prediction](#customer-churn-prediction)
- [Email Spam Classifier](#email-spam-classifier)
- [Healthcare](#healthcare)
- [Intelligent Sales Forecasting System](#intelligent-sales-forecasting-system)
- [Personalized E-Learning Recommendation System](#personalized-e-learning-recommendation-system)
- [Movie Recommender System](#movie-recommender-system)
  

---

# **Brand Reputation Management**

## **Overview**
The **Brand Reputation Management** project uses sentiment analysis to monitor and manage the reputation of a brand. It analyzes customer sentiments from online reviews, social media posts, and forums. This helps businesses gauge customer satisfaction, identify potential issues, and track brand mentions over time.

## **Objective**
- **Sentiment Analysis for Brand Reputation Management**: Build a sentiment analysis tool to classify customer feedback (positive, neutral, negative).
- **Data Extraction**: Collect customer feedback data from platforms like Amazon reviews or social media using APIs or web scraping tools.
- **Text Processing**: Clean and process the data for sentiment classification.
- **Model Development**: Train a sentiment classifier using machine learning models.
- **Visualization**: Display sentiment trends and key insights using visualizations.

## **Features**
- **Data Extraction**: 
  - Use Amazon reviews, Twitter API, or Reddit API to collect customer feedback data.
  - Employ web scraping tools such as **BeautifulSoup** or **Scrapy** to extract relevant text data.
  
- **Text Processing**: 
  - Use libraries like **NLTK** or **SpaCy** for text tokenization, stop-word removal, and stemming/lemmatization.
  - Create word embeddings using **Word2Vec** or **TF-IDF** to represent text data numerically (using **Gensim** or **scikit-learn**).

- **Model Development**: 
  - Train a sentiment classifier using models such as **Logistic Regression**, **SVM**, or deep learning models like **LSTM** (Long Short-Term Memory) with **TensorFlow** or **PyTorch**.
  - The classifier categorizes feedback into **positive**, **neutral**, or **negative** sentiments.

- **Visualization**: 
  - Visualize sentiment analysis results and track reputation trends over time.
  - Use **Plotly** or **Tableau** to generate interactive graphs and reports.

## **Requirements**
To run the project, ensure you have the following dependencies:

- Python 3.x
- **Pandas**: For data manipulation and analysis.
- **Numpy**: For numerical operations.
- **Matplotlib**: For visualizing data.
- **Seaborn**: For statistical data visualization.
- **scikit-learn**: For machine learning models and preprocessing.
- **TensorFlow**: For training deep learning models (e.g., LSTM).
- **NLTK**: For text preprocessing and analysis.
- **SpaCy**: For advanced NLP tasks.
- **Gensim**: For creating word embeddings (Word2Vec).
- **Plotly**: For interactive data visualization.
- **BeautifulSoup / Scrapy**: For web scraping (if collecting data manually).
- **Requests**: For making API calls to services like Twitter or Reddit.

## **Installation**

1. **Clone the repository**:
   First, clone this repository to your local machine:
   ```bash
   git clone https://github.com/PinjuPatel13/Brand_Reputation_Management.git
   cd Brand_Reputation_Management
   ```

2. **Create a Virtual Environment** (optional, but recommended):
   Create a virtual environment to isolate the project's dependencies:
   ```bash
   python -m venv venv
   ```

   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies**:
   Install the required libraries using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## **Running the Project**

1. **Data Extraction**:
   - If you're using **Amazon reviews**, download the dataset (e.g., CSV format) or use the **Amazon Product Review API**.
   - Alternatively, set up an API call to **Twitter** or **Reddit** to collect live data.

2. **Preprocess the Data**:
   - Run the script to preprocess the data (e.g., `preprocess.py`) to clean and prepare the text for sentiment analysis.

3. **Train the Sentiment Model**:
   - Run the script to train the sentiment analysis model:
     ```bash
     python sentiment_analysis.py
     ```

4. **Visualization**:
   - After running the model, use the visualization script (`visualization.py`) to display sentiment trends:
     ```bash
     python visualization.py
     ```

## **Sample Code**

Here’s an example of how you might perform sentiment analysis on Amazon reviews:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('amazon_reviews.csv')

# Preprocess text data
df['cleaned_review'] = df['review_text'].apply(clean_text_function)  # Define cleaning function

# Convert text to numerical data using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment_label']  # Sentiment labels: 0 = Negative, 1 = Positive

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## **Visualization Example**
Use **Plotly** or **Tableau** to visualize the sentiment trends over time. Here’s an example using **Plotly**:

```python
import plotly.express as px

# Sample data
df_sentiment = pd.DataFrame({'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
                             'positive': [50, 60, 55],
                             'negative': [30, 20, 25],
                             'neutral': [20, 20, 20]})

fig = px.line(df_sentiment, x='date', y=['positive', 'negative', 'neutral'],
              title='Sentiment Trend Over Time')

fig.show()
```

## **License**
This project is open-source and available under the [MIT License](LICENSE).

---

### Additional Notes:
- If using **Twitter API** or **Reddit API**, refer to their official documentation to set up API keys and access data.
- For **Amazon review scraping**, ensure you follow Amazon's terms of service, or use public datasets.

---

---

# **Customer Churn Prediction for SaaS Platforms**

## **Overview**
This project aims to predict customer churn for Software as a Service (SaaS) platforms by analyzing customer usage patterns, subscription data, and feedback history. The goal is to help SaaS businesses predict which customers are likely to churn, so they can take preventive measures to retain them.

## **Objective**
- **Customer Churn Prediction**: Predict customer churn based on historical data like subscription details, customer usage patterns, and feedback history.
- **Data Preprocessing**: Clean and handle missing data, encode categorical features, and normalize the data.
- **Model Development**: Train machine learning models such as **Logistic Regression**, **Random Forest**, and **XGBoost**.
- **Evaluation**: Evaluate the model's performance using accuracy, confusion matrix, and ROC curve analysis.
- **Visualization**: Use **Matplotlib** and **Seaborn** to visualize trends and insights. Deploy the model via **Streamlit** for interactive use.

## **Project Files**
- **Data**: `Telco_Customer_Churn.csv` (Contains customer data, churn history, subscription details, etc.)
- **Code**: 
  - `app.py`: Streamlit deployment script for the model.
  - `Customer_Churn_Prediction.ipynb`: Jupyter Notebook with model development, data preprocessing, and visualization.
  - `requirements.txt`: Python dependencies for the project.
  
## **Requirements**
- Python 3.x
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning model development and preprocessing.
- **XGBoost**: For building XGBoost models.
- **Matplotlib**: For creating visualizations.
- **Seaborn**: For advanced statistical visualizations.
- **Streamlit**: For creating interactive dashboards.

Install all dependencies by running:
```bash
pip install -r requirements.txt
```

## **Installation**

### Step 1: Clone the Repository
```bash
git clone https://github.com/PinjuPatel13/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

### Step 2: Create a Virtual Environment (Optional)
(Optional, but recommended for isolating dependencies)

- On **Windows**:
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

- On **macOS/Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## **How to Run the Project**

### Step 1: Prepare the Dataset
Make sure the dataset (`Telco_Customer_Churn.csv`) is available in the project directory. This dataset contains customer details, subscription history, usage data, and churn information.

### Step 2: Preprocess the Data
The data has been preprocessed in the code files (`Customer_Churn_Prediction.ipynb`), including handling missing values, encoding categorical features, and splitting the data into training and testing sets.

### Step 3: Train the Model
The code for training the model is already included in the notebook file (`Customer_Churn_Prediction.ipynb`). The models used are **Logistic Regression**, **Random Forest**, and **XGBoost**.

Run the notebook in your preferred environment (Jupyter Notebook, VSCode, or any Python IDE) to train and evaluate the models.

### Step 4: Run Streamlit App
To deploy the model with an interactive interface:

1. Run the following command to start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. This will open the app in your browser, where you can input customer data and get churn predictions in real-time.

### Example of Streamlit Interface:
- Input customer details (e.g., age, gender, etc.)
- Get churn prediction (whether the customer is likely to churn or not).

## **Visualizations**
The notebook includes visualizations like:
- **Churn Distribution**: Visualize how many customers are likely to churn vs. not.
- **Feature Importance**: Display which features are most important for predicting churn.

## **Evaluation**
- The models are evaluated using **accuracy**, **confusion matrix**, and **ROC curve analysis**.
- You can experiment with different hyperparameters and models to improve accuracy.

## **License**
This project is open-source and available under the [MIT License](LICENSE).

---

