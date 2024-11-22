# 📧 Email/SMS Spam Classifier

A simple web application that classifies emails or SMS messages as **Spam** or **Not Spam** using a machine learning model. Built using **Streamlit**, this project demonstrates how to preprocess text data, apply a trained model, and provide real-time predictions in an interactive web interface.

---

## 🚀 Features

- **Real-time classification**: Predict whether a message is spam or not instantly.
- **Text preprocessing**: Automatically cleans and processes input text to improve accuracy.
- **Interactive UI**: User-friendly interface built with Streamlit.
- **Machine learning**: Uses a trained classification model with TF-IDF vectorization for accurate predictions.

---

## 📂 Project Structure

- `app.py`: Main script containing the Streamlit web app.
- `vectorizer.pkl`: Pre-trained TF-IDF vectorizer for text feature extraction.
- `model.pkl`: Trained machine learning model for spam classification.
- `README.md`: Documentation for the project (this file).

---

## ⚙️ Installation and Setup

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or above
- Required libraries (listed in `requirements.txt`)

### Clone the Repository

```bash
git clone https://github.com/your-username/email-sms-spam-classifier.git
cd email-sms-spam-classifier
```

### Install Dependencies

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

### Download NLTK Data (if needed)

Run the following in Python to download necessary NLTK data files:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## ▶️ Running the App

1. Navigate to the project directory.
2. Start the Streamlit app:

```bash
streamlit run app.py
```

3. Open the app in your browser at `http://localhost:8501`.

---

## 🛠️ How It Works

1. **Text Preprocessing**:
   - Converts text to lowercase.
   - Tokenizes the input into words.
   - Removes stopwords and punctuation.
   - Applies stemming to reduce words to their base form.

2. **Vectorization**:
   - Converts preprocessed text into numerical features using a TF-IDF vectorizer.

3. **Classification**:
   - Uses a trained machine learning model to classify the text as **Spam** or **Not Spam**.

---

## 📊 Model Details

- **TF-IDF Vectorizer**: Extracts numerical features from text data.
- **Classification Model**: (Specify the model, e.g., Logistic Regression, Naive Bayes, etc.)
- Trained on a dataset of labeled spam and non-spam messages for high accuracy.

---

## 📌 Usage

1. Enter a message into the text box.
2. Click **Predict**.
3. View the result displayed as either **Spam** or **Not Spam**.

---

## 🛡️ Limitations

- The model's accuracy depends on the training data.
- May require retraining for different languages or datasets.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests.
- Submit pull requests to improve the project.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or support, reach out at **ayushbadoni7@gmail.com** or connect via [LinkedIn](https://linkedin.com/in/ayyushx).

---

### ✨ Happy Classifying! 🎉