# 📧 Spam/Ham Email Classifier

This project is a complete pipeline to classify emails as **spam** or **ham (non-spam)** using traditional machine learning techniques. It uses the public SpamAssassin dataset and builds a logistic regression model via Scikit-learn pipelines.

---

## 🚀 Features

- 📥 Downloads and extracts SpamAssassin email dataset
- 📄 Parses and cleans raw email content (text or HTML)
- 🔤 Converts emails into word count vectors
- 🧠 Trains a logistic regression model using Scikit-learn
- 📊 Evaluates with precision and recall metrics
- 🔌 Easily extendable via Scikit-learn transformers and pipelines

---

## 🗂️ Project Structure

```
spam\_classifier/
├── data/                      # Data fetching and storage
│   └── fetch\_data.py
├── preprocessing/            # Email loading, parsing, and transformation
│   ├── email\_loader.py
│   ├── transformers.py
│   └── utils.py
├── model/                    # Model training and evaluation
│   ├── train.py
│   └── evaluate.py
├── pipeline/                 # Preprocessing pipeline using Scikit-learn
│   └── pipeline.py
├── notebooks/                # (Optional) Jupyter notebooks for exploration
│   └── exploration.ipynb
├── main.py                   # Entry point: end-to-end run
├── requirements.txt          # Project dependencies
└── README.md

```

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/spam-ham-classifier.git
cd spam-ham-classifier
```


```bash
pip install -r requirements.txt
```

### 3. Download NLTK resources

```python
import nltk
nltk.download('punkt')
```

---

## 🧪 Run the Pipeline

```bash
python main.py
```

This will:

* Download and extract spam/ham datasets
* Parse and preprocess emails
* Vectorize content into sparse matrix
* Train a logistic regression classifier
* Print precision and recall on test set

---

## 📊 Example Output

```
['github.com', 'https://youtu.be/7Pq-S557XQU?t=3m32s']
Precision: 96.88%
Recall: 97.89%
```

---

## 📚 Dataset

* Source: [SpamAssassin Public Corpus](http://spamassassin.apache.org/publiccorpus/)
* Includes:

  * `easy_ham` – non-spam emails
  * `spam` – spam emails

---

## 🛠️ Technologies Used

* Python 3.8+
* Scikit-learn
* NLTK
* URLExtract
* NumPy
* Email parsing libraries


## 🤝 Contribution

PRs and ideas are welcome. If you'd like to contribute, please open an issue first to discuss your proposal.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Rasheek Salhan — [LinkedIn](https://www.linkedin.com/in/rasheek16)

