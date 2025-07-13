# Fake News Classification App

A machine learning-based web application that detects whether a news statement is likely to be reliable or fake. It combines real-world data scraping, NLP preprocessing, and classification techniques to provide accurate predictions. An optional verification layer is also included for ambiguous inputs.

## Overview

The application addresses the growing challenge of online misinformation. It scrapes fact-checked political claims from trusted sources, processes the text, and trains a supervised classification model to distinguish between real and fake statements. For cases that may not align with the training distribution, the app integrates a lightweight content verification API to provide supporting insights.

## Features

- Scrapes political news claims and fact-checks from [Politifact.com](https://www.politifact.com/)
- Cleans and processes data using stemming, stopword removal, and regex filtering
- Converts text to vector form using TF-IDF
- Trains a classification model to detect misinformation
- Streamlit-based UI for user interaction and prediction
- Optional secondary verification for unrecognized statements via external API

## Tech Stack

- **Language:** Python
- **Scraping:** BeautifulSoup
- **NLP:** NLTK, regex
- **Modeling:** scikit-learn (TF-IDF + classifier)
- **Web Interface:** Streamlit
- **Environment Management:** `dotenv`

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sohampattankude/fake-news-classifier.git
cd fake-news-classifier
```   

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```   
### 3. Configure Environment Variables

Create a `.env` file in the root directory with the following content:

```ini
GEMINI_API_KEY=your_api_key_here
```

### 4. Launch the App

```bash
streamlit run app.py
```

### Project Structure

```bash
├── app.py                       # Streamlit application
├── Scraper.ipynb                # Scrapes data from Politifact
├── model.pkl                    # Trained ML model
├── vector.pkl                   # TF-IDF vectorizer
├── politifact_scraped_data.csv # Dataset built from scraping
├── background.jpg               # Custom UI background
├── requirements.txt
├── .env                         # API key (ignored in .gitignore)
└── .gitignore
```

### How It Works

- **Scraping**: Extract fact-checked statements and labels from Politifact  
- **Preprocessing**: Apply NLP transformations to clean the data  
- **Modeling**: Train a classification model using TF-IDF features  
- **Prediction**: User inputs are classified via the model  
- **Optional Validation**: For unfamiliar claims, external verification offers additional insight

