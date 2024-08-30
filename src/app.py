import streamlit as st
import tensorflow as tf
import numpy as np
import os
import pickle
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
import pandas as pd
import nltk

import time

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By

from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException

# Define the path to the models directory and tokenizer
models_dir = os.path.join(os.path.dirname(__file__), '/Users/yanwang/Downloads/Study/24Spring/423_Cloud_Engineering_for_Data_Science/project/web')
tokenizer_path = os.path.join(models_dir, 'tokenizer.pkl')

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def add_sentiment(score):
    if score in [1, 2]:
        return 'negative'
    elif score == 3:
        return 'neutral'
    elif score in [4, 5]:
        return 'positive'
    else:
        return 'unknown'

# Load models
@st.cache_resource
def load_models():
    try:
        cnn_model = tf.keras.models.load_model(os.path.join(models_dir, 'cnn_model.h5'))
        lstm_model = tf.keras.models.load_model(os.path.join(models_dir, 'lstm_model.h5'))
        snn_model = tf.keras.models.load_model(os.path.join(models_dir, 'snn_model.h5'))
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return cnn_model, lstm_model, snn_model, tokenizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

cnn_model, lstm_model, snn_model, tokenizer = load_models()

def preprocess_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text().lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    normalization_dict = {
        'im': "i am", 'u': "you", 'ur': "your", 'r': "are",
        'dont': "do not", 'wont': "will not", 'cant': "cannot",
        'thx': "thanks", 'pls': "please", 'lol': "laugh out loud", 'btw': "by the way"
    }
    normalized_tokens = [normalization_dict.get(word, word) for word in tokens]
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in normalized_tokens if word not in stop_words and word.isalpha() and len(word) > 2]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_tokens)

def scrape_reviews(url):
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument('--start-fullscreen')
    chrome_options.add_argument('--single-process')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    try:
        # Switch to recent reviews
        dropdown_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'cm-cr-sort-dropdown'))
        )
        driver.execute_script("arguments[0].click();", dropdown_element)
        option_to_select = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//select[@id='cm-cr-sort-dropdown']/option[@value='recent']"))
        )
        option_to_select.click()
        
        # Wait for the reviews to load
        WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.CLASS_NAME, 'review')))

        time.sleep(5)  # Add a delay to ensure the page is fully loaded
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        reviews = soup.find_all(class_='review')
        review_data = []

        # Check if the product title exists
        product_name_element = soup.find(id='productTitle')
        product_name = product_name_element.get_text().strip() if product_name_element else "Unknown Product"

        for review in reviews:
            # Check if the rating element exists
            rating_element = review.find(attrs={"data-hook": "review-star-rating"})
            if rating_element:
                star_rating_text = rating_element.find('span').get_text()
                star_rating = float(re.search(r'\d+.\d+', star_rating_text).group())  # Extract the numeric part
            else:
                continue  # Skip this review if rating is not found

            # Check if the review body exists
            review_text_element = review.find(attrs={"data-hook": "review-body"})
            review_text = review_text_element.get_text() if review_text_element else ""

            review_data.append({'Product Name': product_name, 'Star Rating': star_rating, 'Review Text': review_text})

        for review in review_data:
            review['Star Rating'] = add_sentiment(review['Star Rating'])

        for review in review_data:
            review['Review Text'] = preprocess_text(unidecode(review['Review Text']))

        return review_data

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        driver.quit()

def determine_sentiment(predictions):
    sentiment_labels = ['negative', 'neutral', 'positive']
    return [sentiment_labels[np.argmax(prediction)] for prediction in predictions]

def predict(model, data):
    sequences = tokenizer.texts_to_sequences([data])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    prediction = model.predict(padded_sequences)
    return prediction.tolist()

def main():
    # Session state to store scraped reviews
    session_state = st.session_state

    st.title("Amazon Review Sentiment Analysis")

    # Sidebar to display scraped data
    st.sidebar.title("Scraped Reviews")

    # User input for URL
    url = st.text_input("Enter the Amazon URL:", "")
    
    if st.button("Scrape Reviews"):
        if url:
            st.info("Scraping reviews...")

            # Scrape reviews
            reviews = scrape_reviews(url)

            # Display results
            if reviews:
                df = pd.DataFrame(reviews)
                session_state.df = df  # Store scraped reviews in session state
                # Display product name
                #st.sidebar.write(f"Product Name: {reviews[0]['Product Name']}")
                # Display star ratings and reviews in dataframe format
                #st.sidebar.write(df[['Star Rating', 'Review Text']])
            else:
                st.error("No reviews found.")
        else:
            st.warning("Please enter a valid URL.")
    
    if 'df' in session_state:
        # Display product name
        st.sidebar.write(f"Product Name: {session_state.df['Product Name'].iloc[0]}")
        # Display star ratings and reviews in dataframe format
        st.sidebar.write(session_state.df[['Star Rating', 'Review Text']])
        #st.sidebar.write("Scraped Reviews:")
        #st.sidebar.write(session_state.df[['Star Rating', 'Review Text']])
            
    # Model selection buttons
    st.subheader("Select Model for Sentiment Analysis")
    model_choice = st.radio("Choose a Model:", ("CNN Model", "LSTM Model", "SNN Model"))

    # Check if a model is selected before predicting
    if st.button("Predict"):
        model = None
        if model_choice == "CNN Model":
            model = cnn_model
        elif model_choice == "LSTM Model":
            model = lstm_model
        elif model_choice == "SNN Model":
            model = snn_model

        if model is not None:
            if 'df' not in session_state or session_state.df is None:
                st.warning("Please scrape reviews before predicting sentiment.")
            else:
                text_list = session_state.df['Review Text'].tolist()
                
                # Create progress bar
                progress_bar = st.progress(0)
                total_reviews = len(text_list)
                
                with st.spinner("Predicting sentiments..."):
                    predictions = []
                    for i, text in enumerate(text_list):
                        prediction = predict(model, text)
                        predictions.append(prediction[0])
                        progress = (i + 1) / total_reviews
                        progress_bar.progress(progress)
                    
                # Determine sentiment labels
                sentiments = determine_sentiment(predictions)

                # Adding the predictions and sentiments to the DataFrame
                session_state.df['Predicted Sentiment'] = sentiments

                # Display predicted sentiments
                st.write("Predicted Sentiments:")
                st.write(session_state.df[['Review Text', 'Predicted Sentiment']])

                # Count sentiment categories
                sentiment_counts = session_state.df['Predicted Sentiment'].value_counts()
                
                # Make recommendation based on majority sentiment
                if 'positive' in sentiment_counts:
                    positive_count = sentiment_counts['positive']
                else:
                    positive_count = 0

                if 'negative' in sentiment_counts:
                    negative_count = sentiment_counts['negative']
                else:
                    negative_count = 0

                if 'neutral' in sentiment_counts:
                    neutral_count = sentiment_counts['neutral']
                else:
                    neutral_count = 0
                
                if positive_count > negative_count:
                    st.write(f"<span style='font-size: 20px;'>There are {positive_count} positive, {negative_count} negative, and {neutral_count} neutral reviews, so we would recommend this product</span>.", unsafe_allow_html=True)
                elif negative_count > positive_count:
                    st.write(f"<span style='font-size: 20px;'>There are {positive_count} positive, {negative_count} negative, and {neutral_count} neutral reviews, so we would not recommend this product</span>.", unsafe_allow_html=True)
                else:
                    st.write(f"<span style='font-size: 20px;'>There are equal positive and negative reviews ({positive_count} each), so it's hard to make a recommendation</span>.", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
