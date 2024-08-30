import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from palettable.colorbrewer.qualitative import Pastel1_7
from nltk import bigrams, trigrams, word_tokenize
from nltk.corpus import stopwords
import itertools
import nltk
import logging
import os
import yaml
import boto3
from io import StringIO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# AWS Configuration
def aws_login(profile):
    """
    Logs into AWS using the specified profile.
    """
    try:
        import subprocess
        subprocess.run(['aws', 'sso', 'login', '--profile', profile], check=True)
        logging.info("AWS login successful.")
    except Exception as e:
        logging.error(f"Error during AWS login: {e}")
        raise

def create_aws_session(profile):
    """
    Creates an AWS session using the specified profile.
    """
    session = boto3.Session(profile_name=profile)
    logging.info("AWS session created.")
    return session

def get_s3_client(session):
    """
    Returns an S3 client using the provided session.
    """
    return session.client('s3')

def read_s3_csv(s3_client, bucket_name, key):
    """
    Reads a CSV file from S3.
    """
    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        logging.info(f"Data read from s3://{bucket_name}/{key} successfully.")
        return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    except Exception as e:
        logging.error(f"Error reading data from s3://{bucket_name}/{key}: {e}")
        raise

def upload_to_s3(s3_client, bucket_name, file_path, s3_key):
    """
    Uploads a file to the specified S3 bucket.
    """
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        logging.info(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logging.error(f"Error uploading {file_path} to S3: {e}")
        raise

def download_nltk_resources():
    """
    Download necessary NLTK resources.
    """
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        logging.info("NLTK resources downloaded successfully.")
    except Exception as e:
        logging.error(f"Error downloading NLTK resources: {e}")
        raise

def plot_numeric_distributions(data, config):
    """
    Plot distributions of numeric features and save the figure.
    """
    try:
        figsize = config['eda']['numeric_distributions']['figsize']
        bins = config['eda']['numeric_distributions']['bins']
        kde = config['eda']['numeric_distributions']['kde']
        color = config['eda']['numeric_distributions']['color']
        yscale = config['eda']['numeric_distributions']['yscale']

        fig, ax = plt.subplots(2, 2, figsize=figsize)

        sns.histplot(data['HelpfulnessNumerator'], bins=bins, ax=ax[0, 0], kde=kde, color=color)
        ax[0, 0].set_title('Distribution of HelpfulnessNumerator')
        ax[0, 0].set_yscale(yscale)

        sns.histplot(data['HelpfulnessDenominator'], bins=bins, ax=ax[0, 1], kde=kde, color=color)
        ax[0, 1].set_title('Distribution of HelpfulnessDenominator')
        ax[0, 1].set_yscale(yscale)

        sns.histplot(data['Score'], bins=5, ax=ax[1, 0], kde=kde, color=color)
        ax[1, 0].set_title('Distribution of Score')

        sns.scatterplot(data=data, x='Helpfulness', y='Score', ax=ax[1, 1], alpha=0.1, color=color)
        ax[1, 1].set_title('Helpfulness Ratio vs Score')

        plt.tight_layout()
        output_dir = config['eda']['plots_output_dir']
        os.makedirs(output_dir, exist_ok=True)  # Ensure the plots directory exists
        output_path = os.path.join(output_dir, 'numeric_distributions.png')
        plt.savefig(output_path)
        plt.close()
        logging.info("Numeric distributions plotted and saved successfully.")
        return output_path
    except Exception as e:
        logging.error(f"Error plotting numeric distributions: {e}")
        raise

def generate_wordclouds(data, config):
    """
    Generate word clouds for positive, negative, and neutral reviews and save the figure.
    """
    try:
        wordcloud_config = config['eda']['wordcloud']
        width = wordcloud_config['width']
        height = wordcloud_config['height']
        background_color = wordcloud_config['background_color']

        positive_reviews = data[data['Sentiment'] == 'positive']['processed_text']
        negative_reviews = data[data['Sentiment'] == 'negative']['processed_text']
        neutral_reviews = data[data['Sentiment'] == 'neutral']['processed_text']

        wordcloud_pos = WordCloud(width=width, height=height, background_color=background_color).generate(' '.join(positive_reviews))
        wordcloud_neg = WordCloud(width=width, height=height, background_color=background_color).generate(' '.join(negative_reviews))
        wordcloud_neu = WordCloud(width=width, height=height, background_color=background_color).generate(' '.join(neutral_reviews))

        plt.figure(figsize=(15, 10))

        plt.subplot(1, 3, 1)
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis('off')
        plt.title('Positive Reviews')

        plt.subplot(1, 3, 2)
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis('off')
        plt.title('Negative Reviews')

        plt.subplot(1, 3, 3)
        plt.imshow(wordcloud_neu, interpolation='bilinear')
        plt.axis('off')
        plt.title('Neutral Reviews')

        output_dir = config['eda']['plots_output_dir']
        os.makedirs(output_dir, exist_ok=True)  # Ensure the plots directory exists
        output_path = os.path.join(output_dir, 'wordclouds.png')
        plt.savefig(output_path)
        plt.close()
        logging.info("Word clouds generated and saved successfully.")
        return output_path
    except Exception as e:
        logging.error(f"Error generating word clouds: {e}")
        raise

def plot_unique_words(data, config):
    """
    Plot top unique positive and negative words and save the figure.
    """
    try:
        top_n = config['eda']['top_words']['top_n']
        positive_reviews = data[data['Sentiment'] == 'positive']['processed_text']
        negative_reviews = data[data['Sentiment'] == 'negative']['processed_text']

        positive_word_counts = Counter(' '.join(positive_reviews).lower().split())
        negative_word_counts = Counter(' '.join(negative_reviews).lower().split())
        top_positive = dict(positive_word_counts.most_common(top_n))
        top_negative = dict(negative_word_counts.most_common(top_n))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        wedges, texts, autotexts = ax1.pie(
            top_positive.values(),
            labels=top_positive.keys(),
            colors=Pastel1_7.hex_colors,
            autopct='%1.1f%%',
            startangle=90,
            counterclock=False,
            wedgeprops={'width': 0.3}
        )
        plt.setp(autotexts, size=8, weight="bold")
        ax1.set_title('Top 10 Unique Positive Words')

        wedges, texts, autotexts = ax2.pie(
            top_negative.values(),
            labels=top_negative.keys(),
            colors=Pastel1_7.hex_colors,
            autopct='%1.1f%%',
            startangle=90,
            counterclock=False,
            wedgeprops={'width': 0.3}
        )
        plt.setp(autotexts, size=8, weight="bold")
        ax2.set_title('Top 10 Unique Negative Words')

        output_dir = config['eda']['plots_output_dir']
        os.makedirs(output_dir, exist_ok=True)  # Ensure the plots directory exists
        output_path = os.path.join(output_dir, 'unique_words_donut_plots.png')
        plt.savefig(output_path)
        plt.close()
        logging.info("Unique words donut plots generated and saved successfully.")
        return output_path
    except Exception as e:
        logging.error(f"Error plotting unique words: {e}")
        raise

def plot_bigrams(data, config):
    """
    Plot top positive and negative bigrams and save the figure.
    """
    try:
        n = config['eda']['ngram']['n']
        positive_reviews = data[data['Sentiment'] == 'positive']['processed_text']
        negative_reviews = data[data['Sentiment'] == 'negative']['processed_text']

        positive_bigrams = process_text(positive_reviews, n_gram=n)
        negative_bigrams = process_text(negative_reviews, n_gram=n)

        positive_counts = Counter(positive_bigrams)
        negative_counts = Counter(negative_bigrams)
        top_positive = dict(positive_counts.most_common(10))
        top_negative = dict(negative_counts.most_common(10))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        wedges, texts, autotexts = ax1.pie(
            top_positive.values(),
            labels=[f'{x[0]} {x[1]}' for x in top_positive.keys()],
            colors=Pastel1_7.hex_colors,
            autopct='%1.1f%%',
            startangle=90,
            counterclock=False,
            wedgeprops={'width': 0.3}
        )
        plt.setp(autotexts, size=8, weight="bold")
        ax1.set_title('Top 10 Positive Bigrams')

        wedges, texts, autotexts = ax2.pie(
            top_negative.values(),
            labels=[f'{x[0]} {x[1]}' for x in top_negative.keys()],
            colors=Pastel1_7.hex_colors,
            autopct='%1.1f%%',
            startangle=90,
            counterclock=False,
            wedgeprops={'width': 0.3}
        )
        plt.setp(autotexts, size=8, weight="bold")
        ax2.set_title('Top 10 Negative Bigrams')

        output_dir = config['eda']['plots_output_dir']
        os.makedirs(output_dir, exist_ok=True)  # Ensure the plots directory exists
        output_path = os.path.join(output_dir, 'bigrams_donut_plots.png')
        plt.savefig(output_path)
        plt.close()
        logging.info("Bigrams donut plots generated and saved successfully.")
        return output_path
    except Exception as e:
        logging.error(f"Error plotting bigrams: {e}")
        raise

def process_text(texts, n_gram=2):
    """
    Tokenize, create bigrams/trigrams, and filter out stopwords.
    """
    stop_words = set(stopwords.words('english'))
    try:
        if n_gram == 2:
            return list(itertools.chain(*[
                bigrams([word for word in word_tokenize(text.lower()) if word.isalpha() and word not in stop_words])
                for text in texts
            ]))
        elif n_gram == 3:
            return list(itertools.chain(*[
                trigrams([word for word in word_tokenize(text.lower()) if word.isalpha() and word not in stop_words])
                for text in texts
            ]))
    except Exception as e:
        logging.error(f"Error processing text for {n_gram}-grams: {e}")
        raise

if __name__ == "__main__":
    try:
        # Load configuration
        with open('config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        aws_login(config['aws']['profile'])
        session = create_aws_session(config['aws']['profile'])
        s3_client = get_s3_client(session)
        data = read_s3_csv(s3_client, config['aws']['s3']['bucket_name'], config['aws']['s3']['data_key'])
        
        download_nltk_resources()
        
        plots = []
        plots.append(plot_numeric_distributions(data, config))
        plots.append(generate_wordclouds(data, config))
        plots.append(plot_unique_words(data, config))
        plots.append(plot_bigrams(data, config))

        for plot in plots:
            upload_to_s3(s3_client, config['aws']['s3']['bucket_name'], plot, f"{config['output']['eda_plots_path']}/{os.path.basename(plot)}")
    except Exception as e:
        logging.error(f"Error in the main execution: {e}")