#!/usr/bin/env python
# coding: utf-8

import yaml
import boto3
import os
import logging
import subprocess

# Set up logging
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

def run_eda_and_upload(s3_client, bucket_name):
    """
    Runs the EDA script and uploads the generated plots to S3.
    """
    try:
        subprocess.run(['python', 'src/analysis.py'], check=True)
        logging.info("EDA process completed successfully.")
        
        # Upload generated plots to S3
        plots = ['numeric_distributions.png', 'wordclouds.png', 'unique_words_donut_plots.png', 'bigrams_donut_plots.png']
        for plot in plots:
            upload_to_s3(s3_client, bucket_name, os.path.join('plots', plot), f'plots/{plot}')
    except Exception as e:
        logging.error(f"Error during EDA process: {e}")
        raise

if __name__ == "__main__":
    try:
        aws_login(config['aws']['profile'])
        session = create_aws_session(config['aws']['profile'])
        s3_client = get_s3_client(session)
        bucket_name = config['aws']['s3']['bucket_name']

        # Run EDA and upload plots
        run_eda_and_upload(s3_client, bucket_name)

        # Upload models
        upload_to_s3(s3_client, bucket_name, 'src/nn_modeling.py', 'nn_modeling.py')
        upload_to_s3(s3_client, bucket_name, 'models/cnn_model.h5', 'cnn_model.h5')
        upload_to_s3(s3_client, bucket_name, 'models/lstm_model.h5', 'lstm_model.h5')
        upload_to_s3(s3_client, bucket_name, 'models/snn_model.h5', 'snn_model.h5')
        upload_to_s3(s3_client, bucket_name, 'models/tokenizer.pkl', 'tokenizer.pkl')
        
    except Exception as e:
        logging.error(f"Error: {e}")
