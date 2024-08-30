#!/usr/bin/env python
# coding: utf-8

import yaml
import pandas as pd
import boto3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from numpy import asarray, zeros
from io import StringIO
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Flatten, Dense, Conv1D, GlobalMaxPooling1D, LSTM
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import os
import pickle

# Load configuration
with open('../config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# AWS Configuration
def aws_login(profile):
    try:
        import subprocess
        subprocess.run(['aws', 'sso', 'login', '--profile', profile], check=True)
    except Exception as e:
        print(f"Error during AWS login: {e}")
        raise

def create_aws_session(profile):
    session = boto3.Session(profile_name=profile)
    return session

def get_s3_client(session):
    return session.client('s3')

def read_s3_csv(s3_client, bucket_name, key):
    obj = s3_client.get_object(Bucket=bucket_name, Key=key)
    return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')), parse_dates=['Time'])

def encode_sentiment(df):
    label_encoder = LabelEncoder()
    df['Sentiment_encoded'] = label_encoder.fit_transform(df['Sentiment'])
    return df

def balance_dataset(df):
    df_positive = df[df['Sentiment'] == 'positive']
    df_negative = df[df['Sentiment'] == 'negative']
    df_neutral = df[df['Sentiment'] == 'neutral']
    min_count = min(len(df_positive), len(df_negative), len(df_neutral))
    df_balanced = pd.concat([
        df_positive.sample(n=min_count, random_state=42),
        df_negative.sample(n=min_count, random_state=42),
        df_neutral.sample(n=min_count, random_state=42)
    ])
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

def tokenize_and_pad(X_train, X_test, maxlen):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    return X_train, X_test, tokenizer.word_index, tokenizer

def load_glove_embeddings(filepath, vocab_length, word_index, embedding_dim=100):
    embeddings_dictionary = {}
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GloVe embeddings file not found: {filepath}")
    
    with open(filepath, encoding="utf8") as glove_file:
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions
    embedding_matrix = zeros((vocab_length, embedding_dim))
    for word, index in word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix

def build_snn_model(vocab_length, maxlen, embedding_matrix, num_classes):
    model = Sequential()
    layer = Embedding(input_dim=vocab_length, output_dim=100, trainable=False, input_shape=(maxlen,))
    layer.build((None,))
    layer.set_weights([embedding_matrix])
    model.add(layer)
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model(vocab_length, maxlen, embedding_matrix, num_classes):
    model = Sequential()
    layer = Embedding(input_dim=vocab_length, output_dim=100, trainable=False, input_shape=(maxlen,))
    layer.build((None,))
    layer.set_weights([embedding_matrix])
    model.add(layer)
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(vocab_length, maxlen, embedding_matrix, num_classes):
    model = Sequential()
    layer = Embedding(input_dim=vocab_length, output_dim=100, trainable=False, input_shape=(maxlen,))
    layer.build((None,))
    layer.set_weights([embedding_matrix])
    model.add(layer)
    model.add(LSTM(128))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_performance(history, metric='accuracy'):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'Model {metric}')
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    try:
        aws_login(config['aws']['profile'])
        session = create_aws_session(config['aws']['profile'])
        s3_client = get_s3_client(session)
        bucket_name = config['aws']['s3']['bucket_name']
        data_key = config['aws']['s3']['data_key']
        df = read_s3_csv(s3_client, bucket_name, data_key)
        
        df = encode_sentiment(df)
        if config['data']['balance_classes']:
            df_balanced = balance_dataset(df)
        else:
            df_balanced = df

        X = df_balanced['processed_text']
        y = df_balanced['Sentiment_encoded']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['data']['test_size'], random_state=config['data']['random_state'], stratify=y)
        
        X_train, X_test, word_index, tokenizer = tokenize_and_pad(X_train, X_test, config['data']['max_sequence_length'])
        
        # Save tokenizer
        with open('../models/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)

        vocab_length = len(word_index) + 1
        embedding_matrix = load_glove_embeddings(config['data']['glove_embeddings_path'], vocab_length, word_index)
        
        num_classes = df_balanced['Sentiment_encoded'].nunique()

        # Simple Neural Network
        snn_model = build_snn_model(vocab_length, config['data']['max_sequence_length'], embedding_matrix, num_classes)
        snn_model_history = snn_model.fit(X_train, y_train, epochs=config['model']['epochs'], validation_data=(X_test, y_test))
        snn_model.save(config['output']['snn_model_path'])
        plot_performance(snn_model_history, 'accuracy')
        plot_performance(snn_model_history, 'loss')

        # Convolutional Neural Network
        cnn_model = build_cnn_model(vocab_length, config['data']['max_sequence_length'], embedding_matrix, num_classes)
        cnn_model_history = cnn_model.fit(X_train, y_train, batch_size=config['model']['batch_size'], epochs=config['model']['epochs'], verbose=1, validation_data=(X_test, y_test))
        cnn_model.save(config['output']['cnn_model_path'])
        plot_performance(cnn_model_history, 'accuracy')
        plot_performance(cnn_model_history, 'loss')

        # Recurrent Neural Network (LSTM)
        lstm_model = build_lstm_model(vocab_length, config['data']['max_sequence_length'], embedding_matrix, num_classes)
        lstm_model_history = lstm_model.fit(X_train, y_train, batch_size=config['model']['batch_size'], epochs=config['model']['epochs'], verbose=1, validation_data=(X_test, y_test))
        lstm_model.save(config['output']['lstm_model_path'])
        plot_performance(lstm_model_history, 'accuracy')
        plot_performance(lstm_model_history, 'loss')
    
    except Exception as e:
        print(f"Error: {e}")
