import pandas as pd
import numpy as np
from ast import literal_eval
import spacy
import classy_classification
import os
import glob
import random
from transformers import pipeline
import nltk
from spacy.matcher import Matcher
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Function to read in multiple files
def get_csv_filenames(folder_path):
    # Get a list of all CSV files in the folder
    file_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(file_pattern)

    # Extract filenames from the file paths
    csv_filenames = [os.path.basename(file_path) for file_path in csv_files]

    return csv_filenames


# Read in files
files = get_csv_filenames('Scraped_news')

# Create a df to put files in
articles = pd.DataFrame()

# Fill df with all the articles
for file in files:
    article = pd.read_csv(f'Scraped_news/{file}', encoding='utf-8',  converters={'Paragraphs': literal_eval})
    articles = pd.concat([articles, article], axis=0)

# Drop index column
articles = articles.drop(columns='Unnamed: 0')

# Function to remove empty strings from each article
def remove_empty_strings(lst):
    return [item for item in lst if (item.strip() != '')]


# Remove empty strings from the articles
articles['Paragraphs'] = articles['Paragraphs'].apply(remove_empty_strings)

# Drop any duplicate articles
articles = articles.drop_duplicates(subset=['Title'])


# Get rid of articles that have no content
articles = articles.drop(article[article['Paragraphs'].apply(len) == 0].index)
articles = articles.reset_index(drop=True)

# define a dictionary with the titles from the articles as not trash
data = {"not_trash": articles.iloc[:, 0].tolist()}

# Add trash sentences to the data dictionary
with open ("trash.txt", "r") as f:
    trash = f.read().splitlines()
    data["trash"] = trash

# Open NLP model for classification
nlp = spacy.blank("en")
nlp.add_pipe(
    "classy_classification",
    config={
        "data": data,
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "device": "gpu"
    }
)

# Function to get rid of "trash" sentences in articles
def clean_article(article):
    clean = []
    for sentence in article:
        doc = nlp(sentence)
        if doc._.cats["trash"] > .95:
            continue
        else:
            clean.append(sentence)
    return clean

# Apply trash filter to articles
articles['Paragraphs'] =  articles['Paragraphs'].apply(clean_article)

# Create a list to hold all the sentences in
sentences = []

# Add all the sentences from each article to the sentence list
for index, row in articles.iterrows():
    sentences += row['Paragraphs']

# Shuffle the sentences
random.shuffle(sentences)

# Classify a sentiment analyzer
sentiment_analyzer = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

# Create a dictionary to hold each sentences sentiment and confidence score
pos_neg_sentences = {
    'Sentence': [],
    'Label': [],
    'Score': []
}

# Extract 500 positive or negative sentences between 6 and 50 words long
for sentence in sentences:
    try:
        if (len(sentence.split(' ')) <= 50) & (len(sentence.split(' ')) >= 6):
            sent_results = sentiment_analyzer(sentence)

            if len(sent_results) > 0 and sent_results[0]['label'] != 'NEU':
                pos_neg_sentences['Sentence'].append(sentence)
                pos_neg_sentences['Label'].append(sent_results[0]['label'])
                pos_neg_sentences['Score'].append(sent_results[0]['score'])
    except Exception as e:
        print(f"Error processing sentence: '{sentence}'")
        print(f"Error message: {e}")
    if (len(pos_neg_sentences['Sentence']) == 500):
        break

# Store pos/neg sentences in a df
pos_neg_sentences_df = pd.DataFrame(pos_neg_sentences)

#pos_neg_sentences_df.to_csv('pos_neg_sentences.csv')

# Load another NLP model to extract phrases from each sentence
nlp = spacy.load('en_core_web_sm')

# Function to extract phrases from each sentence
def extract_phrases(text):
    doc = nlp(text)
    phrases = set()

    # Extract noun chunks
    for chunk in doc.noun_chunks:
        phrases.add(chunk.text)

    # Extract verb phrases using dependency parsing
    for token in doc:
        if token.pos_ == 'VERB':
            verb_phrase = ' '.join(
                [child.text for child in token.children if child.dep_ in {'aux', 'neg', 'advmod'}] + [token.text])
            phrases.add(verb_phrase)

    # Extract additional phrases using patterns
    matcher = Matcher(nlp.vocab)
    patterns = [
        [{"POS": "ADJ"}, {"POS": "NOUN"}],  # Adjective + Noun
        [{"POS": "NOUN"}, {"POS": "NOUN"}],  # Noun + Noun
        [{"POS": "ADV"}, {"POS": "VERB"}],  # Adverb + Verb
    ]
    matcher.add("PhrasePatterns", patterns)

    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        phrases.add(span.text)

    return list(phrases)


# Create a dictionary to hold each sentence's phrases and words
phrases_words = {
    'phrases_words' : []
}

# For each sentence, extract the relevant phrases and words
for index, row in pos_neg_sentences_df.iloc[0:].iterrows():
    phrases = extract_phrases(row['Sentence'])
    words = row['Sentence'].split(' ')

    for word in words:
        if (word not in phrases) and (word.lower() not in stop_words):
            phrases.append(word)
    phrases_words['phrases_words'].append(phrases)

# Add phrases to the df with pos/neg sentences
phrases_words_df = pd.DataFrame(phrases_words)
pos_neg_sentences_df = pd.concat([pos_neg_sentences_df, phrases_words_df], axis=1)

#pos_neg_sentences_df.to_csv('sentences_and_phrases.csv')

