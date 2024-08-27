import pandas as pd
import numpy as np
from ast import literal_eval
import spacy
from IPython.display import clear_output, display

# Import sentences and phrases as df
pos_neg_sentences_df = pd.read_csv('sentences_and_phrases.csv', encoding='utf-8',  converters={'phrases_words': literal_eval})

# Create dictionary to hold sentence and labels
sentence_encodings = {
    'Sentence': [],
    'Sentence Encodings': []
}

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


# Iterate through each sentence, showing each phrase/word allowing user to label as important or not
for index, row in pos_neg_sentences_df.iloc[0:].iterrows():

    phrases = extract_phrases(row['Sentence'])
    words = row['Sentence'].split(' ')

    for word in words:
        if (word not in phrases) and (word.lower() not in stop_words):
            phrases.append(word)
    clear_output(wait=True)
    print(row['Sentence'])
    print(row['Label'])
    word_importance = {}
    for word in phrases:
        print(word)
        answer = input()
        word_importance[word] = answer
    sentence_encodings['Sentence'].append(row['Sentence'])
    sentence_encodings['Sentence Encodings'].append(word_importance)
    sentence_encodings_df = pd.DataFrame(sentence_encodings)
    sentence_encodings_df.to_csv('sentence_encodings11.csv')