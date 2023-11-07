import pandas as pd
import spacy
from spacy import displacy
from nltk.corpus import wordnet
from flask import Flask, render_template, request


# Load the Dutch language model
nlp = spacy.load("nl_core_news_sm")

# import data from .csv file
dataset = pd.read_csv("rechtspraakcsv.csv", delimiter=";", header=0)

app = Flask(__name__)

@app.route("/go")
def serve_html():
    return render_template("index.html")

@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        search_term = request.form["search_term"]
        preprocessed_search_term = preprocess_text(search_term)
        search_term_tokens = preprocessed_search_term.split()
        doc = nlp(preprocessed_search_term)

        synonyms_list = []
        for token in search_term_tokens:
            synonyms = find_synonyms(token)
            if synonyms:
                synonyms_list.extend(synonyms)
            else:
                synonyms_list.append(token)

        matching_results_list = find_results_by_keywords(dataset, synonyms_list)

        return render_template("index.html", search_term=search_term, doc=doc, synonyms_list=synonyms_list, matching_results_list=matching_results_list)

    return render_template("index.html", search_term="", doc=None, synonyms_list=None, matching_results_list=None)

if __name__ == "__main__":
    app.run(debug=True)

#Function to find synonyms of a given keyword using WordNet
def find_synonyms(keyword):
    synonyms = set()
    for syn in wordnet.synsets(keyword, lang="nld"):
        for lemma in syn.lemmas(lang="nld"):
            synonyms.add(lemma.name().lower())
    return list(synonyms)
    
def preprocess_text(search_term):
    # Tokenize the input text
    terms = nlp(search_term)

    # Filter out stop words and punctuation
    tokens = [token.text for token in terms if not token.is_stop and not token.is_punct]

    # Join the remaining tokens to form the preprocessed text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Function to find results containing a given keyword or its synonyms in the summary
def find_results_by_keywords(dataset, keywords):
    matching_results = []
    search_result_id = 0  # Initialize search result ID

    for index, row in dataset.iterrows():
        summary = row['summary'].lower()
        matching_keywords = []

        for keyword in keywords:
            if keyword in summary:
                matching_keywords.append(keyword)

        if matching_keywords:
            # Calculate the score based on the number of matching terms
            score = 0
            for keyword in matching_keywords:
                if is_proper_noun(keyword):
                    score += 50  # Add 50 for proper nouns
                elif is_verb(keyword):
                    score += 20  # Add 20 for verbs
                else:
                    score += 10  # Add 10 for everything else


            # Increment the search result ID for each result
            search_result_id += 1
            matching_results.append((search_result_id, row['title'], row['issued'], matching_keywords, summary, score))

    return matching_results

# Function to check if a keyword is a pronoun
def is_verb(keyword):
    doc = nlp(keyword)
    for token in doc:
        if token.pos_ == 'VERB':
            return True
    return False

# Function to check if a keyword is a proper noun (named entity)
def is_proper_noun(keyword):
    doc = nlp(keyword)
    for token in doc:
        if token.pos_ == 'PROPN':
            return True
    return False

# Function to print a snippet of the summary containing the search term and its synonyms
def print_summary_snippet(summary, search_terms):
    # Tokenize the summary using spaCy
    summary_tokens = nlp(summary)
    snippet = []
    window_size = 20  # Number of words to include before and after the term

    for search_term in search_terms:
        # Count occurrences of each term
        term_count = count_term_occurrences(summary, search_term)
        print(f"'{search_term}' occurs {term_count} times in the summary.")
        
        snippet = []  # Initialize the snippet for the current search term
        for token in summary_tokens:
            if token.text.lower() == search_term:
                start = max(0, token.i - window_size)
                end = min(len(summary_tokens), token.i + window_size + 1)
                snippet.append(" ".join([t.text for t in summary_tokens[start:end]]))
        
        # Print the snippet for the current search term
        if snippet:
            print(f"Snippet for '{search_term}':")
            print("\n".join(snippet))
            print()

# Function to find the count of a term in the summary
def count_term_occurrences(summary, term):
    # Tokenize the summary using spaCy
    summary_tokens = nlp(summary)
    count = 0

    for token in summary_tokens:
        if token.text.lower() == term.lower():
            count += 1

    return count
