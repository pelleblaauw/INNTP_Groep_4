import pandas as pd
import spacy
from spacy import displacy
from nltk.corpus import wordnet
from flask import Flask, render_template, send_from_directory, request
from flask import redirect, url_for
import os


# Load the Dutch language model
nlp = spacy.load("nl_core_news_sm")

# import data from .csv file
dataset = pd.read_csv("rechtspraakcsv.csv", delimiter=";", header=0)

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

app = Flask(__name__)

app.static_folder = 'static'
app.static_url_path = "/static"


@app.route("/")
def serve_html():
    return render_template("index.html")

@app.route('/favicon.ico')
def fav():
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico')

@app.route("/search", methods=["GET", "POST"])
def search():
    page = int(request.args.get('page', 1))  # Get the current page from the URL parameter
    search_term = request.args.get('search_term', '')


    if request.method == "POST":
        search_term = request.form["search_term"]
        preprocessed_search_term = preprocess_text(search_term)
        search_term_tokens = preprocessed_search_term.split()
        doc = nlp(preprocessed_search_term)

        # Render the dependency parsing visualization and pass it to the template
        dep_parsing_html = displacy.render(doc, style="dep", page=True)

        synonyms_list = []
        for token in search_term_tokens:
            synonyms = find_synonyms(token)
            if synonyms:
                synonyms_list.extend(synonyms)
            else:
                synonyms_list.append(token)

        matching_results_list = find_results_by_keywords(dataset, synonyms_list)

        # Sort the matching results by score in descending order
        sorted_results = sorted(matching_results_list, key=lambda x: x[5], reverse=True)

        # Define the number of results to display per page
        results_per_page = 5

        # Calculate the range of results to display for the current page
        start_idx = (page - 1) * results_per_page
        end_idx = start_idx + results_per_page

        # Slice the sorted results to get the results for the current page
        paginated_results = sorted_results[start_idx:end_idx]

        # Determine if there are more pages
        has_next_page = end_idx < len(sorted_results)
        has_prev_page = page > 1


        return render_template("results.html", search_term=search_term, synonyms_list=synonyms_list, matching_results_list=paginated_results, page=page, has_next_page=has_next_page, has_prev_page=has_prev_page, dep_parsing_html=dep_parsing_html)

    if request.method == "GET":
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

        # Sort the matching results by score in descending order
        sorted_results = sorted(matching_results_list, key=lambda x: x[5], reverse=True)

# Define the number of results to display per page
        results_per_page = 5

        # Calculate the range of results to display for the current page
        start_idx = (page - 1) * results_per_page
        end_idx = start_idx + results_per_page

        # Slice the sorted results to get the results for the current page
        paginated_results = sorted_results[start_idx:end_idx]

        # Determine if there are more pages
        has_next_page = end_idx < len(sorted_results)
        has_prev_page = page > 1


        return render_template("results.html", search_term=search_term, synonyms_list=synonyms_list, matching_results_list=paginated_results, page=page, has_next_page=has_next_page, has_prev_page=has_prev_page)
    return render_template("index.html", search_term="", synonyms_list=None)

@app.route("/feedback", methods=["POST"])
def feedback():
    if request.method == "POST":
        search_term = request.form["search_term"]
        
        matching_keywords = request.form["matching_keywords"]
        summary = request.form["summary"]
        feedback = request.form["feedback"]

        # Do something with the feedback data, e.g., print it to the console
        print("Feedback Received:")
        print("Search Term:", search_term)
        print("Matching Keywords:", matching_keywords)
        print("Summary:", summary)
        print("Feedback:", feedback)

        # You can also store the feedback data in a database or perform other actions as needed.

    # Redirect back to the search results page
    return redirect(url_for("search", search_term=search_term))


if __name__ == "__main__":
    app.run(debug=True)