import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("")
displacy.serve(doc, style="ent")
