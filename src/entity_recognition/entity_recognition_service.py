import spacy
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import pipeline


class EntityRecognitionService:
    def __init__(self):
        self.tagger = SequenceTagger.load("ner")  # Load Flair's named entity recognition model

    def extract_entities_spacy(self, text: str):
        # Extract named entities using spaCy.
        doc = self.spacy_nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def extract_entities_flair(self, text: str):
        # Extract named entities using Flair.
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        return [(ent.text, ent.tag) for ent in sentence.get_spans("ner")]

    def extract_entities_transformers(self, text: str):
        # Extract named entities using Hugging Face Transformers NER model.

        entities = self.ner(text)
        return [(ent["word"], ent["entity"]) for ent in entities]

    def extract_entities_from_models(self, text: str):

        #Extract named entities from multiple models (spaCy, Flair, Transformers),

        combined_entities = set()
        combined_entities.update(self.extract_entities_flair(text))

        return combined_entities

    def normalize_entities_labels(self, entities):
        #Normalize entity labels from different models to a unified format.
        
        normalized = set()
        for entity_text, entity_label in entities:
            label_upper = entity_label.upper()
            
            if any(x in label_upper for x in ["PERSON", "PER", "B-PER", "I-PER"]):
                normalized.add((entity_text, "PERSON"))
            elif any(x in label_upper for x in ["NORP", "NATIONALITY", "RELIGION", "POLITICAL"]):
                normalized.add((entity_text, "NATIONALITY"))
            elif any(x in label_upper for x in ["FACILITY", "FAC"]):
                normalized.add((entity_text, "FACILITY"))
            elif any(x in label_upper for x in ["ORG", "B-ORG", "I-ORG", "COMPANY"]):
                normalized.add((entity_text, "ORGANIZATION"))
            elif any(x in label_upper for x in ["LOC", "GPE", "B-LOC", "I-LOC"]):
                normalized.add((entity_text, "LOCATION"))
            elif "PRODUCT" in label_upper:
                normalized.add((entity_text, "PRODUCT"))
            elif "EVENT" in label_upper:
                normalized.add((entity_text, "EVENT"))
            elif "WORK_OF_ART" in label_upper or "ART" in label_upper:
                normalized.add((entity_text, "WORK_OF_ART"))
            elif "LAW" in label_upper:
                normalized.add((entity_text, "LAW"))
            elif any(x in label_upper for x in ["DATE", "TIME"]):
                normalized.add((entity_text, "DATE"))
            elif "PERCENT" in label_upper:
                normalized.add((entity_text, "PERCENT"))
            elif any(x in label_upper for x in ["MONEY", "CURRENCY"]):
                normalized.add((entity_text, "MONEY"))
            elif "QUANTITY" in label_upper:
                normalized.add((entity_text, "QUANTITY"))
            elif "ORDINAL" in label_upper:
                normalized.add((entity_text, "ORDINAL"))
            elif "CARDINAL" in label_upper:
                normalized.add((entity_text, "CARDINAL"))
            elif "MISC" in label_upper: 
                normalized.add((entity_text, "MISC"))
            else:
                normalized.add((entity_text, entity_label))

        return normalized
    
    def extract_named_entities(self, text: str):
        # Extract and normalize named entities from the text.
        combined_entities = self.extract_entities_from_models(text)
        return self.normalize_entities_labels(combined_entities)



    