# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        # Extract probabilities for entailment, neutral, and contradiction
        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        entailment_prob = probs[0]  # Assuming entailment is the first class

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        return entailment_prob


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise NotImplementedError()


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold=0.6):
        """
        Initialize the fact checker with a specific recall threshold.
        :param threshold: The recall threshold for deciding "S" (Supported) vs "NS" (Not Supported)
        """
        self.threshold = threshold
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess(self, text: str) -> set:
        """
        Preprocess the input text by tokenizing, lowercasing, and removing stopwords.
        :param text: The text to preprocess
        :return: A set of preprocessed words
        """
        # Use spaCy to tokenize, lowercase, and filter out stopwords and punctuation
        doc = self.nlp(text)
        words = {token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct}
        return words

    def calculate_jaccard(self, fact_words: set, passage_words: set) -> float:
        """
        Calculate recall as the fraction of fact words found in the passage.
        :param fact_words: Set of words from the fact
        :param passage_words: Set of words from the passage
        :return: Recall score (float)
        """
        if not fact_words:
            return 0.0
        intersection = fact_words.intersection(passage_words)
        return len(intersection) / len(fact_words)

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Predict whether the fact is supported or not based on word recall.
        :param fact: A string representing the fact
        :param passages: A list of passage dictionaries with "text" content
        :return: "S" if supported, otherwise "NS"
        """
        fact_words = self.preprocess(fact)
        for passage in passages:
            passage_words = self.preprocess(passage['text'])
            recall_score = self.calculate_jaccard(fact_words, passage_words)
            
            # If any passage meets the threshold, classify as "Supported"
            if recall_score >= self.threshold:
                return "S"
        return "NS"

class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model, entailment_threshold=0.8):
        """
        :param ent_model: Pre-trained entailment model (DeBERTa fine-tuned on MNLI, FEVER, etc.)
        :param entailment_threshold: Probability threshold for classifying a sentence as entailed
        """
        self.ent_model = ent_model
        self.entailment_threshold = entailment_threshold

    def predict(self, fact: str, passages: List[dict]) -> str:
        # Tokenize and split the fact into sentences using simple heuristics
        best_score = 0.0  # To keep track of the maximum entailment probability
        for passage in passages:
            # Split passage into sentences for finer entailment checking
            sentences = self.split_into_sentences(passage['text'])
            for sentence in sentences:
                # Get entailment probability for the sentence-fact pair
                entailment_prob = self.ent_model.check_entailment(sentence, fact)
                
                # Track the highest entailment probability across all sentences in all passages
                best_score = max(best_score, entailment_prob)

        # Classify based on max entailment score across all sentence pairs
        if best_score >= self.entailment_threshold:
            return "S"  # Supported
        else:
            return "NS"  # Not Supported

    def split_into_sentences(self, text):
        """
        Utility function to split a passage into sentences.
        """
        # For simplicity, let's use basic punctuation as sentence boundaries
        sentences = text.split('. ')
        return [sentence.strip() for sentence in sentences if sentence]



# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

