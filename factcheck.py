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
        with torch.no_grad(): # Disable gradient tracking for memory efficiency
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits # tensor of shape (batch_size, num_labels (3 labels))

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Find the maximum probability label
            label_idx = torch.argmax(probs).item() # without item() it will return a tensor so item returns a number
            labels = ["entailment", "neutral", "contradiction"]
            predicted_label = labels[label_idx]

            # Map the result to a binary decision: S or NS
            if predicted_label == "entailment":
                return "S"
            else:
                return "NS"

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
        Initialize the fact checker with a specific threshold for deciding
        "S" (Supported) vs "NS" (Not Supported)
        """
        self.threshold = threshold
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess(self, text: str) -> set:
        """
        Preprocess the input text by tokenizing, lowercasing, and removing stopwords/punctuation.
        """
        doc = self.nlp(text)
        words = {token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct}
        return words

    def calculate_jaccard(self, fact_words: set, passage_words: set) -> float:
        """
        :return: Jaccard similarity (float)
        """
        intersection = fact_words.intersection(passage_words)
        return len(intersection) / len(fact_words)

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Predict whether the fact is supported or not based on word recall.
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
    def __init__(self, ent_model):
        self.ent_model = ent_model # roberta_ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        max_score = float('-inf')
        final_label = "NS"  # default unless supported
        
        for passage in passages:
            text = passage["text"]
            sentences = text.split(".")
            
            for sentence in sentences:
                sentence = sentence.strip()  # Clean up any extra spaces
                # Get the entailment result for this sentence vs the fact
                result = self.ent_model.check_entailment(sentence, fact)
                if result == "S":
                    final_label = "S"  # If any sentence supports the fact, set it as supported
                    break  # No need to check other sentences

        return final_label

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

