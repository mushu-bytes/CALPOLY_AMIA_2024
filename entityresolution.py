import pandas as pd
import numpy as np
import transformers
from sklearn.metrics import classification_report

import spacy
nlp = spacy.load("en_core_web_sm")

#Dataset Preprocessing

def get_noun_chunk(row):
    sent, entity = row[1], row[0]
    doc = nlp(sent)
    for chunk in doc.noun_chunks:
        if chunk.text.lower().find(entity.lower()) != -1:
            return chunk.text
    return entity

def get_root_chunk(row):
    sent, entity = row[1], row[0]
    doc = nlp(sent)
    for chunk in doc.noun_chunks:
        if chunk.text.lower().find(entity.lower()) != -1:
            return chunk.root.text
    return entity



#GPT Prompts

def GPT_sentence(term1, sent1, term2, sent2):
    PROMPT = "You are a healthcare professional who specializes in chronic pain. This chat thread will be used to predict entity resolution between two entities and their given context. Each entity will be given as a word and the sentence that it came from. Be sure to consider the context of the entity when labeling each entity pair. These are your labeling instructions: identify the label that best represents the given pair of entities: Matching - the two entities are identical or very closely similar in contextual meaning, and could be combined without losing information such that the two entities could be swapped in their respective sentences and would still hold the same definition in the context of the sentence. Not Matching - the two entities are not the same in context, even though they may be similar or related. A “not matching” indicates entities which do not fit the criteria to be labeled as “matching”. Output ONLY a 'yes' for matching, and ONLY a 'no' if they do not match."
    max_tokens = 4097
    relevant_paragraphs = []
    non_relevant_paragraphs = []
    messages=[
          {"role": "system", "content": PROMPT},
          {"role": "user", "content": "The first sentence and entity: A. %s, '%s'.\n The second sentence and entity: 2. %s, '%s'"%(sent1, term1, sent2, term2) }
    ]
    finished = False
    c = 1
    while not finished:
        if c > 3:
            break
        if c > 1:
            print(f"Trying openai for {c} time")
        try:
            response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=messages,
              temperature=0,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )
            finished = True
        except Exception as e:
            print(e)
        c += 1
    
    if finished:
        response = response['choices'][0]['message']['content'].strip()
        try:
            if response.find("yes") != -1:
                response = 1
            else:
                response = 0
        except Exception as e:
            print(e)
            response = 0
        return response
    else:
        return None
      
def GPT_compare_terms(term1, term2):
    PROMPT = "You are a healthcare professional who specializes in chronic pain. Here are your labeling instructions: identify the category that best represents the given pair of entities. Matching - the two entities are identical or very closely similar in contextual meaning, and could be used interchangeably without losing information. Not Matching - the two entities are not the same in context and cannot be used interchangeably, even though they may be similar or related. Ignore grammar and puntuation. Output ONLY a 'yes' for matching, and ONLY a 'no' if they do not match."
            
    max_tokens = 4097
    relevant_paragraphs = []
    non_relevant_paragraphs = []
    messages=[
          {"role": "system", "content": PROMPT},
          {"role": "user", "content": "1. %s\n2. %s"%(term1,term2) }
    ]
    finished = False
    c = 1
    while not finished:
        if c > 3:
            break
        if c > 1:
            print(f"Trying openai for {c} time")
        try:
            response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=messages,
              temperature=0,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )
            finished = True
        except Exception as e:
            print(e)
        c += 1
    
    if finished:
        response = response['choices'][0]['message']['content'].strip()
        try:
            if response.find("yes") != -1:
                response = 1
            else:
                response = 0
        except Exception as e:
            print(e)
            response = 0
        return response
    else:
        return None


#Entity Resolution Outputs

def compare_roots(row):
    rootA, rootB = row["Root A"], row["Root B"]
    return GPT_compare_terms(rootA, rootB)

def compare_chunks(row):
    chunkA, chunkB = row["Chunk A"], row["Chunk B"]
    return GPT_compare_terms(chunkA, chunkB)

def compare_entities(row):
    entityA, entityB = row["Entity A"], row["Entity B"]
    return GPT_compare_terms(entityA, entityB)

def compare_context(row):
    entityA, entityB = row["Entity A"], row["Entity B"]
    sentA, sentB = row["Sentence A"], row["Sentence B"]
    return GPT_sentence(entityA, sentA, entityB, sentB)



#Determining whether context is needed

def context_type(row):
    entityA, chunkA = row["Entity A"], row["Chunk A"]
    entityB, chunkB = row["Entity B"], row["Chunk B"]
    A = GPT_entities_helper(entityA, chunkA) == 1
    B = GPT_entities_helper(entityB, chunkB) == 1
    if A and B:
        return 0
    elif not A and B:
        return 1
    elif A and not B:
        return 1
    else:
        return 2

def ModContext(row):
    entityA, rootA = row["Entity A"], row["Root A"]
    entityB, rootB = row["Entity B"], row["Root B"]
    context = row["ContextType"]
    if context == 0:
        return row["Entities"]
    elif context == 1:
        return row["Roots"]
    else:
        return row["Roots"]

def check_positives(row):
    if row["Entities"]:
        return row["ModifiedContext"]
    return 0

    


                            





