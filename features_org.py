import nltk
import json
import numpy
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordNeuralDependencyParser
from nltk.tag.stanford import StanfordPOSTagger, StanfordNERTagger
from nltk.tokenize.stanford import StanfordTokenizer

from collections import Counter
import itertools

# Feature extraction

    # Feature 1.1: Number of tokens
def get_no_of_tokens(line):
    tokens = nltk.word_tokenize(line)
    return len(tokens)

    #Feature 1.2: Average character count
def get_avg_char_count(line):
    tokens = nltk.word_tokenize(line)
    sum = 0
    for token in tokens:
        sum = sum + len(token)
    return sum/(1.0*len(tokens))

    #Feature 1.3: Length of Syntactic Dependencies
def sd_len(line):
    tokens = nltk.word_tokenize(line)
    dp = StanfordDependencyParser()
    syndep = dp.raw_parse(line)
    dep = syndep.next()
    dep_size = numpy.zeros(19)
    for i in list(dep.triples()):
        gov = 0
        dep = 0
        it = 0
        for token in tokens:
            gov_word = "((u\'" + token +"\', u\'"
            dep_word = "(u\'" + token + "\'"
            if gov_word in str(i): 
                gov = it
            if dep_word in str(i): 
                dep = it
            it += 1
        if((gov - dep) != 0):
            dep_size[abs(gov - dep)] += 1
    return numpy.argmax(dep_size)

    # Feature 3.1: Find the most common subjects in headlines
def sub_count(line):
    tokens = nltk.word_tokenize(line)
    dp = StanfordDependencyParser()
    syndep = dp.raw_parse(line)
    dep = syndep.next()
    sub = "u\'nsubj\'"
    sub_list = []
    for i in list(dep.triples()):
        if sub in str(i):
            for token in tokens:
                if(", (u\'" + token +"\', u\'") in str(i):
                    sub_list.append(token)
                    continue
    return sub_list

## CLICKBAIT DUMP to JSON

def load_cb():
    datafile = open('clickBaitData.json', 'r')
    return [json.loads(line)['title'] for line in datafile]

def main_cb():
    headlines = load_cb()
    result = []
    i=0
    for line in headlines:
        result.append({i: {
            'title' : line, 
            'noOfTokens' : get_no_of_tokens(line), 
            'avgCharCount' : get_avg_char_count(line),
            'maxSyntacticLength' : sd_len(line),
            'subjectList' : sub_count(line)
            }})
        i = i+1
    with open('cb_features.json', 'w+') as outfile:
        json.dump(result, outfile, indent=4)

## NON CLICKBAIT DUMP to JSON

def load_ncb():
    datafile = open('nonClickBaitData.json', 'r')
    return [json.loads(line)['title'] for line in datafile]

def main_ncb():
    headlines = load_ncb()
    result = []
    i = 0
    for line in headlines:
        result.append({i: {
            'title' : line, 
            'noOfTokens' : get_no_of_tokens(line), 
            'avgCharCount' : get_avg_char_count(line),
            'maxSyntacticLength' : sd_len(line),
            'subjectList' : sub_count(line)        
            }})
        i = i+1
    with open('noncb_features.json', 'w+') as outfile:
        json.dump(result, outfile, indent=4)

## Common features
def main_common():
    # Clickbait
    headlines = load_cb()
    result_cb = []
    for line in headlines:
        result_cb.append(sub_count(line))
    flattened_cb = list(itertools.chain.from_iterable(result_cb))
    rcb = Counter(flattened_cb)

    # Non clickbait
    headlines = load_ncb()
    result_ncb = []
    for line in headlines:
        result_ncb.append(sub_count(line))  
    flattened_ncb = list(itertools.chain.from_iterable(result_ncb))
    rncb = Counter(flattened_ncb)

    # Add all to results
    result = []
    result.append({
        'clickBait' : [key for key,value in rcb.most_common(40)],
        'nonClickBait' : [key for key,value in rncb.most_common(40)]
        })
    with open('common_features.json', 'w+') as outfile:
        json.dump(result, outfile, indent=4)

## Call ClickBait and Non Clickbait functions
def main():
    main_cb()   
    main_ncb()
    main_common()

## MAIN

if __name__ == '__main__':
    main()

# Faltu Functions

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError