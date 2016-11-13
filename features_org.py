import nltk
import json
import numpy
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordNeuralDependencyParser
from nltk.tag.stanford import StanfordPOSTagger, StanfordNERTagger
from nltk.tokenize.stanford import StanfordTokenizer

parser=StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

# Feature extraction

    # Feature 1: Number of tokens
def get_no_of_tokens(line):
    tokens = nltk.word_tokenize(line)
    return len(tokens)

    #Feature 2: Average character count
def get_avg_char_count(line):
    tokens = nltk.word_tokenize(line)
    sum = 0
    for token in tokens:
        sum = sum + len(token)
    return sum/(1.0*len(tokens))

    #Feature 3: Length of Syntactic Dependencies
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
    return dep_size


def main():
    main_cb()
    main_ncb()

## FEATURE EXTRACTION CLICKBAIT

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
            'syntacticDependencyArray' : sd_len(line).tolist()
            }})
        i = i+1
    with open('cb_features.json', 'w+') as outfile:
        json.dump(result, outfile, indent=4)

## FEATURE EXTRACTION NON CLICKBAIT

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
            'syntacticDependencyArray' : sd_len(line).tolist()
            }})
    with open('noncb_features.json', 'w+') as outfile:
        json.dump(result, outfile, indent=4)

## MAIN

if __name__ == '__main__':
    main()

# Faltu Functions

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError