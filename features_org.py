# Author : Srinidhi Moodalagiri

import nltk
import json

def get_no_of_tokens(line):
    tokens = nltk.word_tokenize(line)
    return len(tokens)

def get_avg_char_count(line):
    tokens = nltk.word_tokenize(line)
    sum = 0
    for token in tokens:
        sum = sum + len(token)
    return sum/(1.0*len(tokens))

def main():
    main_cb()
    main_noncb()

## FEATURE EXTRACTION CLICKBAIT

def load_cb():
    datafile = open('clickBaitData.json', 'r')
    return [json.loads(line)['title'] for line in datafile]

def main_cb():
    headlines = load_cb()
    result = []
    for line in headlines:
        result.append({line: {'noOfTokens' : get_no_of_tokens(line), 'avgCharCount' : get_avg_char_count(line)}})
    with open('cb_features.json', 'w+') as outfile:
        json.dump(result, outfile, indent=4)

## FEATURE EXTRACTION NON CLICKBAIT

def load_ncb():
    datafile = open('nonClickBaitData.json', 'r')
    return [json.loads(line)['title'] for line in datafile]

def main_ncb():
    headlines = load_ncb()
    result = []
    for line in headlines:
        result.append({line: {'noOfTokens' : get_no_of_tokens(line), 
                              'avgCharCount' : get_avg_char_count(line)}})
    with open('noncb_features.json', 'w+') as outfile:
        json.dump(result, outfile, indent=4)

## MAIN

if __name__ == '__main__':
    main()
