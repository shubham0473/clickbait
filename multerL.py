import nltk
import json
import numpy
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordNeuralDependencyParser
from nltk.tag.stanford import StanfordPOSTagger, StanfordNERTagger
from nltk.tokenize.stanford import StanfordTokenizer
import logging
logging.basicConfig(filename='logger.log',level=logging.DEBUG)
from collections import Counter
import itertools
import os
from multiprocessing import Process, Lock
import signal

st = StanfordNERTagger("/opt/stanford-ner-2015-12-09/classifiers/english.muc.7class.distsim.crf.ser.gz" , "/opt/stanford-ner-2015-12-09/stanford-ner.jar", encoding='utf-8')
TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$'
, 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

with open('slangs.json','r') as inp:
    slangs = json.load(inp)['slangs']

nProcess = 10

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
def load_cb():
    datafile = open('clickBaitData.json', 'r')
    return [json.loads(line)['title'] for line in datafile]

headlines = load_cb()
batchSize = len(headlines)/nProcess


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

def get_pos_stats(line):
    words = nltk.word_tokenize(line)
    classified_text = st.tag(words)

    lower_str = []
    for i in classified_text:
        # print i
        if i[1] != 'O':
            lower_str.append(i[0])
        else:
            lower_str.append(i[0].lower())
    tagged_words = nltk.pos_tag(lower_str)
    vector = {}
    for tag in TAGS:
        vector[tag] = 0
    for w in tagged_words:
        try:
            vector[w[1]] += 1
        except:
            pass
    return vector

def get_det_count(line):
    words = nltk.word_tokenize(line)

    tagged_words = nltk.pos_tag(words)
    count = 0
    for word in tagged_words:
        if word[1] == 'DT':
            count += 1

    return count


def get_no_of_stopwords(line):
    stop = nltk.corpus.stopwords.words('english')
    return len([i for i in line.lower().split() if i in stop])

def get_no_of_slangs(line):
    return len([i for i in line.split() if i in slangs])


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def get_ngrams(line):
    result = []
    words = nltk.word_tokenize(line)
    for i in range(1, 5):
        result.append(find_ngrams(words, i))
    return str(result)


def main(headlines, batch,lock):
    print 'starting with batch', batch
    for id,line in enumerate(headlines):
        result = {batch*batchSize + id: {
            'title' : line,
            'maxSyntacticLength' : sd_len(line),
            }}
        lock.acquire()
        with open('cb_features.json', 'a') as outfile:
            json.dump(result, outfile, indent=4)
        lock.release()
        print 'Done' + str(batch*batchSize + id)
        logging.info('Done' + str(batch*batchSize + id))
    pid = os.getpid()
    os.kill(pid, signal.SIGKILL)

if __name__ == '__main__':
    for batch in xrange(nProcess):
        lock = Lock()
        proc = Process(target=main, args=(headlines[batch*batchSize:batchSize*(batch+1)],batch,lock))
        proc.start()
