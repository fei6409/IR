import math
import os.path
import pickle
import subprocess
import sys
import time
import xml.etree.ElementTree as etree
import numpy as np

delim = '\r\n 　'


def printTime():
    print('clock =', time.clock(), file=sys.stderr)


def getDocName(docID):
    tree = etree.parse(fileList[docID])
    root = tree.getroot()

    return root.find('doc').find('id').text 


def grade(queryVec, docsVec, queryID, fp):
    score = {}

    queryWeight = float(np.dot(queryVec, queryVec))

    for i in docsVec:
        score[i] = float(np.dot(queryVec, docsVec[i])) / math.sqrt( docWeight[i] * queryWeight )

    sortedScore = sorted(score.items(), key=lambda x:x[1], reverse=True)

    for i in range(min(len(sortedScore), 120)):
        print(sortedScore[i], file=sys.stderr)
        print(queryID, getDocName(sortedScore[i][0]), file=fp)


def genVector(queryDic):
    wordID = [c for c in queryDic.keys()]
    L = len(wordID)
    queryVec = np.zeros(L)
    docsVec = {}

    for key in queryDic: # may have unigram and bigram
        ind = wordID.index(key)
        queryVec[ind] = queryDic[key]
        
        for docID in TFIDF[key]:
            if docID not in docsVec:
                docsVec[docID] = np.zeros(L)
            docsVec[docID][ind] = TFIDF[key][docID]
        
    return (queryVec, docsVec)


def addWeight(string, vector, w):
    strLen = len(string)
#    visit = [0 for i in range(strLen)]
    symbol = '，。？！、：；'
    for i in range(strLen):
#        if string[i] in symbol:
#            continue
        #unigram
        try:
            charid = vocab.index(string[i])
        except:
            print('Unigram not found in vocab list', file=sys.stderr)
            continue

        if charid not in invIndexUnigram:
            print('Unigram not found in inverted index', file=sys.stderr)
            continue

        if charid not in vector:
            vector[charid] = 0
        vector[charid] += w

        #bigram
        if i+1 < strLen:
            try:
                charid2 = vocab.index(string[i+1])
            except:
#                print('Bigram not found in vocab list', file=sys.stderr)
                continue

            tup = (charid, charid2)
            if tup not in invIndexBigram:
#                print('Bigram not found in inverted index', file=sys.stderr)
                continue
            
            if tup not in vector:
                vector[tup] = 0
            vector[tup] += w


def isChar(char):
    return ('a' <= char and char <= 'z') or ('A' <= char and char <= 'Z')

def parseString(string):
    ans = []
    i = 0
    strLen = len(string)
    while i < strLen:
        if string[i] == ' ': # use delim?
            continue
        elif isChar(string[i]):
            j = i+1
            while isChar(string[j]):
                j += 1
            ans.append(string[i:j])
            i = j
            if string[j] != ' ':
                i -= 1
        else:
            ans.append(string[i])
        i += 1
    return ans


def stopWordRemoval(string):
    for i in range(len(stopWords)):
        while True:
            try:
                string.remove(stopWords[i])
            except:
                break
    return string
    

def queryProcess():
    print('In query process', file=sys.stderr)
    tree = etree.parse(inputFileName)
    root = tree.getroot()
    nodes = ['title', 'question', 'narrative', 'concepts']
    w = [30,4,1,30]  #tuning
    strings = [ [] for i in range(len(nodes)) ]


    fp = open(outputFileName, 'w')

    for topic in root:
        queryDic = {}
        queryID = topic.find('number').text
        queryID = queryID[len(queryID)-3:len(queryID)]
        print('query ID =', queryID, file=sys.stderr)

        for i in range(len(nodes)):
            strings[i] = topic.find(nodes[i]).text.lstrip(delim).rstrip(delim)
            strings[i] = stopWordRemoval(strings[i])
            strings[i] = parseString(strings[i])

            addWeight(strings[i], queryDic, w[i])
        print('query len =', len(queryDic), file=sys.stderr)

        (queryVec, docsVec) = genVector(queryDic)

        grade(queryVec, docsVec, queryID, fp)

    fp.close()

    printTime()


def TF_IDF():
    print('Doing TF_IDF', file=sys.stderr)
    global TFIDF, docWeight

    if os.path.isfile('TFIDF.dat') and os.path.isfile('docWeight.dat'): 
        f = open('TFIDF.dat', 'rb')
        TFIDF = pickle.load(f)
        f.close()
        f = open('docWeight.dat', 'rb')
        docWeight = pickle.load(f)
        f.close()

    else:
        print('.dat not exist, generating', file=sys.stderr)

        TFIDF = {}
        docCnt = len(docSize)
        avgSize = 0
        for i in range(docCnt):
            avgSize += docSize[i]
        avgSize /= docCnt
        
        docWeight = [0 for i in range(docCnt)]
        para_b = 0.7 # tuning
        d = [(1 - para_b + para_b*docSize[i]/avgSize) for i in range(docCnt)]


        for i in invIndexUnigram: # word id
            IDF = math.log( docCnt / len(invIndexUnigram[i]) )
            TFIDF[i] = {}
            for j in invIndexUnigram[i]: # doc id
                v =  (invIndexUnigram[i][j] / d[j]) * IDF
                TFIDF[i][j] = v
                docWeight[j] += v * v

        for i in invIndexBigram: # word id
            IDF = math.log( docCnt / len(invIndexBigram[i]) )
            TFIDF[i] = {}
            for j in invIndexBigram[i]: # doc id
                v =  (invIndexBigram[i][j] / d[j]) * IDF
                TFIDF[i][j] = v
                docWeight[j] += v * v

        f = open('TFIDF.dat', 'wb')
        pickle.dump(TFIDF, f)
        f.close()
        f = open('docWeight.dat', 'wb')
        pickle.dump(docWeight, f)
        f.close()

    printTime()


def getDocSize():
    global docSize
    path = 'docSize.dat'

    print('Getting doc size', file=sys.stderr)

    if os.path.isfile(path) == True:
        f = open(path, 'rb')
        docSize = pickle.load(f)
        f.close()
    
    else:
        print('.dat not exist, generating', file=sys.stderr)
        
        docSize = [0 for i in range(len(fileList))]

        for i in invIndexUnigram:
            for j in invIndexUnigram[i]:
                docSize[j] += invIndexUnigram[i][j]

        f = open(path, 'wb')
        pickle.dump(docSize, f)
        f.close()
    printTime()


def readFile():
    global vocab, fileList, invIndexUnigram, invIndexBigram, stopWords
    
    print('Reading stopword.txt', file=sys.stderr)
    f = open('stopword.txt', 'r')
    stopWords = [line.rstrip(delim) for line in f.readlines()]
    f.close()
    print(stopWords, file=sys.stderr)

    print('Reading vocab.all', file=sys.stderr)
    vocab = []
    f = open(os.path.join(modelDir, 'vocab.all'), 'r')
    for line in f.readlines():
        vocab.append(line.rstrip(delim))
    f.close()

    print('Reading file-list', file=sys.stderr)
    fileList = []
    f = open(os.path.join(modelDir, 'file-list'), 'r')
    for line in f.readlines():
        fileList.append(line.rstrip(delim))
    f.close()

    print('Reading inverted-index', file=sys.stderr)

    invIndexUnigram = {}
    invIndexBigram = {}

    if os.path.isfile('invIndexUnigram.dat') and os.path.isfile('invIndexUnigram.dat'):
        f = open('invIndexUnigram.dat', 'rb')
        invIndexUnigram = pickle.load(f)
        f.close()
        f = open('invIndexBigram.dat', 'rb')
        invIndexBigram = pickle.load(f)
        f.close()

    else:
        print('.dat not exist, generating', file=sys.stderr)
        f = open(os.path.join(modelDir, 'inverted-index'), 'r')
        while True:
            line = f.readline()
            if(line == ''):
                break
            (id1, id2, n) = list( map( int, line.split(' ') ) )
            
            if n == 0:
                continue

            dic = {}

            for i in range(n):
                subline = f.readline()
                (file_id, cnt) = list( map( int, subline.split(' ') ) )
                dic[file_id] = cnt            

            if id2 == -1:
                invIndexUnigram[id1] = dic

            else:
                invIndexBigram[(id1, id2)] = dic

        f = open('invIndexUnigram.dat', 'wb')
        pickle.dump(invIndexUnigram, f)
        f.close()
        f = open('invIndexBigram.dat', 'wb')
        pickle.dump(invIndexBigram, f)
        f.close()

    printTime()

def argvProcess(): 
    global releFeedback, inputFileName, outputFileName, modelDir, ntcirDir
    releFeedback = False
    for i in range(0, len(sys.argv)):
        if sys.argv[i] == '-r':
            releFeedback = True
        elif sys.argv[i] == '-i':
            inputFileName = sys.argv[i+1]
        elif sys.argv[i] == '-o':
            outputFileName = sys.argv[i+1]
        elif sys.argv[i] == '-m':
            modelDir = sys.argv[i+1]
        elif sys.argv[i] == '-d':
            ntcirDir = sys.argv[i+1]
    print('relevance feedback: ' + str(releFeedback), file=sys.stderr)
    print('query file: ' + inputFileName, file=sys.stderr)
    print('ranked list: ' + outputFileName, file=sys.stderr)
    print('input model dictionary: ' + modelDir, file=sys.stderr)
    print('NTCIR dictionary: ' + ntcirDir, file=sys.stderr)


def main():
    argvProcess()
    readFile()
    getDocSize()
    TF_IDF()
    queryProcess()
    return

if __name__ == '__main__':
    main()
