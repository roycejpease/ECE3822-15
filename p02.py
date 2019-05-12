#!/usr/bin/env python

'''
Royce Pease
ECE-3822
HW15 - P02
royce.pease@temple.edu
'''

#libs
from collections import Counter
import math
import sys
import os
import operator
import time
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.dummy import Process, Queue

#config options
THREADS = 1                 #default single threading (usually faster for this program)
COMPARE_LEN = 50            #default max length for first pass histograms
LIST_SCALE_FACTOR = 2       #factor by which to increase histogram lengh each pass, ideal range (2,6)

#function to get a list of all files in a directory
def getFileList(dir):
    #list to hold all filenames found in the directory and subdirs
    fileList = []

    #use os.walk, gets the root, list of sub dirs in desired path, and list of files in those subdirs
    for root, dirs, files in os.walk(dir):
        #use os path join to add full path to filenames
        fileList += [os.path.join(root, file) for file in files]     #add all the files to the fileList

    #return the list of all files
    return fileList

#mapper to return list of tuples containing each word and a 1 to indicate its occurrence
def mapper(words):
    return [(hash(word), 1) for word in words]  #hashes words to improve memory usage compared to storing strings

#reducer to sum the mapped occurrences
def reducer(mappedWords):
    output = dict()     #directory to hold output in form {word:count}
    #itterate over the mapped words
    for word, occurrence in mappedWords:
        try:    #try and add to the words word count
            output[word] = output.get(word) + occurrence
        except: #if word was not added yet, create an entry for it
            output.update({word:occurrence})

    #return the resultant dictionary
    return output

#function to generate histogram of words in a file
def genHistogram(fname):
    allWords = []   #list to hold all words from a file

    if not os.path.isfile(fname):
        print("ERROR: " + fname + " not found.")
        return []

    #open the file an split into lines, then close
    file = open(fname, "r")
    lines = file.read().split('\n')
    file.close()

    #get the words off each line and add them to allWords
    for line in lines:
        allWords += [word.rstrip() for word in line.split(' ') if len(word.rstrip()) > 0]

    #get the counts of each word
    countedWords = reducer(mapper(allWords))
    #sort the word/count lists to form a sorted histogram
    return sorted(countedWords.items(), key=operator.itemgetter(1), reverse=True)[:COMPARE_LEN]

#function to orchstrate generating histograms from a list of files
def genHistograms(fnames):
    #list to hold histograms
    histograms = list()

    #multithreading available, default single threaded as overhead and IO ops make processing slower than
    #single threading here at most times
    pool = ThreadPool(THREADS)
    histograms += pool.map(genHistogram, fnames)

    return histograms

#function to compare two histograms for similarity
#uses the cosine similarity function
def similarity(hist1, hist2):
    completeWordList = set(hist1).union(hist2)
    #find dot procuct of the two sets, (uses hashes)
    dotProduct = sum(hist1.get(word, 0) * hist2.get(word, 0) for word in completeWordList)
    #find magnitudes of the sets (again uses hashes)
    m1 = math.sqrt(sum(hist1.get(word, 0)**2 for word in completeWordList))
    m2 = math.sqrt(sum(hist2.get(word, 0)**2 for word in completeWordList))
    #compare magnitudes (return 0 if one/both of them was empty)
    if (m1 * m2) == 0: return 0
    return dotProduct / (m1 * m2)  #similartity is the dotprod over the prod of magnitudes

#function to score a single pair (for multithreading)
def scorePair(pair):
    return similarity(Counter(pair[0]), Counter(pair[1]))

#score the histograms
def getScore(sortedHistograms, sortedIDX):
    top = [0,0,0]   #[score, file1_idx, file2_idx]
    scores = list() #list of all scores

    #generate a list of pairs as a list of tuples
    pairs = [(sortedHistograms[i], sortedHistograms[i+1], i) for i in range(len(sortedHistograms)-1)]

    #multiprocessing code / almost never faster than single threading due to overhead
    if len(pairs) > 100 and THREADS >= 4:
        pool = ThreadPool(THREADS)
        scores += pool.map(scorePair, pairs)

    #if multiprocessing disabled/unavailable, do entire pair list at once
    else:
        scores = [similarity(Counter(pair[0]), Counter(pair[1])) for pair in pairs]

    #generate the top list, [high score, file1 idx, file3 idx]
    top = [ max(scores), sortedIDX[ scores.index(max(scores)) ],   sortedIDX[ scores.index(max(scores)) + 1 ] ]

    #return the max and the list of all scores
    return top, scores

#driver function to compare a list of files for similarity
def compare(fnames, size):

    #set the global variable for compare length for the top n histograms
    global COMPARE_LEN
    if size is None:    #set the size, default to as large as is safe
        COMPARE_LEN = 50000000 / len(fnames)  #max stable is 1mill files at length 50, so divide 50 million by the number of files
    else:
        COMPARE_LEN = size  #if set, use the given value

    print("Comparing " + str(len(fnames)) + " files using histograms of up to top " + str(COMPARE_LEN) + " words in each file")

    print("\tGenerating histograms...")
    #generate the initial histogram, these are limited to COMPARE_LEN in size (default top 50 words)
    histograms = genHistograms(fnames)

    print("\tSorting...")
    #get indexes for a sorted list of histograms
    sortedIDX = sorted(range(len(histograms)), key=lambda k: histograms[k])

    #use the sorted indexes to form a sorted list of histograms
    histograms = [histograms[i] for i in sortedIDX]

    print("\tScoring...")
    #score the histograms by comparing adjacent pairs
    top, scores = getScore(histograms, sortedIDX)

    print("\tGenerting short-list...")
    #use the matches form the shortend histograms to make a "shortlist" of likely matches to compare complete histograms
    shortListIDX = list()

    for i, score in enumerate(scores):  #if the score is high enough, add the pair's indexes to the shortlist
        if score > (.90 * top[0]):
            shortListIDX += [i, i+1]
    shortListIDX = list(set(shortListIDX))    #remove dups

    #fish out the filenames of the files associated with the short list
    shortList = [fnames[sortedIDX[i]] for i in shortListIDX]

    #set found high if a single high score was found
    found = scores.count(max(scores)) == 1

    return top, shortList, found

def help():
    print("Usage: " + str(sys.argv[0]) + " <DIRECTORY>")
    print("Finds two most similar files in a given directory")
    exit(0)

#main function
def main():
    #store the start time
    start = time.time()

    #check arg count, show help if needed
    if(not len(sys.argv) == 2): help()

    #get dir from user
    dir = sys.argv[1]

    #get filenames from inside the directory
    fnames = getFileList(dir)

    #if something not a dir is passed in, show help and exit (this will also catch help flags)
    if len(fnames) == 0:
        print("Directory: " + str(dir) + " not found.")
        help()

    c_len = COMPARE_LEN  #begininng histogram size limit

    #perform first pass
    top, shortList, found = compare(fnames, c_len)

    #compute next passes
    while len(shortList) > 1:

        if len(shortList) < 500:    #if less than 500 files left, just compare the entire histograms of each
            c_len = None
        else:                       #increase the histogram size limit each iteration
            c_len *= LIST_SCALE_FACTOR

        fnames = shortList    #keep the old shortlist on hand
        top, shortList, found = compare(fnames, c_len)   #perform next pass
        if (shortList == fnames and found) or (c_len == None):
            break   #if the shortlist didn't change, and only 1 high score found, done

        if c_len > 10000: break #case to break if no difinitive match found, one of the best matches will show

    #show time elapsed
    print('\n' + str(time.time() - start) + ' seconds elapsed')

    #print the results
    print('\nMatches found with score ' + str(top[0]) + ':\n\n' + fnames[top[1]] + '\n' + fnames[top[2]])



#call main
if __name__ == '__main__':
    main()
    exit(0)
