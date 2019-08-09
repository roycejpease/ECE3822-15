# ECE3822-15

Python application to find the two most similar files in an arbitrarily large set of text files.

Works by creating histograms of top N words in each file. The list of histograms is then sorted and all adjacent pairs compaired.
The top scoring file pairs move on to the next scan, where N is increased. This continues until a difinitive match is found.

Multithreading is supported but disabled by default. Currently faster in single threaded mode in most cases.
