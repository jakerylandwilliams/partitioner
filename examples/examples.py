from partitioner import partitioner
from partitioner.methods import *
## This file scripts partitioner examples,
## which are to be run as standalone vignettes
## exhibiting partitioner's functionality

## Vignette 1: Build informed partition data from a dictionary, 
##             and store in a local repo
def preprocessENwiktionary():
    pa = partitioner(informed = True, dictionary = "./dictionaries/enwiktionary.txt")
    pa.dumpqs(qsname="enwiktionary")
    
## Note: Cleaning of text and determination of clauses
##       occurs in the partitionText method. 
##       Because of this, 
##       it is unwise to pass large, uncleaned pieces of text as 'clauses' 
##       directly through the .partition() method (regardless of the type of partition being taken),
##       as this will simply tokenize the text by splitting on " ", 
##       producing many long, punctuation-filled phrases, 
##       and likely run very slow.
##       As such, best practices only use .partition()
##       for testing and exploring the tool on case-interested clauses.
    
## Vignette 2: An informed, one-off partition of a single clause
def informedOneOffPartition(clause = "How are you doing today?"):
    pa = oneoff()
    print pa.partition(clause)

## Vignette 3: An informed, stochastic partition of a single clause
def informedStochasticPartition(clause = "How are you doing today?"):
    pa = stochastic()
    print pa.partition(clause)
    
## Vignette 4: An uniform, one-off partition of a single clause
def uniformOneOffPartition(informed = False, clause = "How are you doing today?", qunif = 0.25):
    pa = oneoff(informed = informed, qunif = qunif)
    print pa.partition(clause)

## Vignette 5: An informed, stochastic partition of a single clause
def uniformStochasticPartition(informed = False, clause = "How are you doing today?", qunif = 0.25):
    pa = stochastic(informed = informed, qunif = qunif)
    print pa.partition(clause)
    
## Vignette 6: Use the default partitioning method to partition the main partitioner.py file and compute rsq
def testPartitionTextAndFit():
    pa = oneoff()
    pa.partitionText(textfile = pa.home+"/partitioner.py")
    pa.testFit()
    print pa.rsq