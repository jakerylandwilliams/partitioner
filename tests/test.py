
# coding: utf-8

# ## Partitioner examples
# ### This is a jupyter notebook with a few vignettes that present some of the Python partitioner package's functionality.
# Note: Cleaning of text and determination of clauses  occurs in the partitionText method.  Because of this, it is unwise to pass large, uncleaned pieces of text as 'clauses' directly through the .partition() method (regardless of the type of partition being taken), as this will simply tokenize the text by splitting on " ", producing many long, punctuation-filled phrases, and likely run very slow. As such, best practices only use .partition() for testing and exploring the tool on case-interested clauses.
# 

# In[1]:

import sys
# add the module
sys.path.append("..")

from partitioner import partitioner
from partitioner.methods import *


# ### Process the English Wiktionary to generate the (default) partition probabilities.
# Note: this step can take significant time for large dictionaries (~5 min).

# In[2]:

## Vignette 1: Build informed partition data from a dictionary, 
##             and store to local collection
def preprocessENwiktionary():
    pa = partitioner(informed = True, dictionary = "../partitioner/dictionaries/enwiktionary.txt")
    pa.dumpqs(qsname="enwiktionary")


def test_preprocess():
    preprocessENwiktionary()


# ### Perform a few one-off partitions.

# In[4]:

## Vignette 2: An informed, one-off partition of a single clause
def informedOneOffPartition(clause = "How are you doing today?"):
    pa = oneoff()
    print pa.partition(clause)


# In[5]:
    
def test_oneoff():
    informedOneOffPartition()
    informedOneOffPartition("Fine, thanks a bunch for asking!")


# ### Solve for the informed stochastic expectation partition (given the informed partition probabilities).

# In[6]:

## Vignette 3: An informed, stochastic expectation partition of a single clause
def informedStochasticPartition(clause = "How are you doing today?"):
    pa = stochastic()
    print pa.partition(clause)


# In[7]:

def test_informed():
    informedStochasticPartition()


# ### Perform a pure random (uniform) one-off partition.

# In[8]:

## Vignette 4: An uniform, one-off partition of a single clause
def uniformOneOffPartition(informed = False, clause = "How are you doing today?", qunif = 0.25):
    pa = oneoff(informed = informed, qunif = qunif)
    print pa.partition(clause)


# In[15]:
    
def test_uniformOneOff():
    uniformOneOffPartition()
    uniformOneOffPartition(qunif = 0.75)


# ### Solve for the uniform stochastic expectation partition (given the uniform partition probabilities).

# In[16]:

## Vignette 5: An uniform, stochastic expectation partition of a single clause
def uniformStochasticPartition(informed = False, clause = "How are you doing today?", qunif = 0.25):
    pa = stochastic(informed = informed, qunif = qunif)
    print pa.partition(clause)


# In[17]:

def test_uniformStochastic():
    uniformStochasticPartition()
    uniformStochasticPartition(clause = "Fine, thanks a bunch for asking!")


# ### Build a rank-frequency distribution for a text and determine its Zipf/Simon (bag-of-phrase) $R^2$.

# In[18]:

## Vignette 6: Use the default partitioning method to partition the main partitioner.py file and compute rsq
def testPartitionTextAndFit():
    pa = oneoff()
    pa.partitionText(textfile = pa.home+"/../README.md")
    pa.testFit()
    print "R-squared: ",round(pa.rsq,2)
    print
    phrases = sorted(pa.counts, key = lambda x: pa.counts[x], reverse = True)
    for j in range(25):
        phrase = phrases[j]
        print phrase, pa.counts[phrase]


# In[19]:

# testPartitionTextAndFit()

# In[ ]:

if __name__ == "__main__":
    test_preprocess()
    test_oneoff()
    test_informed()
    test_uniformOneOff()
    test_uniformStochastic()
    testPartitionTextAndFit()
    
