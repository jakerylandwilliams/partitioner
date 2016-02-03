import re, sys, json, os
import math as ma
import random as ra
import itertools as it

class partitioner:
    
    def __init__(self, 
                 informed = True, 
                 qunif = 0.5, 
                 dictionary = "NA", 
                 qsname = "enwiktionary",
                 case = False,
                 URLS = False,
                 hashtags = False,
                 handles = False,
                 markup = False,
                 seed = None
                ):
        self.home = os.path.dirname(os.path.realpath(__file__))
        self.seed = seed
        ra.seed(self.seed)
        self.case = case
        self.URLS = URLS
        self.hashtags = hashtags
        self.handles = handles
        self.markup = markup        
        self.informed = informed
        self.qunif = qunif
        self.dictionary = dictionary
        self.qsname = qsname
        if self.informed:
            if self.dictionary != "NA" or self.qsname != "NA":
                self.loadqs()
            else:
                print("Informed partitions require preprocessed q-probabilities or a dictionary!")
                sys.exit()

    def qprob(self,words):
        pair = " ".join(words).lower()
        if self.qs.get(pair, False):
            return self.qs[pair]
        else:
            return 1.0

    def dumpqs(self, qsname):
        with open(self.home+"/qdumps/"+qsname+".json","w") as f:
            f.writelines(json.dumps(self.qs))

    def loadqs(self, dictionary = "NA", qsname = "enwiktionary"):
        ## load in the boundary probs from the dictionary
        self.qs = {}
        if self.dictionary == "NA":
            ## add switch here to check if preprocessed set exists
            try:
                with open(self.home+"/qdumps/"+self.qsname+".json","r") as f:
                    self.qs = json.loads(f.read().strip())
            except IOError:
                print("Preprocessed probabilities for "+self.qsname+" have not yet been created!")
                print("Place preprocessed probabilities in partitionerPATH/qdumps/,")
                print("or load from a dictionay and run partitioner.dumpqs(qsname).")
                sys.exit()
        else:
            ## load in the boundary probs from the dictionary
            left = {}
            right = {}
            counts = {}
            pairs = {}
            defined = {}
            N = 0.
            try:
                f = open(self.dictionary,"r")
            except IOError:
                print("Specified dictionary does not appear to exist: "+self.dictionary)
                sys.exit()
            for phrase in f:
                phrase = phrase.strip().lower()
                defined[phrase] = 1

                words = re.split(" ",phrase)
                counts.setdefault(phrase,{})

                left.setdefault(words[-1],[])
                left[words[-1]].append(phrase)

                right.setdefault(words[0],[])
                right[words[0]].append(phrase)    

                ## add pairs for this phrase
                for i in range(1,len(words)):
                    pair = " ".join(words[i-1:i+1])

                    pairs.setdefault(pair,0.)
                    pairs[pair] += 1.

                    counts[phrase].setdefault(pair,0.)
                    counts[phrase][pair] += 1.
                N += 1.
            f.close()

            for pair in pairs:
                w1,w2 = re.split(" ",pair)
                k = 0.
                PSUM = 0.
                loss = 0.
                ## any pairs not in loop will contribute 0 to sum,
                ## so just need to know how many possible, for the denominator:
                ## = f(A B)*N + (N - f(A B))*f(A B) = f(A B)*(2N - f(A B))
                ## and then subtract this off by the number covered in the loop
                k = pairs[pair]*(2.*N - pairs[pair])
                if left.get(w1, False) and right.get(w2, False):
                    for L_phrase in left[w1]:
                        if counts[L_phrase].get(pair, False):
                            L_NUMPAIR = counts[L_phrase][pair]
                        else:
                            L_NUMPAIR = 0.
                        for R_phrase in right[w2]:
                            k += 1.
                            if counts[R_phrase].get(pair, False):
                                R_NUMPAIR = counts[R_phrase][pair]
                            else:
                                R_NUMPAIR = 0.
                            PSUM += 1./(1. + L_NUMPAIR + R_NUMPAIR)
                            if L_NUMPAIR + R_NUMPAIR:
                                loss += 1.
                ## correct for the phrases covered in the loop
                k -= loss
                self.qs[pair] = PSUM/k
    
    def washText(self, text):
        ## remove additional whitespace
        text = re.sub("^[ ]+","",text)
        text = re.sub("[ ]+$","",text)
        text = re.sub("[ ]+"," ",text)
        ## drop to lower case
        if self.case:
            text = text.lower()
        ## replace URLS with 'http'
        if self.URLS:
            text = re.sub("http[^ \n]+","http",text)
        ## replace hashtags with '#hash'
        if self.hashtags:
            text = re.sub("\#[^ ]+","#hash",text)
        ## replace handles with '@handle'
        if self.handles:
            text = re.sub("\@[^ ]+","@hand",text)
        ## revert common markup back to human readable
        if self.markup:
            text = re.sub("\&lt","\<",text)
            text = re.sub("\&gt","\>",text)
            text = re.sub("\&amp","\&",text)
            text = re.sub("\\n","\n",text)
            text = re.sub("\\t","\t",text)
        return text
    
    def testFit(self):
        self.rsq = "NA"
        sizes = {}
        for phrase in self.counts:
            count = self.counts[phrase]
            sizes.setdefault(count,0)
            sizes[count] += 1
        pairs = [[size,sizes[size]] for size in sizes]
        N = 0.0
        M = 0.0
        cumNumbers = []
        cumSizes = []
        for size,number in sorted(pairs,key=lambda x: x[0],reverse=True):
            N += float(number)
            M += float(size)*float(number)
            cumNumbers.append(N)
            cumSizes.append(float(size))
        if M:
            ## Zipf/Simon model fit:
            m = -(1. - N/M)
            b = -m*ma.log(N,10)

            f = [10**(m*ma.log(cumNumbers[i],10) + b) for i in range(len(cumSizes))]
            r = [cumSizes[i] - f[i] for i in range(len(cumNumbers))]
            
            fmean = sum(f)/float(len(f))
            mss = sum([(f[i] - fmean)**2.0 for i in range(len(f))])
            rss = sum([r[i]**2.0 for i in range(len(r))])
            self.rsq = mss/(mss + rss)
        else:
            print("There's no data on which to test a fit!")
            sys.exit()
            
    def partitionText(self,text = "", textfile = "NA"):
        ## set things up
        reg = re.compile("((\#|\@)?[a-zA-Z]+((\'|\-)[a-zA-Z]+)*\'?[ ]?)+")
        self.counts = {}
        if textfile != "NA":
            try:
                with open(textfile, "r") as f:
                    text=f.read()
            except IOError:
                print("Specified text file does not appear to exist: "+textfile)
                sys.exit()            
        text = self.washText(text)
        ## count the words/phrases
        for clause in reg.finditer(text):
            clause = clause.group()
            partition = self.partition(clause)
            if type(partition) is dict:
                for phrase in partition:
                    self.counts.setdefault(phrase,0.)
                    self.counts[phrase] += partition[phrase]
            else:
                for phrase in partition:
                    self.counts.setdefault(phrase,0.)
                    self.counts[phrase] += 1.
                    
    def partition(self, text = "", textfile = "NA"):
        print("Partition method not yet set!")
        print("To partition text, import and run 'oneoff' or 'stochastic'.")
        sys.exit()
