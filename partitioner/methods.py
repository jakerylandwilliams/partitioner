import re, sys, json, os
import math as ma
import random as ra
import itertools as it
from partitioner import partitioner

class stochastic(partitioner):
    
    def testFit(self):
        print("Can't test fit goodness on non-deterministic partitions!")
        print("Import and run 'oneoff' to test goodness of fit.")
        sys.exit()
    
    def partition(self, clause):
        words = re.split(" ", clause)
        counts = {}
        if self.informed:
            ## get the probs:
            qs = []
            for i in range(1,len(words)):
                qs.append(self.qprob(words[i-1:i+1]))
        
        orders = [len(words) + 1 - k for k in range(1,len(words)+1)]
        for order in orders:    
            for start in range(len(words) - order + 1):
                end = start + order
                
                fq = 0.
                possible = 1
                if self.informed:
                    if start:
                        q = qs[start-1]
                        if q:
                            fq += ma.log(q,2.)
                        else:
                            possible = 0
                    if end - len(words):
                        q = qs[end-1]
                        if q:
                            fq += ma.log(q,2.)
                        else:
                            possible = 0

                    for i in range(1,len(words[start:end])):
                        q = qs[start:end-1][i-1]
                        if 1. - q:
                            fq += ma.log(1. - q, 2.)
                        else:
                            possible = 0
                            break
                else:
                    bnd = 0.
                    if not start:
                        bnd += 1.
                    if end == len(words):
                        bnd += 1.
                    if 1. - self.qunif:
                        fq += (order-1.)*ma.log(1. - self.qunif,2.) 
                    if self.qunif:
                        fq += (2. - bnd)*ma.log(self.qunif,2.)

                if possible:
                    fq = 2. ** fq
                    phrase = " ".join(words[start:end])
                    counts.setdefault(phrase,0.)
                    counts[phrase] += fq
                else:
                    fq = 0.    
        return counts

class oneoff(partitioner):
    
    def partition(self, clause):
        partition = []
        words = clause.split(" ")
        length = len(words)
        if length - 1:
            randNums = [ra.random() for i in range(length-1)]
            ## compute the 1-off partition for qInf here
            start = 0
            order = 1
            ends = []
            end = 1
            for randNum in randNums:
                if self.informed:
                    pprob = self.qprob(words[end-1:end+1])
                    if randNum <= pprob:
                        ends.append(end)
                else:
                    if randNum <= self.qunif:
                        ends.append(end)
                end += 1
            ends.append(end)
            start = 0
            for end in ends:
                phrase = str(" ".join(words[start:end]))
                partition.append(phrase)
                start = end
        else:
            phrase = str(words[0])
            partition.append(phrase)
        return partition