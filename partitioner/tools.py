# -*- coding: utf-8 -*-
import re, json, os
import math as ma
import random as ra
import numpy as np
from nltk.tag.perceptron import PerceptronTagger

class partitioner(object):
    def __init__(self, q0 = 1., C = 0., informed = True, qunif = .5, lengths = 0, qsname = "en-wiktionary", doPOS = False, doLFD = True, maxgap = 8, seed = None, q = {"type": 0.76, "POS": 0.46}):

        ## universal initialization
        self.home = os.path.dirname(os.path.realpath(__file__))
        self.isPartition = False ## special flag makes sure 1-off partitions not thrown in expectations!
        with open(self.home+"/data/chars.txt","r") as f: ## load in the word characters
            self.chars = f.read().strip().decode('utf8')
            self.chars = re.sub(" ", "",self.chars)
        self.builtins = [re.sub("-qcounts.json", "", f) for f in os.listdir(self.home+"/data/") if re.search("-qcounts.json", f)] ## update this call to suit new partition data 

        ## method-setting parameters
        self.doLFD = doLFD        
        self.doPOS = doPOS
        self.informed = informed        
        self.q = q
        self.qunif = qunif        
        self.q0 = q0
        self.C = C
        
        ## lengths for expectations
        if lengths:
            for length in lengths:
                if not isinstance(length,(int, long)) and length > 0:
                    print "specified lengths must be a positive integers!"
                    return
        self.lengths = lengths

        ## stochastic partition seeds
        self.seed = seed
        ra.seed(self.seed)

        ## initiate part-of-speech tagger
        if self.doPOS:
            self.tagger = PerceptronTagger()

        ## gappy expressions limits
        if type(maxgap) is int:
            if maxgap > 0:
                self.maxgap = maxgap
            else:
                print("maxgap must be a non-negative integer!")
                return
        else:
            print("maxgap must be a non-negative integer!")
            return
        
        ## load partition data
        self.qsname = qsname ## presently qsname does not do anything (loadqs is a fixed function)
        if self.informed:
            if self.qsname in self.builtins:
                self.loadqs()
            else:
                print("Informed partitions must specify qsname for q-count information!")
                print("qsname: \'"+self.qsname+"\' is not available. Make sure \'"+self.home+"/data/"+self.qsname+"-qcounts.json\' is available!")
                print("Available options include:\n"+"\n".join(self.builtins))
                return
        else:
            self.qs = {"type": {}, "POS": {}}
            self.forms = {"type": {}, "POS": {}}
            self.contractions = {}
            self.ltypecounts = {"type": {}, "POS": {}}

    ## loads the partition data
    def loadqs(self):
        with open(self.home+"/data/en_counts.json","r") as f:
            self.qs = json.loads(f.read())
        with open(self.home+"/data/en_forms.json","r") as f:
            self.forms = json.loads(f.read())
        with open(self.home+"/data/en_gapcounts.json","r") as f:
            gapcounts = json.loads(f.read())
            for ky in gapcounts:
                for pair in gapcounts[ky]:
                    self.qs[ky][pair] = gapcounts[ky][pair]

        with open(self.home+"/data/en_ltypecounts.json","r") as f:
            self.ltypecounts = json.loads(f.read())
        with open(self.home+"/data/en_contractions.json","r") as f:
            self.contractions = json.loads(f.read())                

    ## gets a partition probability
    def qprob(self, pair, ky = "type"):
        ## probabilities are smoothed on the fly
        ## to produce one-off partitions:
        ## if self.q[ky] is outside of [0,1],
        ## then return value is (self.q[ky] +/- 1)
        ## with probability qprob()/1-qprob
        if self.informed:
            qcounts = self.qs[ky].get(pair, [0,0])
            if sum(qcounts):
                pprob = float(qcounts[0] + self.C*self.q0)/(float(sum(qcounts)) + self.C)
            else:
                pprob = 1.
        else:
            pprob = self.qunif
        if self.isPartition:
            if self.q[ky] < 0 or self.q[ky] > 1:
                if ra.random() <= pprob:
                    pprob = self.q[ky] + 1
                else:
                    pprob = self.q[ky] - 1
        return pprob

    ## connect adjacent words
    def addconnection(self, connection, connections):
        if len(connections):
            if connections[-1][-1] == connection[0]:
                connections[-1].append(connection[1])
            else:
                connections.append(list(connection))
        else:
            connections.append(list(connection))
        return connections

    ## connect non-adjacent words
    def insertconnection(self, connection, connections):
        if len(connections):
            sourceistarget = 0
            for i in range(len(connections)):
                if connections[i][-1] == connection[0]:
                    connections[i].append(connection[1])
                    sourceistarget = 1
            if not sourceistarget:
                connections.append(list(connection))
        else:
            connections.append(list(connection))
        return connections

    ## accept the longest-known MWEs from left-to-right
    def LFD(self, terms, indices, ky = "type"):
        forpartition = []
        while len(terms):
            ixs = sorted(range(1,len(terms)+1), reverse = True)
            for ix in ixs:
                if ky == "POS":
                    form = " ".join(terms[0:ix])
                else:
                    form = "".join(terms[0:ix])
                ## forms has all type-level and possibly POS forms, keyed by each
                if self.forms[ky].get(form,False) or not ix - 1:
                    forpartition.append([indices[j] for j in range(0,ix)])
                    if len(terms) == 1:
                        terms = []
                        indices = []
                    else:
                        terms = list(terms[ix:len(terms)])
                        indices = list(indices[ix:len(indices)])
                    break
        return forpartition

    ## Fitness of the current counts Zipf/Simon model
    def testFit(self):
        self.rsq = "NA"
        f = np.array(sorted(self.counts.values(), reverse=True))
        N = float(len(f))
        M = sum(f)
        if M:
            ## Zipf/Simon model fit:
            theta = -(1. - N/M)
            alpha = 1 + theta

            fhat = np.array([((alpha + n - 1.)/(alpha*M))**theta for n in range(1,int(N)+1)])

            r = (f - fhat)
            SSE = sum(r**2)
            mss = sum(((fhat - np.mean(fhat))**2.0))
            rss = sum(r**2.0)
            if mss + rss:
                self.rsq = mss/(mss + rss)
            else:
                self.rsq = 1.
        else:
            print("There's no data on which to test a fit!")
            ## sys.exit()
            return

    ## tokenization
    def process(self, text = ""):
        text = re.sub("\\\\n", " ", text)
        text = re.sub("\\\\t", " ", text)
        text = re.sub("\\\\\"", "\"", text)
        text = re.sub(" +", " ", text)
        text = re.sub("^ | $", " ", text)        
        protoblocks = re.split(" ", text)
        blocks = []
        for protoblock in protoblocks:
            newblocks = []

            if re.match("(\#|\@)", protoblock):
                newblocks = []
                protoblock, punk = re.findall("^(.*?)([^"+self.chars+"0-9\']+)?$",protoblock)[0]
                newblocks.extend(re.findall("((?:\#|\@)[^\#\@]*)", protoblock))
                newblocks.append(punk)
            elif re.match("http",protoblock) or re.match("["+self.chars+"0-9]+\@["+self.chars+"0-9]+\.["+self.chars+"0-9]+", protoblock):
                result = re.findall("^(.*?)([^"+self.chars+"0-9\']+)?$",protoblock)
                newblocks = [block for block in result[0] if len(block)]
            else:
                newblocks.extend(re.findall("([^"+self.chars+"0-9\']+|["+self.chars+"0-9\']+)", protoblock))
            for block in newblocks:
                if self.contractions.get(block, False):
                    blocks.extend(self.contractions[block])
                elif not re.search("["+self.chars+"0-9\']+", block):
                    blocks.extend(block)
                else:
                    blocks.append(block)
            blocks.append(" ")
        if len(blocks):
            del blocks[-1]
        ##
        ALLPOS = []        
        if self.doPOS:
            bls = [(blocks[x],x) for x in range(len(blocks)) if blocks[x] != " "]
            blks = [blk[0] for blk in bls]
            idxs = {str(bls[x][1]): x for x in range(len(bls))}
            POSS = [tag[1] for tag in self.tagger.tag(blks)]
            for x in range(len(blocks)):
                if str(x) in idxs.keys():
                    if re.match("\#|\@|http",blocks[x]):
                        ALLPOS.append({"#": "HASH", "@": "HAND", "h": "URL"}[blocks[x][0]])
                    elif re.match("["+self.chars+"0-9]+\@["+self.chars+"0-9]+\.["+self.chars+"0-9]+", blocks[x]):
                        ALLPOS.append("ADDR")
                    else:
                        ALLPOS.append(POSS[idxs[str(x)]])
                else:
                    ALLPOS.append("SP")
        ##
        return blocks, ALLPOS

    ## updates counts with partition
    def update(self, partition):
        if type(partition) is dict:
            for phrase in partition:
                if phrase != "":
                    self.counts.setdefault(phrase,0.)
                    self.counts[phrase] += partition[phrase]
        else:
            for phrase in partition:
                if phrase != "":
                    self.counts.setdefault(phrase,0.)
                    self.counts[phrase] += 1.

    ## partitions a whole text file or string and stores counts
    def partitionText(self,text = "", textfile = "NA"):
        ## initialize the counts dictionary
        self.counts = {}
        if textfile != "NA":
            try:
                f = open(textfile, "r")
                for text in f:
                    self.update(partition = self.partition(text = text))
                    
            except IOError:
                print("Specified text file does not appear to exist: "+textfile)
                ## sys.exit()
                return
        else:
            self.update(partition = self.partition(text = text))

    ## determine partition of a string
    def partition(self, text = ""):
        self.isPartition = True
        partition = []
        blocks, ALLPOS = self.process(text)
        
        ALLattached = {str(i): 0 for i in range(len(blocks)) if blocks[i] != " "} 
        ALLconnections = []

        if self.doPOS:
            kys = ["type", "POS"]
        else:
            kys = ["type"]
            
        for ky in kys:
            attached = {str(i): 0 for i in range(len(blocks)) if blocks[i] != " "} 
            connections = []
            unconnected = []
            
            l_block = ""
            if ky == "POS":
                punk = " "
            else:
                punk = ""
            r_block = ""

            firstword = 1
            l_ix = 0
            r_ix = 0
            p_ix = []

            for i in range(len(blocks)):
                block = blocks[i]
                if re.search("["+self.chars+"0-9\']",block):
                    if ky == "POS":
                        r_block = ALLPOS[i]
                    else:
                        r_block = block
                    r_ix = i
                    if len(l_block) or (not len(l_block) and (firstword and ((len(punk) and ky != "POS") or (len(punk) > 1 and ky == "POS")) )):
                        pair = l_block + punk + r_block
                        if self.ltypecounts[ky].get(pair, [0,0])[0] >= self.ltypecounts[ky].get(pair, [0,0])[1]:
                            thetype = "_"
                        else:
                            thetype = "~"

                        if self.qprob(pair, ky) < self.q[ky]:#### options engage here
                            if len(p_ix):
                                connections = self.addconnection([l_ix, p_ix[0]], connections)
                                attached[str(p_ix[0])] = thetype
                                for j in range(0,len(p_ix)-1):
                                    connections = self.addconnection([p_ix[j], p_ix[j+1]], connections)
                                    attached[str(p_ix[j+1])] = thetype
                                connections = self.addconnection([p_ix[-1], r_ix], connections)
                                attached[str(r_ix)] = thetype
                            else:
                                connections = self.addconnection([l_ix, r_ix], connections)
                                attached[str(r_ix)] = thetype
                        else:
                            unconnected.append(l_ix)

                        #### end methods and qs loops
                        if ky == "POS":
                            punk = " "
                        else:
                            punk = ""
                        p_ix = []
                    firstword = 0
                    l_block = r_block
                    l_ix = r_ix
                    r_block = ""
                    r_ix = 0
                else:
                    if ky == "POS":
                        punk = punk + ALLPOS[i] + " "
                    else:
                        punk = punk + block
                    if block != " ":
                        p_ix.append(i)

            if not len(r_block) and ((len(punk) and len(l_block) and ky != "POS") or (len(punk) > 1 and ky == "POS") ):
                ## q and methods loop here
                pair = l_block + punk
                if self.ltypecounts[ky].get(pair, [0,0])[0] >= self.ltypecounts[ky].get(pair, [0,0])[1]:
                    thetype = "_"
                else:
                    thetype = "~"        

                if self.qprob(pair, ky) < self.q[ky]: ## options engage here
                    if len(p_ix):
                        connections = self.addconnection([l_ix, p_ix[0]], connections)
                        attached[str(p_ix[0])] = thetype
                        for j in range(0,len(p_ix)-1):
                            connections = self.addconnection([p_ix[j], p_ix[j+1]], connections)
                            attached[str(p_ix[j+1])] = thetype
                else:
                    unconnected.append(l_ix)
                ## end qs and methods loops

            ## handle gappy MWEs
            if self.maxgap:
                for i in unconnected:
                    ## targets are either not attached in the current round, or not used in previous rounds
                    targets = sorted([int(j) for j in attached.keys() if not attached[j] and int(j) > i and int(j) - i <= self.maxgap])
                    for target in targets:
                        if ky == "POS":
                            pair = ALLPOS[i] + " _GAP_ " + ALLPOS[target]
                        else:
                            pair = blocks[i] + " _GAP_ " + blocks[target]

                        if self.ltypecounts[ky].get(pair, [0,0])[0] >= self.ltypecounts[ky].get(pair, [0,0])[1]:
                            thetype = "_"
                        else:
                            thetype = "~"

                        if self.qprob(pair, ky) < self.q[ky]: ## options engage here
                            connections = self.insertconnection([i, target], connections)
                            attached[str(target)] = thetype

            connections = sorted(connections, key = lambda x: x[0])

            ## apply the LFD
            if self.doLFD:
                newConnections = []
                for connection in connections:
                ## apply the LFD here!
                    terms = []
                    indices = []
                    j = connection[0]
                    for  i in connection:
                        if i - j >= 2:
                            if ky == "POS":
                                terms.append(ALLPOS[i-1])
                            else:
                                terms.append(blocks[i-1])
                            indices.append(i-1)
                        if ky == "POS":
                            terms.append(ALLPOS[i])
                        else:
                            terms.append(blocks[i])
                        indices.append(i)
                        j = i
                    for chunks in self.LFD(terms, indices, ky):
                        newConnection = []
                        for chunk in chunks:
                            if chunk in connection:
                                newConnection.append(chunk)
                        if len(newConnection) > 1:
                            newConnections.append(newConnection)
                connections = list(newConnections)

            ## merge connections here
            for ix in attached:
                if not ALLattached[ix]:
                    ALLattached[ix] = attached[ix]

            used = {}
            for connection in ALLconnections:
                for idex in connection:
                    used[str(idex)] = 1
            for connection in connections:
                for idex in connection:
                    if used.get(str(idex), False):
                        break
                else:
                    ALLconnections.append(list(connection))
            ALLconnections = sorted(ALLconnections, key = lambda x: x[0])
        #################

        ## work on making the final output
        ALLattached = [ALLattached.get(str(ix),0) for ix in range(len(blocks))]
        if not self.doPOS:
            ALLPOS = ["" for block in blocks]
        ix = 0
        for connection in ALLconnections:
            while ix < connection[0]:
                ## partition.append([(ix+1, blocks[ix], ALLPOS[ix], "0")])
                partition.append(blocks[ix])
                ix += 1
            MWE = []
            MWE.append((connection[0]+1, blocks[connection[0]], ALLPOS[connection[0]], "0"))
            inGap = []
            for j in range(1,len(connection)):
                if connection[j] - connection[j-1] > 1:
                    MWE.append((connection[j], blocks[connection[j]-1],ALLPOS[connection[j]-1],"0"))
                    if connection[j] - connection[j-1] > 2:
                        for k in range(connection[j-1]+1,connection[j]-1):
                            inGap.append([(k+1, blocks[k], ALLPOS[k], "0")])
                MWE.append((connection[j]+1, blocks[connection[j]], ALLPOS[connection[j]], str(connection[j-1]+1)+ALLattached[connection[j]]))
            ix = connection[-1] + 1
            ## partition.append(MWE)
            partition.append("".join([unit[1] for unit in MWE]))
            if len(inGap):
                ## partition.extend(inGap)
                for MWE in inGap:
                    partition.append("".join([unit[1] for unit in MWE]))
                        
        while ix < len(blocks):
            ## partition.append([(ix+1, blocks[ix], ALLPOS[ix], "0")])
            partition.append(blocks[ix])
            ix += 1
        #################
        return partition

    ## stochastic average partition counts of a string
    def expectation(self, text = ""):
        self.isPartition = False
        expectations = {}
        blocks, ALLPOS = self.process(text)
        words = [(word,i) for i,word in enumerate(blocks) if re.search("["+self.chars+"]",word)]
        gaps = []
        numblocks = len(blocks)
        numwords = len(words)

        ## get the q-boundary probabilities and associated gaps
        qs = []
        for i in range(0,numwords+1):
            ## not the first word
            if i:
                ## not the last word
                if i < numwords:
                    pairblocks = blocks[words[i-1][1]:words[i][1]+1]
                    if self.informed:
                        qs.append(self.qprob("".join(pairblocks), ky = "type"))
                    else:
                        qs.append(self.qunif)
                    ## there is a gap
                    if len(pairblocks) > 2:
                        gaps.append("".join(pairblocks[1:-1]))
                    ## there is no gap
                    else:
                        gaps.append("")
                ## is the last word
                else:
                    ## there is a trailing gap
                    if  words[-1][1] < numblocks - 1:
                        pairblocks = blocks[words[-1][1]:]
                        if self.informed:
                            qs.append(self.qprob("".join(pairblocks), ky = "type"))
                        else:
                            qs.append(self.qunif)
                        gaps.append("".join(pairblocks[1:]))
            ## is the first word
            else:
                ## there is a leading gap
                if words[0][1]:
                    pairblocks = blocks[0:words[i][1]+1]
                    if self.informed:
                        qs.append(self.qprob("".join(pairblocks), ky = "type"))
                    else:
                        qs.append(self.qunif)
                    gaps.append("".join(pairblocks[0:-1]))
        
        ## count up the gap mass that will arise from splitting
        for q,gap in zip(qs,gaps):
            if q:
                for char in gap:
                    expectations.setdefault(char,0.)
                    expectations[char] += q

        ## standardize the orders to collect
        if self.lengths:
            orders = []
            for length in self.lengths:
                if length <= len(words):
                    orders.append(length)
        else:
            orders = sorted(range(1,numwords+1),reverse=True)

        ## shift qs relative to words by leading punctuation blocks
        qshift = words[0][1]

        ## start from top to bottom by order
        for order in orders:
            ## start from left to right
            for start in range(len(words) - order + 1):
                end = start + order

                fq = 0.
                possible = 1
                ## count up internal bonding probabilities
                if start:
                    q = qs[qshift + start-1]
                    if q:
                        fq += ma.log(q,2.)
                    else:
                        possible = 0
                if end - len(words):
                    q = qs[qshift + end-1]
                    if q:
                        fq += ma.log(q,2.)
                    else:
                        possible = 0

                for i in range(1,len(words[start:end])):
                    q = qs[qshift + start:qshift + end-1][i-1]
                    if 1. - q:
                        fq += ma.log(1. - q, 2.)
                    else:
                        possible = 0
                        break

                if possible:
                    ## do bond/break combinations if leading/trailing gaps exist
                    if not start and words[0][1]:
                        ## there are four possible phrases: bond/break left/right combinations
                        if end == numwords and words[-1][1] < numblocks - 1:
                            ## break both gaps
                            if qs[0] and qs[-1]:
                                phrase = "".join(blocks[words[start][1]:words[end-1][1]+1])
                                expectations.setdefault(phrase,0.)
                                expectations[phrase] += 2. ** (fq + ma.log(qs[0], 2.) + ma.log(qs[-1], 2.))
                            ## bond left, break right
                            if 1. - qs[0] and qs[-1]:
                                phrase = "".join(blocks[0:words[end-1][1]+1])
                                expectations.setdefault(phrase,0.)
                                expectations[phrase] += 2. ** (fq + ma.log(1. - qs[0], 2.) + ma.log(qs[-1], 2.))
                            ## break left, bond right
                            if qs[0] and 1. - qs[-1]:
                                phrase = "".join(blocks[words[start][1]:])
                                expectations.setdefault(phrase,0.)
                                expectations[phrase] += 2. ** (fq + ma.log(qs[0], 2.) + ma.log(1. - qs[-1], 2.))
                            ## bond both gaps
                            if 1. - qs[0] and 1. - qs[-1]:
                                phrase = "".join(blocks)
                                expectations.setdefault(phrase,0.)
                                expectations[phrase] += 2. ** (fq + ma.log(1. - qs[0], 2.) + ma.log(1. - qs[-1], 2.))
                                
                        ## there are two possible phrases: bond/break gap on the left
                        else:
                            if qs[0]:
                                ## break the left gap
                                phrase = "".join(blocks[words[start][1]:words[end-1][1]+1])
                                expectations.setdefault(phrase,0.)
                                expectations[phrase] += 2. ** (fq + ma.log(qs[0], 2.))
                            if 1. - qs[0]:
                                ## bond the left gap
                                phrase = "".join(blocks[0:words[end-1][1]+1])
                                expectations.setdefault(phrase,0.)
                                expectations[phrase] += 2. ** (fq + ma.log(1. - qs[0], 2.))
                    else:
                        ## there are two possible phrases: bond/break gap on the right
                        if end == numwords and words[-1][1] < numblocks - 1:
                            if qs[-1]:
                                ## break the right gap
                                phrase = "".join(blocks[words[start][1]:words[end-1][1]+1])
                                expectations.setdefault(phrase,0.)
                                expectations[phrase] += 2. ** (fq + ma.log(qs[-1], 2.))
                            if 1. - qs[-1]:
                                ## bond the right gap
                                phrase = "".join(blocks[words[start][1]:])
                                expectations.setdefault(phrase,0.)
                                expectations[phrase] += 2. ** (fq + ma.log(1. - qs[-1], 2.))
                            
                        ## there are no leading/trailing gaps
                        else:
                            fq = 2. ** fq
                            phrase = "".join(blocks[words[start][1]:words[end-1][1]+1])
                            expectations.setdefault(phrase,0.)
                            expectations[phrase] += fq
                ## an impossible phrase has no mass; doesn't get counted
                else:
                    fq = 0.

        return expectations
