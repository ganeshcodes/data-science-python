'''
    ** Purpose : Mining frequent itemsets and generating strong association rules using Apriori algorithm
    ** Steps to run : 
        * Navigate to the current directory
        * python apriori.py filename min_sup min_conf
        * file should be in the current directory
        * min_sup should be a valid integer value (1-N)
        * min_conf should be a valid integer value (1-100) 
        * E.g : python retail.dat 4000 65
    ** Created By : Ganeshbabu Thavasimuthu on 03/19/2017
'''

# Import sys module for reading the arguments
import sys

# Implementation of Apriori algorithm
class Apriori:
    def __init__(self, transactionsFileName, min_sup, min_conf):
        # Load all the transactions from the file to the python lists, trim/ignore newline characters
        D = self.loadTransactions(transactionsFileName)
        # L has the list of all frequent itemsets
        L = []
        # Start with finding frequent 1-itemset
        L1 = self.find_frequent_1_itemsets(D, min_sup)
        print("Frequent 1-itemset (L1) :")
        print("L1 = ",L1)
        L.append(L1)
        k = 1
        # Proceed to find frequent 2 to k-itemsets till we can't generate anymore frequent itemsets
        while L[k-1]!={}:
            # Call apriori_gen to generate candidate k-itemsets 
            print("Calculating candidate {}-itemset...".format(k+1))
            Ck = self.apriori_gen(L[k-1], k)
            print("C{} = {}".format(k+1,Ck))
            Lk = {}
            ck = {}
            # Generate frequent k-itemsets from candidate k-itemsets 
            print("Calculating frequent {}-itemset...".format(k+1))
            for transaction in D:
                Ct = self.subset(Ck,transaction)
                for subset in Ct:
                    if subset in ck:
                        ck[subset] += 1
                    else:
                        ck[subset] = 1
                # Add only the itemsets that meets the min_sup criteria
                Lk = {x:ck[x] for x in ck if ck[x] >= min_sup}
            # Add each frequent itemsets to L
            print("L{} = {}".format(k+1,Lk))
            L.append(Lk)
            k += 1
        print("**************FINAL OUTPUT (frequent itemsets and strong association rules)*************************\n")
        # Print all the generated frequent itemsets
        self.printAllFrequentItemsets(L,min_sup)
        # Generate and print all the strong association rules from the frequent itemsets, L
        self.generateAssociationRules(L,min_conf)
        print("**************FINAL OUTPUT (frequent itemsets and strong association rules)*************************\n")

    ''' 
        Generates the strong association rules with min_conf value 
    '''
    def generateAssociationRules(self, L, min_conf):
        print("STRONG ASSOCIATION RULES (Minimum Confidence = {}%)".format(min_conf))
        print("==================================================\n")
        k=1
        for freq_itemsets in L:
            if(freq_itemsets and k>1):
                for x in freq_itemsets:
                    itemset = eval(x) 
                    for i in range(len(itemset)):
                        A = itemset[i]
                        B = [x for x in itemset if x!=A]
                        # itemset is nothing but AUB. AUB is same for different combinations of A and B below
                        conf_A_B = self.calculateConfidence(A,itemset,L)
                        if(conf_A_B >= min_conf):
                            print("{} => {} confidence = {}%".format(A,B,conf_A_B))  
                        # From L3, generate rules for both A=>B and B=>A
                        # Eg: 39 => [38,48] and [38,48] => 39
                        if k>2:
                            conf_A_B = self.calculateConfidence(B,itemset,L)
                            if(conf_A_B >= min_conf):
                                print("{} => {} confidence = {}%".format(B,A,conf_A_B))
            k += 1
        print("\n")

    ''' 
        Calculate the confidence for each association rule using :
        Confidence(A=>B) = support(A U B) / support(A) 
    '''
    def calculateConfidence(self, A, AUB, L):
        # A can be just '38' or [38,39]
        if type(A) is str:
            A = repr([A])
        else:
            A = repr(A)
        conf_A_B = self.supportCount(AUB,L) / self.supportCount(eval(A),L)
        conf_A_B = round(conf_A_B*100, 2)
        return conf_A_B
    
    '''
        Returns the support count of the given itemset
    '''
    
    def supportCount(self, itemset, L):
        key = repr(itemset)
        for freq_itemsets in L:
            if(freq_itemsets):
                if(freq_itemsets.get(key)):
                    return freq_itemsets[key]

    '''
        Outputs all the frequent itemsets in the console
    '''
    def printAllFrequentItemsets(self,L,min_sup):
        print("FREQUENT ITEMSETS (Minimum Support = {})".format(min_sup))
        print("==================================================\n")
        for freq_itemsets in L:
            if(freq_itemsets):
                print(freq_itemsets)
                print("------------------------------------------------")
        print("\n")

    ''' 
        Load all the transactions in the file to the python list
    '''
    def loadTransactions(self, transactionsFileName):
        D = []
        with open(transactionsFileName) as fileobj:
            for line in fileobj:
                D.append(self.processLine(line))
        return D
    
    '''
        Convert each line to a valid list of items for processing
    '''
    def processLine(self, line):
        return [x.strip() for x in line.split(" ") if x!= '\n']

    '''
        Returns the frequent 1-itemset, L1 for the given min_sup value
    '''
    def find_frequent_1_itemsets(self, D, min_sup):
        L1 = {}
        for transaction in D:
            for item in transaction:
                if(item in L1):
                    L1[item] += 1
                else:
                    L1[item] = 1
        L1 = {str([x]):L1[x] for x in L1 if L1[x] >= min_sup}
        #print(L1)
        return L1

    '''
        Generates the candidate k-itemset using previous frequent itemset and apriori property
    '''
    def apriori_gen(self, Lk, k):
        l1 = [x for x in Lk]
        l2 = l1
        Ck = []
        for i in range(len(l1)):
            for j in range(i+1,len(l1)):
                itemset = []
                subset1 = eval(l1[i])
                subset2 = eval(l2[j])
                # Join step : Join only if first K-1 items are equal
                join = True
                for index in range(k-1):
                    if(subset1[index] != subset2[index]):
                        join = False
                        break
                if join:
                    itemset = self.joinSubsets(subset1,subset2)
                    # Prune step : Don't add unfruitful candidate to the candidate k-itemset
                    if(k>1 and self.has_infrequent_subset(itemset, Lk)):
                        pass
                    else:
                        Ck.append(repr(itemset)) 
                
        return Ck
    
    '''
        Finds whether the itemset is the potential candidate or not. 
        Checks if the all the subsets of itemset is frequent. (Apriori property)
    '''
    def has_infrequent_subset(self, itemset, Lk):
        l1 = [x for x in itemset]
        l2 = l1
        subsets = []
        for i in range(len(l1)):
            for j in range(i+1,len(l1)):
                s = []
                s.append(l1[i])
                s.append(l2[j])
                subsets.append(repr(s))
        for subset in subsets:
            if subset not in Lk:
                return True
        return False
    
    '''
        Generate the subsets for all the itemsets in Ck
    '''
    def subset(self, Ck, transaction):
        Ct = []
        for itemset in Ck:
            if len([x for x in eval(itemset) if x in transaction]) == len(eval(itemset)):
                Ct.append(itemset)
        return Ct
    
    ''' 
        Joins two subsets and returns a combined itemset
    '''
    def joinSubsets(self, subset1, subset2):
        #print("join called")
        itemset = []
        itemset.extend(subset1)
        for x in subset2:
            if x not in itemset:
                itemset.append(x)
        return itemset

# Read the arguments from the command and invoke Apriori algorithm
filename = sys.argv[1]
min_sup = int(sys.argv[2])
min_conf = int(sys.argv[3])
apriori = Apriori(filename, min_sup, min_conf)