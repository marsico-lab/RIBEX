
import pandas as pd
import numpy as np

# plotting (does not work on JUWELS because )
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles

from scripts.initialize import *

FIGURES_DATA_SETS = FIGURES.joinpath("data_sets")
if FIGURES_DATA_SETS.exists() == False:
    FIGURES_DATA_SETS.mkdir()

#Dataset balance analysis
def analyze_Balance(dataSet):
    total = len(dataSet)
    pos = sum(dataSet.positive == True)
    neg = sum(dataSet.positive == False)
    none = sum(pd.isnull(dataSet.positive))
    log(f"\t\t\tbinding None: {(none/total)*100:.2f}%\t ({none})")
    log(f"\t\t\tbinding +: {(pos/total)*100:.2f}%\t ({pos})")
    log(f"\t\t\tbinding -: {(neg/total)*100:.2f}%\t ({neg})")
    log(f"\t\t\tbalance = 1 :  {(neg/pos):.2f}")

#analyze annotations
def analyze_Annotations(dataSet):
    total = len(dataSet)
    counter = [0,0,0]
    for annotations in dataSet.annotations:
        
        if(type(annotations) != str): #if its nan
            continue
        
        annotations = eval(annotations)
        counter_local = [0,0,0]
        for annotation in annotations:
            (fr, to, ty, name, sName) = annotation
            counter_local[ty] += 1
        
        counter[0] = counter[0]+1 if counter_local[0] > 0 else counter[0]
        counter[1] = counter[1]+1 if counter_local[1] > 0 else counter[1]
        counter[2] = counter[2]+1 if counter_local[2] > 0 else counter[2]
        
    log(f"\t\t\tHaving RBD: {(counter[1]/total)*100:.2f}%\t ({counter[1]})")
    log(f"\t\t\tHaving IDR: {(counter[2]/total)*100:.2f}%\t ({counter[2]})")
    log(f"\t\t\tHaving other: {(counter[0]/total)*100:.2f}%\t ({counter[0]})")

## Overlap analysis
def overlap(l1,l2):
    s1,s2 = set(l1), set(l2)
    overlap = s1&s2
    union = s1|s2

    o = len(overlap)
    p = o/len(union)

    p1 = o/len(s1)
    p2 = o/len(s2)

    return o,p, p1, p2

def overlapLists(dictOfLists):
    keys = list(dictOfLists.keys())
    for key1 in keys:
        for key2 in keys:
            if(key1 == key2):
                continue
            o,p, p1, p2 = overlap(dictOfLists[key1],dictOfLists[key2])
            log(f" Overlap({key1}, {key2}): {o} ({p*100:.2f}%)\n\t of {key1}: {p1*100:.2f}%\n\t of {key2}: {p2*100:.2f}%")

#overlap between positives
#def posOverlap(df1,df1_Name,df2,df2_Name, key="index"):

def plotOverlapLists2(list1, list2, name1, name2, suffix=None):

    set1, set2 = set(list1), set(list2)
    plt.figure()
    venn2(subsets=[set1, set2], set_labels=[f"{name1}\n({len(set1)})", f"{name2}\n({len(set2)})"])

    if(suffix != None):
        plt.savefig(FIGURES_DATA_SETS.joinpath(f"VENN2_{name1} VS {name2} {suffix}.svg"))
    else:
        plt.savefig(FIGURES_DATA_SETS.joinpath(f"VENN2_{name1} VS {name2}.svg"))


def plotOverlap2(dataSets, name1, name2, title=None, key="Gene_ID"):
    
    plotOverlapLists2( dataSets[name1][key], dataSets[name2][key], name1, name2, title)


def plotOverlapLists3(list1, list2, list3, name1, name2, name3, suffix=None):
    

    set1, set2, set3 = set(list1), set(list2), set(list3)
    plt.figure()
    venn3(subsets=[set1, set2, set3], set_labels=[f"{name1}\n({len(set1)})", f"{name2}\n({len(set2)})", f"{name3}\n({len(set3)})"])

    if(suffix != None):
        plt.savefig(FIGURES_DATA_SETS.joinpath(f"VENN3_{name1} VS {name2} VS {name3} {suffix}.svg"))
    else:
        plt.savefig(FIGURES_DATA_SETS.joinpath(f"VENN3_{name1} VS {name2} VS {name3}.svg"))


def plotOverlap3(dataSets, name1, name2, name3, suffix=None, key="Gene_ID"):
    
    plotOverlapLists3( dataSets[name1][key], dataSets[name2][key], dataSets[name3][key], name1, name2, name3, suffix)



#    if(key == "index"):
#        intersection = set(df1.index) & set(df2.index)
#        df1_inter = df1.loc[ [(v in intersection) for v in df1.index]]
#        df2_inter = df2.loc[ [(v in intersection) for v in df2.index]]
#        overlapLists({
#                df1_Name: df1_inter.loc[df1_inter["positive"]].index,
#                df2_Name: df2_inter.loc[df2_inter["positive"]].index
#            })
#    else:
#        intersection = set(df1[key]) & set(df2[key])
#        df1_inter = df1.loc[ [(v in intersection) for v in df1[key]]]
#        df2_inter = df2.loc[ [(v in intersection) for v in df2[key]]]
#        overlapLists({
#                df1_Name: df1_inter.loc[df1_inter["positive"]][key],
#                df2_Name: df2_inter.loc[df2_inter["positive"]][key]
#            })
    

