from random import *
from decimal import Decimal

class TSDomainMapper():

    def __init__(self,ts):
        # sous serie de base ou sous serie segmentee ?
        self.mTS = ts

    def map_exp_ids(self,exp, positions=False):
        if positions:
            exp = [(self.mTS[x[0]],x[0], x[1])
                   for x in exp]
        else:
            exp = [(self.mTS[x[0]], x[1]) for x in exp]
        return exp

    def visualize_instance_html(self,ts):
        # TODO : html = aie

def generateTS(size=100,min=0,max=10):
    mlist = []
    for i in range(size):
        mlist.append(round(uniform(min,max),2))
    return mlist

def segmentationTS(ts,seg_length=5):
    res = [ts[x:x+seg_length] for x in range(0,len(ts),seg_length)]
    return res

def generateMockExp(ts):
    res = [(x,uniform(0,1)) for x in range(0,len(ts))]
    return res

"""------------------------------------------ TEST -----------------------------------------------"""
ts = generateTS(7,0,2)
ts_seg = segmentationTS(ts)
exp = generateMockExp(ts_seg)
print ts
print ts_seg
print exp
print "----------------------------"

TSDM = TSDomainMapper(ts_seg)
mapids = TSDM.map_exp_ids(exp,positions=True)
print mapids