#!/usr/bin/python

############################################################################
# Begin license text.
# Copyright Feb. 27, 2020 Robert A. Murphy

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# End license text.
############################################################################

############################################################################
##
## File:      utils.py
##
## Purpose:   Utility functions.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Oct. 12, 2021
##
############################################################################

import constants as const

from math                        import log,ceil,floor,sqrt,inf
from itertools                   import combinations
from matplotlib                  import colors
from sklearn.metrics             import roc_curve,RocCurveDisplay,auc,confusion_matrix,ConfusionMatrixDisplay,r2_score,PrecisionRecallDisplay,precision_recall_curve,precision_score,recall_score

import matplotlib.pyplot         as plt
import numpy                     as np
import pandas                    as pd
import seaborn                   as sns
import tensorflow                as tf

import os

# set the color palette
sns.set_palette(['#000000','#005C29'])

np.random.seed(const.constants.SEED)
tf.random.set_seed(const.constants.SEED)

############################################################################
##
## Purpose:   Unique list because np.unique returns strange results
##
############################################################################
def unique(l=[]):
    ret  = []
    if not (len(l) == 0):
        s    = l
        if type(s[0]) == type([]):
            s    = [tuple(t) for t in s]
        ret  = list(set(s))
        if type(s[0]) == type([]):
            ret  = [list(t) for t in ret]
    return np.asarray(ret)

############################################################################
##
## Purpose:   Permutations of a list of integers
##
############################################################################
def permute(dat=[],mine=True,l=const.constants.MAX_FEATURES):
    ret  = []
    sz   = len(dat)
    if not (sz == 0):
        if mine:
            # permute the array of indices beginning with the first element
            for j in range(0,sz+1):
                # all permutations of the array of indices
                jdat = list(dat[j:])
                jdat.extend(list(dat[:j]))
                for i in range(0,sz):
                    # only retain the sub arrays that are length >= 2
                    tmp = [list(x) for x in combinations(jdat,i+2)]
                    if len(tmp) > 0:
                        ret.extend(tmp)
        else:
            # permute the array of indices beginning with the first element
            lsz  = min(l,const.constants.MAX_FEATURES) if not (0 < l and l < min(const.constants.MAX_FEATURES,sz)) else l
            ret.extend(list(combinations(dat,lsz)))
    return unique(ret)

############################################################################
##
## Purpose:   Calculate the number of clusters to form using random clusters
##
############################################################################
def calcN(pts=None):
    ret = 2**2
    if (pts != None and type(pts) == type(0)):
        M    = ceil(sqrt(pts))
        for i in range(2,const.constants.MAX_FEATURES+1):
            N    = max(floor(sqrt(i)),2)
            if pts/(pts+(2*M*((N-1)^2))) <= 0.5:
                ret  = N**2
            else:
                break
    return ret

############################################################################
##
## Purpose:   Most frequently appearing element in a list
##
############################################################################
def most(l=[]):
    ret  = 0
    if (type(l) == type([]) or type(l) == type(np.asarray([]))):
        ll   = list(l)
        ret  = max(set(ll),key=ll.count)
    return ret

############################################################################
##
## Purpose:   Generate a list of hex colors for visual display of variance
##
############################################################################
def clrs(num=0):
    ret  = {}
    if not num <= 0:
        gc   = lambda n: list(map(lambda i: "#" + "%06x" % np.random.randint(0,0xFFFFFF),range(0,num)))
        clr  = gc(num)
        ret  = {i:clr[i] for i in range(0,num)}
    return ret

############################################################################
##
## Purpose:   Visual display of variance is plotted
##
############################################################################
def pclrs(mat=[],fl=None,title=None):
    ret  = {}
    if (type(mat) == type([]) or type(mat) == type(np.asarray([]))):
        if not len(mat) == 0:
            # need the data to be square if it is not already
            lm   = list(mat)
            dim  = int(ceil(sqrt(len(mat))))
            # extend the list of variance indicator
            if dim*dim > len(lm):
                lm.extend(np.full((dim*dim)-len(lm),0))
            # reshape the data for the heat map
            lm   = list(np.asarray(lm).reshape((dim,dim)))
            # now to map the variance indicators to colors
            disp = sns.heatmap(pd.DataFrame(lm))#,annot=True,annot_kws={"size":16})
            if title is not None:
                disp.ax_.set_title(title+": Variance")
            disp.plot()
            # save the image if requested
            if type(fl) == type(""):
                plt.savefig(fl)
                plt.close()
            else:
                plt.show()
            plt.cla()
            plt.clf()
    return

############################################################################
##
## Purpose:   Plot the ROC curve
##
############################################################################
def roc(tgts=[],scrs=[],fl=None):
    if (type(tgts) == type([]) or type(tgts) == type(np.asarray([]))) and \
       (type(scrs) == type([]) or type(scrs) == type(np.asarray([]))):
        if not (len(tgts) == 0 or len(scrs) == 0):
            # the false and true positve rate and the threshold
            fpr,tpr,thres = roc_curve(tgts,scrs)
            rauc          = auc(fpr,tpr)
            disp          = RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=rauc)
            disp.plot()
            # save the image if requested
            if type(fl) == type(""):
                plt.savefig(fl)
                plt.close()
            else:
                plt.show()
            plt.cla()
            plt.clf()
    return

############################################################################
##
## Purpose:   Plot the confusion matrix
##
############################################################################
def confusion(tgts=[],prds=[],fl=None):
    if (type(tgts) == type([]) or type(tgts) == type(np.asarray([]))) and \
       (type(prds) == type([]) or type(prds) == type(np.asarray([]))):
        if not (len(tgts) == 0 or len(prds) == 0):
            # the false and true positve rate and the threshold
            lbls = list(np.unique(tgts))
            cm   = confusion_matrix(tgts,prds,labels=lbls)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=lbls)
            disp.plot(cmap='Greens')
            mn   = min(min(tgts),min(prds))
            mx   = max(max(tgts),max(prds))
            #plt.axis([mn,mx,mn,mx])
            # save the image if requested
            if type(fl) == type(""):
                plt.savefig(fl)
                plt.close()
            else:
                plt.show()
            plt.cla()
            plt.clf()
    return

############################################################################
##
## Purpose:   Print the pagerank
##
############################################################################
def rank(dat=None,fl=None):
    # save the image if requested
    if type(dat) != type(None) and type(fl) == type(""):
        try:
            with open(fl,"w") as f:
                f.write(dat)
                f.close()
        except:
            print("Unable to write " + dat)
    else:
        print(dat)
    return

############################################################################
##
## Purpose:   Print the precision, recall, fscore
##
############################################################################
def prf(tgts=[],prds=[],fl=None):
    if (type(tgts) == type([]) or type(tgts) == type(np.asarray([]))) and \
       (type(prds) == type([]) or type(prds) == type(np.asarray([]))):
        if not (len(tgts) == 0 or len(prds) == 0):
            # the false and true positve rate and the threshold
            pre  = precision_score(tgts,prds,average="weighted")
            rec  =    recall_score(tgts,prds,average="weighted")
            dat  = "\n".join(["precision: "+str(pre),"recall: "+str(rec)])
            # save the image if requested
            if type(fl) == type(""):
                try:
                    with open(fl,"w") as f:
                        f.write(dat)
                        f.close()
                except:
                    print("Unable to write " + dat)
            else:
                print(dat)
    return

############################################################################
##
## Purpose:   Precision vs Recall
##
############################################################################
def pvr(tgts=[],prds=[],fl=None):
    if (type(tgts) == type([]) or type(tgts) == type(np.asarray([]))) and \
       (type(prds) == type([]) or type(prds) == type(np.asarray([]))):
        if not (len(tgts) == 0 or len(prds) == 0):
            # the false and true positve rate and the threshold
            pre,rec,_  = precision_recall_curve(tgts,prds)
            disp = PrecisionRecallDisplay(precision=pre,recall=rec)
            disp.plot()
            # save the image if requested
            if type(fl) == type(""):
                plt.savefig(fl)
                plt.close()
            else:
                plt.show()
            plt.cla()
            plt.clf()
    return

############################################################################
##
## Purpose:   Pair plot of marginal distributions of inputs
##
############################################################################
def pair(df=None,fl=None,title=None):
    if not type(df) == type(None):
        hue  = [t for t in title.split() if t in df.columns] \
               if title is not None                          \
               else title
        # pair plot of marginals
        g    = sns.pairplot(df,hue=hue[0] if not (hue is None or len(hue) == 0) else None)
        if type(title) == type(""):
            g.fig.suptitle(title+": Grid of Marginals")
        # save the image if requested
        if type(fl) == type(""):
            plt.savefig(fl)
            plt.close()
        else:
            plt.show()
        plt.cla()
        plt.clf()
    return

############################################################################
##
## Purpose:   Swarm plot of classifications
##
############################################################################
def swarm(tgts=None,prds=None,fl=None,title=None):
    if not (type(tgts) == type(None) or type(prds) == type(None)):
        # get the paired plots and save them
        sns.swarmplot(y=tgts,x=prds)
        if type(title) == type(""):
            plt.title("Classification of "+title)
        # save the image if requested
        if type(fl) == type(""):
            plt.savefig(fl)
            plt.close()
        else:
            plt.show()
        plt.cla()
        plt.clf()
    return

############################################################################
##
## Purpose:   Joint plot of regression outputs
##
############################################################################
def joint(tgts=[],prds=[],xlim=[],ylim=[],fl=None,title=None):
    if (not (type(tgts) == type(None) or type(prds) == type(None)))   and \
       (type(xlim) == type([]) or type(xlim) == type(np.asarray([]))) and \
       (type(ylim) == type([]) or type(ylim) == type(np.asarray([]))):
        # regression plot
        g    = sns.jointplot(x=tgts
                            ,y=prds
                            ,kind="reg"
                            ,xlim=xlim
                            ,ylim=ylim)
        if type(title) == type(""):
            g.fig.suptitle("Forecast of "+title)
        # save the image if requested
        if type(fl) == type(""):
            plt.savefig(fl)
            plt.close()
        else:
            plt.show()
        plt.cla()
        plt.clf()
    return

############################################################################
##
## Purpose:   Plot of fit vs residuals
##
############################################################################
def fitVres(tgts=[],prds=[],res=[],fl=None,title=None):
    if (not (type(tgts) == type(None) or type(tgts) == type(None)))   and \
       (type(res ) == type([]) or type(res ) == type(np.asarray([]))):
        # Two plots
        fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(12,6))
        # Histogram of residuals
        sns.distplot(res,ax=ax1)
        ax1.set_title(title+": Histogram of Residuals")
        # Fitted vs residuals
        sns.kdeplot(tgts,prds,ax=ax2,n_levels=40,xlim=[min(tgts)-1,max(tgts)+1])
        sns.regplot(x=tgts,y=prds,scatter=False,ax=ax2)
        ax2.set_title(title+": Fitted vs. Actual Values")
        # save the image if requested
        if type(fl) == type(""):
            plt.savefig(fl)
            plt.close()
        else:
            plt.show()
        plt.cla()
        plt.clf()
    return

############################################################################
##
## Purpose:   R-square
##
############################################################################
def r2(tgts=[],prds=[],fl=None):
    if (type(tgts) == type([]) or type(tgts) == type(np.asarray([]))) and \
       (type(prds) == type([]) or type(prds) == type(np.asarray([]))):
        if not (len(tgts) == 0 or len(prds) == 0):
            fig,ax = plt.subplots()
            ax.scatter(tgts,prds)
            ax.plot([tgts.min(),tgts.max()],[prds.min(),prds.max()],'k--',lw=4)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            #regression line
            ax.plot(tgts,prds)
            plt.annotate("r-squared = {:.3f}".format(r2_score(tgts,prds)),(0,1))
            # save the image if requested
            if type(fl) == type(""):
                plt.savefig(fl)
                plt.close()
            else:
                plt.show()
            plt.cla()
            plt.clf()
    return

############################################################################
##
## Purpose:   Check if argument is a string, int or float
##
############################################################################
def sif(arg=None):
    #ret  = type(arg)
    ret  = type(None)
    if not (type(arg) == type(None)):
        if type(arg) in [type([]),type(np.asarray([]))]:
            ret  = [sif(t) for t in arg]
        else:
            if type(arg) == type(""):
                try:
                    dump = int(arg)
                    ret  = type(0)
                except:
                    try:
                        dump = float(arg)
                        ret  = type(0.0)
                    except:
                        dump = str(arg)
                        ret  = type("")
            else:
                ret  = type(arg)#if type(arg) in [type([]),type(np.asarray([])),type(range(1)),type(pd.DataFrame(list(range(1))))] else type(None)
    return ret

############################################################################
##
## Purpose:   Utils class
##
############################################################################
class utils():
    @staticmethod
    def _unique(l=[]):
        return unique(l)
    @staticmethod
    def _permute(dat=[],mine=True,l=const.constants.MAX_FEATURES):
        return permute(dat,mine,l)
    @staticmethod
    def _calcN(pts=None):
        return calcN(pts)
    @staticmethod
    def _most(l=[]):
        return most(l)
    @staticmethod
    def _clrs(num=0):
        return clrs(num)
    @staticmethod
    def _pclrs(mat=[],fl=None,title=None):
        return pclrs(mat,fl,title)
    @staticmethod
    def _roc(tgts=[],scrs=[],fl=None):
        return roc(tgts,scrs,fl)
    @staticmethod
    def _confusion(tgts=[],prds=[],fl=None):
        return confusion(tgts,prds,fl)
    @staticmethod
    def _rank(dat=None,fl=None):
        return rank(dat,fl)
    @staticmethod
    def _prf(tgts=[],prds=[],fl=None):
        return prf(tgts,prds,fl)
    @staticmethod
    def _pvr(tgts=[],prds=[],fl=None):
        return pvr(tgts,prds,fl)
    @staticmethod
    def _pair(df=None,fl=None,title=None):
        return pair(df,fl,title)
    @staticmethod
    def _swarm(tgts=[],prds=[],fl=None,title=None):
        return swarm(tgts,prds,fl,title)
    @staticmethod
    def _joint(tgts=[],prds=[],xlim=[],ylim=[],fl=None,title=None):
        return joint(tgts,prds,xlim,ylim,fl,title)
    @staticmethod
    def _fitVres(tgts=[],prds=[],res=[],fl=None,title=None):
        return fitVres(tgts,prds,res,fl,title)
    @staticmethod
    def _r2(tgts=[],prds=[],fl=None):
        return r2(tgts,prds,fl)
