
import sys, os
#import json
#import sqlite3
#import codecs
#import nltk
#import re
#import string
from glob import glob
from collections import Counter
#from bs4 import BeautifulSoup
#from bs4 import element as bs4_element# for type checking

#import itertools
import pickle
#import numpy as np
import pylab
#import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import savemat

#from operator import itemgetter

from datetime import datetime
from optparse import OptionParser

"""
    frame id (img#, sent#, fr#): frame-name, frame-elements
    word-pair (w1, w2)
"""
EXT_TXT_FEAT = '.txt-feat'
EXT_FRM_FEAT = '.frame-feat'
EXT_FRM_PAIR = '.frame-pair'
EXT_FRM_WORD = '.frame-word'

def read_frm_word(in_file):
    out_dict = {}
    cnt = 0
    wcnt = 0
    with open(in_file, 'rt') as fh:
        for curline in fh:
            cl = curline.strip().split(None, 1)
            if len(cl)>1 and cl[1]:
                # parse frame id
                frm_id = tuple( cl[0].split("_") ) #820480653_02_00 --> (820480653, 02, 00)
                # parse BoW
                word_list = map(lambda s: s.rstrip(':1'), cl[1].split())
                out_dict[frm_id] = word_list
                cnt += 1
                wcnt += len(word_list)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s read %d entries %d words from %s" % (tt, cnt, wcnt, in_file)
    return out_dict
    
def read_frm_feat(in_file):
    out_frm_name = {}
    #out_frm_desc = {}
    cnt = 0
    fecnt = 0
    with open(in_file, 'rt') as fh:
        for curline in fh:
            cl = curline.strip().split(None, 1)
            if len(cl)>1 and cl[1]:
                # parse frame id
                frm_id = tuple( cl[0].split("_") ) #820480653_02_00 --> (820480653, 02, 00)
                # parse BoFE
                #82041665_08_02 Statement:1 Target:1 Message:1 Speaker:1
                fe_list = map(lambda s: s.rstrip(':1'), cl[1].split())
                out_frm_name[frm_id] = [ fe_list[0] ] # "Statement"
                if len(fe_list) > 2:
                    out_frm_name[frm_id] += map(lambda s: s.lower(), fe_list[2:])  # [Message, Speaker]
                    
                cnt += 1
                fecnt += len( out_frm_name[frm_id] )
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s read %d entries %d frame-elements from %s" % (tt, cnt, fecnt, in_file)            
    return out_frm_name        

def tagpair_get_frame(argv):
    if len(argv)<2:
        argv = ['-h']
    
    parser = OptionParser(description='parse word and frame features')
    parser.add_option("-i", "--input_dir", dest="input_dir", 
        default='', help="input dir containing sentence files")
    parser.add_option("-o", "--out_dir", dest="out_dir", 
        default='', help="out dir for semafor features and word pairs")
    parser.add_option("-g", "--glob_str", dest="glob_str", 
        default='8[0-2]', help="input file wild card string")
    parser.add_option("-f", "--feat_file_type", dest="feat_file_type", 
        default='FRAME_WORD', help="data file type [frame-word, frame-feat, ...]")
    
    parser.add_option('-D', '--DEBUG', dest="DEBUG", type='int', default=0)
    opts, __ = parser.parse_args(argv)
    
    in_file_list = glob(os.path.join(opts.input_dir, opts.glob_str+EXT_FRM_PAIR))
    in_file_list.sort()
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s processing %d input files: %s" % (tt, len(in_file_list), str(in_file_list)) 
    
    feat_type = opts.feat_file_type.lower()
    
    wpair_dict = {}
    wpair_cnt = Counter()
    lcnt = 0
    skip_cnt = 0
    for in_file in in_file_list: 
        pf, _ef = os.path.splitext(in_file)
        
        """ first load the data file """
        if feat_type == "frame_word":
            dfile_name = pf + EXT_FRM_WORD
            doc_dict = read_frm_word(dfile_name)
        elif feat_type == "frame_feat":
            dfile_name = pf + EXT_FRM_FEAT
            doc_dict = read_frm_feat(dfile_name)
        else:
            print "ERR! unknown filetype: %s" % opts.feat_file_type.lower()
        
        """ then load the word-pair file to join with it"""
        with open(in_file, 'rt') as in_fh:
            for curline in in_fh:
                cl = curline.strip().split(None, 1)
                if len(cl)>1 and cl[1]:
                    frm_id = tuple( cl[0].split("_") ) #820480653_02_00 --> (820480653, 02, 00)
                    wpair = tuple( cl[1].split(",") )
                    assert wpair[0] < wpair[1], " word pair ordering error!! %s " % str(wpair)
                    if frm_id not in doc_dict:
                        skip_cnt += 1
                        continue 
                    if wpair not in wpair_dict:
                        wpair_dict[wpair] = Counter(doc_dict[frm_id])
                        wpair_cnt[wpair] = 1
                    else:
                        wpair_dict[wpair].update(doc_dict[frm_id])
                        wpair_cnt[wpair] += 1
                
                lcnt += 1
    
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s finished loading %s " % (tt, in_file)
        print "\t %d unique word-pairs, total entries %d, %d skipped \n" % ( len(wpair_dict), lcnt, skip_cnt)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d_%Hh%Mm%S')
    out_pfile = os.path.join(opts.out_dir, "%s.%s.pickle" % (feat_type, tt) )
    pickle.dump( wpair_dict, open(out_pfile, 'wb') )
    
    print "%s: done saving to %s" % (tt, out_pfile)
    nump = 50
    numf = 6
    for wp, pcnt in wpair_cnt.most_common( nump ):
        flist = map(lambda t: " %s:%d" % (t[0], t[1]), wpair_dict[wp].most_common( numf ) )
        flist.sort()
        print "%s, # %d: %s" % ( str(wp), pcnt, " ".join( flist ) )
    
    return

def tagpair_get_sentence(argv):
    return

def aggregate_rel_features(argv):
    if len(argv)<2:
        argv = ['-h']
    
    parser = OptionParser(description='parse flickr json files')
    parser.add_option("-i", "--input_pickle", dest="input_pickle", 
        default='', help="input pickle file containing bigram features")
    parser.add_option("-r", "--relation_file", dest="relation_file", 
        default='', help="input pickle file for conceptnet relations")
    
    parser.add_option('-D', '--DEBUG', dest="DEBUG", type='int', default=0)
    opts, __ = parser.parse_args(argv)
    
    wpair_dict = pickle.load( open(opts.input_pickle, 'rb') )
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s load %d bigram features from %s" % (tt, len(wpair_dict), opts.input_pickle)
                
    rel_dict = {}
    rel_cnt = {}
    
    cnt_line = 0
    cnt_notfound = 0
    
    with open(opts.relation_file) as fh:
        for curline in fh:
            tmp = curline.strip().split(",")
            rel = tmp[0]
            if rel not in rel_dict:
                rel_dict[rel] = Counter()
                rel_cnt[rel] = 0
            
            w1 = min(tmp[1:])
            w2 = max(tmp[1:])
            if (w1, w2) in wpair_dict:
                rel_dict[rel].update(wpair_dict[(w1, w2)])
                rel_cnt[rel] += 1
            else:
                cnt_notfound += 1
            cnt_line += 1
            
            if cnt_line % 10000 == 0:
                tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                print "%s load %d lines, %d not found" % (tt, cnt_line, cnt_notfound)
    
    nump = 25
    #numf = 6
    wfrq = Counter()
    for r in rel_dict.iterkeys():
        cur_cntr = dict( map(lambda t: (t[0], 1.*t[1]/rel_cnt[r]) , rel_dict[r].most_common( nump ) ) )
        #print "%s - %d \t" % (r, rel_cnt[r]),
        #print " ".join( cur_cntr )
        
        wfrq.update( cur_cntr )
    
    wdim = sorted(wfrq, key=wfrq.__getitem__, reverse=True)
    rdim = rel_dict.keys()
    mat = pylab.rand(len(rel_dict), len(wdim))
    for ir, r in enumerate(rdim):
        for iw, w in enumerate(wdim):
            mat[ir, iw] = float( 1.*rel_dict[r][w]/rel_cnt[r] )
    
    # save the output
    out_mat = os.path.splitext(opts.input_pickle)[0] + ".mat"
    savemat(out_mat, {"rdim":rdim, "wdim":wdim, 'm':mat})
    
    """
        # now draw this
        fig = pylab.figure()
        ax = fig.add_subplot(111)
        
        ax.imshow(pylab.log10(mat + 1), cmap=cm.jet, interpolation='nearest') #, vmin=clevs[0],vmax=clevs[-1]) )
        #ax.colorbar()
        pylab.show()
    """
    return

def cmap_map(function,cmap):
    """ Applies function (which should operate on vectors of shape 3:
        [r, g, b], on colormap cmap. This routine will break any discontinuous     points in a colormap.
    """
    
    cdict = cmap._segmentdata
    #step_dict = {}
    step_dict = dict.fromkeys(('red','green', 'blue'))
    # Firt get the list of points where the segments start or end
    for key in ('red','green', 'blue'):         
        step_dict[key] = map(lambda x: x[0], cdict[key])
    
    step_list = sum(step_dict.values(), [])
    step_list = pylab.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : pylab.array(cmap(step)[0:3])
    old_LUT = pylab.array(map( reduced_cmap, step_list))
    new_LUT = pylab.array(map( function, old_LUT))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i,key in enumerate(('red','green','blue')):
        this_cdict = {}
        for j,step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j,i]
            elif new_LUT[j,i]!=old_LUT[j,i]:
                this_cdict[step] = new_LUT[j,i]
        colorvector=  map(lambda x: x + (x[1], ), this_cdict.items())
        colorvector.sort()
        cdict[key] = colorvector

    return LinearSegmentedColormap('colormap',cdict,1024)
  
if __name__ == '__main__':  
    argv = sys.argv
    if "--aggregate_rel_features" in argv:
        argv.remove('--aggregate_rel_features')
        aggregate_rel_features(argv)
        
    elif '--tagpair_get_sentence' in argv:
        argv.remove('--tagpair_get_sentence')
        tagpair_get_sentence(argv)
    else:
        tagpair_get_frame(argv)
