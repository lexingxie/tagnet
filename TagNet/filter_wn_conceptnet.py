import sys,os
import sqlite3
#import codecs
import numpy as np
#import math
#import gzip

from scipy import io
from datetime import datetime
from optparse import OptionParser


try:        
    #from nltk.corpus import wordnet as wn
    import conceptnet.models as cm
    en = cm.Language.get('en')
    import divisi2
except:
    print "WARNING: WN/CN related imports failed!"    
    
    
"""
    go through conceptnet tuples (v4 and v5), keep those:
    * uses flickr tag vocab \ proper location names(?)
    * has non-trivial bigram count
    * within a filtered set of relations
"""
def filter_conceptnet(argv):
    opts = options_get_wnet_tag(argv)
    
    # read unigram
    vocab_flickr = open(os.path.join(opts.data_home, opts.voacb_flickr), 'r').read()
    vocab_flickr = map(lambda s: s.strip(), vocab_flickr.split("\n"))
    vocab_place = open(os.path.join(opts.data_home, opts.vocab_place), 'r').read()
    vocab_place = map(lambda s: s.strip(), vocab_place.split("\n"))
    
    vocab_tag = list( set(vocab_flickr)-set(vocab_place) )
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s read %d flickr-vocab, %d without %d place names" % \
        (tt, len(vocab_flickr), len(vocab_tag), len(vocab_place))
    
    bigram_tag = {}
    bigram_cache = os.path.join(opts.data_home, opts.bigram_cache)
    cnt = [0, 0]
    if os.path.isfile(bigram_cache): # read from cache
        for cl in open(bigram_cache, "r").readlines():
            tmp = cl.strip().split() 
            m1 = min(tmp)
            if m1 not in bigram_tag:
                bigram_tag[m1] = []
            bigram_tag[m1] += [ max(tmp) ]
            cnt[1] += 1
    else:
        fo = open(bigram_cache, "w")
        for cl in open(os.path.join(opts.data_home, opts.bigram_flickr), 'r'):
            tmp = cl.strip().split()
            # 33744   nature  wildlife
            # 32754   flower  plant
            cnt[0] += 1
            if tmp[1] in vocab_tag and tmp[2] in vocab_tag:
                m1 = min(tmp[1:])
                m2 = max(tmp[1:])
                if m1 not in bigram_tag:
                    bigram_tag[m1] = []
                bigram_tag[m1] += [ m2 ]
                cnt[1] += 1
                fo.write("%s %s\n" % (m1, m2))
            if cnt[0] % 10000 == 0:
                tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                print "%s read %7d bigrams, %7d in-vocabulary " % (tt, cnt[0], cnt[1])
        fo.close()
                
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s Done. %d bigrams, %d in-vocabulary of %d\n" % (tt, cnt[0], cnt[1], len(vocab_tag))
    
    rel_list = open(os.path.join(opts.data_home, opts.rel_list), 'r').read()
    rel_list = map(lambda s: s.strip().lower(), rel_list.split("\n"))
    rel_list = filter(lambda s: not s[0]=="#", rel_list)
    
    out_file = open(os.path.join(opts.data_home, "cn%s_filtered.txt" % opts.conceptnet_version), 'w')
    
    if opts.conceptnet_version=='4':
        # filter based on bigram
        cnt = [0, 0, 0]
        A = divisi2.network.conceptnet_matrix('en')
        eA = A.named_entries()
        # [(0.792, u'fawn', ('right', u'IsA', u'deer')), (0.5, u'fawn', ('right', u'AtLocation', u'forest'))]
        for u in eA:
            cnt[0] += 1
            r = u[2][1]
            if u[2][0]=='right': 
                w1 = u[1]
                w2 = u[2][2]
            else:
                w1 = u[2][2]
                w2 = u[1]  
            
            if r.lower() in rel_list:
                cnt[1] += 1
                m1 = min([w1, w2])      
                m2 = max([w1, w2])          
                if m1 in bigram_tag and m2 in bigram_tag[m1]: #w1 in vocab_tag and w2 in vocab_tag:
                    cnt[2] += 1
                    out_file.write( ",".join([r, w1, w2]) + "\n")
            
            if cnt[0] % 10000 == 0:
                tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                print "%s scanned %d tuples, %d within relations, %d in vocab" % \
                        (tt, cnt[0], cnt[1], cnt[2])
                                        
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s Done. %d tuples, %d bigrams, %d within %d relations\n" % \
                (tt, cnt[0], cnt[2], cnt[1], len(rel_list))
        
    elif opts.conceptnet_version=='5':
        # filter based on tuples
        cnt = [0, 0, 0, 0]
        for cl in open(os.path.join(opts.data_home, opts.cn5_list), 'r'):
            tmp = cl.strip().split(",")
            cnt[0] += 1
            
            if not len(tmp):
                continue
            try:
                r = tmp[0]
                w1 = tmp[1]
                w2 = tmp[2]
            except:
                print cnt[0], cl
                continue
            
            if r.lower() in rel_list:
                cnt[1] += 1
                m1 = min([w1, w2])      
                m2 = max([w1, w2])
                if m1 in bigram_tag and m2 in bigram_tag[m1]: #w1 in vocab_tag and w2 in vocab_tag:
                    cnt[2] += 1
                    out_file.write( ",".join([r, w1, w2]) + "\n")
            
            if cnt[0] % 100000 == 0:
                tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                print "%s scanned %d tuples, %d within relations, %d in vocab" % \
                        (tt, cnt[0], cnt[1], cnt[2])
        
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s Done. scanned %d tuples, %d within relations, %d in vocab\n" % \
                (tt, cnt[0], cnt[1], cnt[2])
            
    else:
        print "unkonwn conceptnet version! quit"
    
    out_file.close()
    
def options_get_wnet_tag(argv):
    if len(argv)<2:
            argv = ['-h']
    parser = OptionParser(description='filter tuples for conceptnet tuples (v4 and v5)')
    parser.add_option('-d', '--data_home', dest='data_home', default="", help='parent dir of db, wnet-list, cache, etc')
    
    parser.add_option('-v', '--conceptnet_version', dest='conceptnet_version', default="4", help='')
    # input files
    parser.add_option("", '--cn5_list', dest='cn5_list', default="short_tuples_conceptnet5.1b.txt", help='')
    parser.add_option("", '--voacb_flickr', dest='voacb_flickr', default="vocab_flickr.txt", help='flickr tag vocabulary')
    parser.add_option("", '--vocab_place', dest='vocab_place', default="placenames_201211.txt", help='')
    parser.add_option("", '--bigram_flickr', dest='bigram_flickr', default="flickr.bigram.ge5.txt", help='')
    parser.add_option("", '--rel_list', dest='rel_list', default="cn4_relation_list.txt", help='')
    # cache files
    parser.add_option("", '--bigram_cache', dest='bigram_cache', default="flickr.bigram.ge5.conceptnet.txt", help='')
    
    (opts, __args) = parser.parse_args(sys.argv)
    
    return opts

if __name__ == '__main__':
    filter_conceptnet(sys.argv)



"""
for w1 in bigram_tag:
   for w2 in bigram_tag[w1]:
       assr12 = cm.Assertion.objects.filter(concept1__text=w1, concept2__text=w2,language=en)
       assr21 = cm.Assertion.objects.filter(concept1__text=w1, concept2__text=w2,language=en)
       atxt = map(lambda a: str(a).strip("[]"), assr12)
       atxt += map(lambda a: str(a).strip("[]"), assr21)
       cnt[0] += len(atxt)
       for ia in range(len(atxt)):
           atxt[ia] = atxt[ia].replace('(', ',').replace(')',',').replace(' ','').strip(',')
           
       atxt = filter(lambda s: s.split(",")[0].lower() in rel_list, atxt)
       cnt[1] += len(atxt)
       if len(atxt):
           out_file.write("\n".join(atxt) + "\n")
           
           cnt[2] += 1
           if cnt[2] and cnt[2] % 1000 == 0:
               tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
               print "%s %7d bigrams, %7d tuples, %7d within %d set relations" % \
                       (tt, cnt[2], cnt[0], cnt[1], len(rel_list))
"""   