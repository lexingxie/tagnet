
import sys, os
import json
import sqlite3
import codecs
import nltk
import re
import itertools
from collections import Counter
from bs4 import BeautifulSoup

#import pickle
#from glob import glob
#from operator import itemgetter

from datetime import datetime
from optparse import OptionParser

sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')

"""
    walk a dir of flickr json files
    ... read/write to cache file
    ... clean tags
    ... clean captions
"""
def flickr_get_txt(argv):
    if len(argv)<2:
        argv = ['-h']
    
    parser = OptionParser(description='parse flickr json files')
    parser.add_option("-i", "--in_dir", dest="in_dir", 
        default='', help="input dir containing all json files")
    parser.add_option("-o", "--out_dir", dest="out_dir", 
        default='', help="out txt")
    parser.add_option("-r", "--redo_data", dest="redo_data", action="store_true", 
        default=False, help="re-extract docs, ignore existing cache")
    parser.add_option('-d', '--db_file', dest='db_file', 
        default='dict.db', help='file containing dictionary db')
    parser.add_option('-a', '--addl_vocab', dest="addl_vocab", default="")
    parser.add_option('-v', '--vocab', dest="vocab", default="")
    parser.add_option('-p', '--preposition_list', dest="preposition_list", default="")
    
    opts, __ = parser.parse_args(argv)
    
    if opts.vocab: #additional out-of-dictionary words 
        vocab = open(opts.vocab, 'rt').read().strip().split()
    else:
        vocab = []
    
    if opts.addl_vocab: #additional out-of-dictionary words 
        addl_vocab = open(opts.addl_vocab, 'rt').read().strip().split()
    else:
        addl_vocab = []
    
    if opts.preposition_list: #additional out-of-dictionary words 
        prepo_list = open(opts.preposition_list, 'rt').read().strip().split("\n")
        prepo_list = prepo_list[::-1]
        pp = r"|".join(prepo_list)
        prepo_re = re.compile(r'\b'+pp+'\b', re.IGNORECASE)
        print "read %d prepositions: %s" %(len(prepo_list), pp)
    else:
        prepo_list = []
        prepo_re = None
    
    prepo_print = {}
    for p in prepo_list:
        prepo_print[p] = "_".join(p.split())
    
    print os.path.split(opts.in_dir.strip(os.sep)) 
    __, dir_name = os.path.split(opts.in_dir.strip(os.sep))
    #out_file = os.path.join(opts.out_dir, dir_name+".bigram")
    #uni_file = os.path.join(opts.out_dir, dir_name+".unigram")
    if not os.path.exists(opts.out_dir):
        os.makedirs(opts.out_dir)

    tags_file = os.path.join(opts.out_dir, dir_name+".tags")
    desc_file  = os.path.join(opts.out_dir, dir_name+".caption")
    psnt_file  = os.path.join(opts.out_dir, dir_name+".pair-sent")
    fsnt_file  = os.path.join(opts.out_dir, dir_name+".feat-sent")
    ftxt_file  = os.path.join(opts.out_dir, dir_name+".feat-txt")

    cfh = codecs.open(tags_file, encoding='utf-8', mode='wt')
    dfh = codecs.open(desc_file, encoding='utf-8', mode='wt')
    pfh = codecs.open(psnt_file, encoding='utf-8', mode='wt')
    sfh = codecs.open(fsnt_file, encoding='utf-8', mode='wt')
    tfh = codecs.open(ftxt_file, encoding='utf-8', mode='wt')
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s processing dir %s \n" % (tt, opts.in_dir)
    
    """ get dictionary from SQlite """
    conn = sqlite3.connect(opts.db_file)
    cursor = conn.cursor()
    jcnt = 0
    errcnt = 0
    emtcnt = 0

    
    for cp, __cd, fn in os.walk(opts.in_dir):         # UnusedVariable
        jn = filter(lambda s: s.find(".json")>0, fn)
        for j in jn:
            meta_name = os.path.join(cp, j)
            jstr = codecs.open(meta_name, encoding='utf-8', mode='rt').read()
            try:
                jinfo = json.loads( jstr )
            except Exception, e:
                print " ERR parsing json from %s" % j 
                print e
                print ""
                
            if 'stat' not in jinfo or not jinfo['stat']=='ok' :
                errcnt += 1
            else:
                jcnt += 1  # total # of non-empty json                  
                tag_raw = map(lambda s:s["_content"], jinfo["photo"]["tags"]["tag"])
                in_ttl = jinfo["photo"]["title"]["_content"]
                in_desc = jinfo["photo"]["description"]["_content"]
                imid,__ = os.path.splitext(j)
                
                if tag_raw:             
                    """    clean tags
                    """
                    tt = map(lambda s: norm_tag(s, cursor, addl_vocab=addl_vocab,filter_stopword=1), tag_raw)
                    tt = filter(lambda s: len(s)>1, tt)
                    tt = list(set(tt)) #unique
                else:
                    tt = []
                    
                if in_desc:
                    """  clean caption
                    """                    
                    wpairs, sent_feat, txt_cnter, txt_nolink = proc_caption(in_desc, prepo_list, vocab, cursor, addl_vocab)
                    #print in_desc
                    #print txt_nolink
                    
                else:
                    txt_nolink = ""
                    sent_feat = []
                
                # write the output    
                if len(sent_feat):
                    cfh.write("%s\t%s\n" % (imid, ",".join(tt)))
                    dfh.write("%s\t%s\n" % (imid, txt_nolink) )                    
                    tfh.write("%s\t%s\n" % (imid, " ".join(map(lambda t:"%s:%d"%(t[0],t[1]), \
                                                           txt_cnter.iteritems()) ) ) )
                    for i, sf in enumerate(sent_feat):
                        sf_str = ''
                        for k, v in sf.iteritems():
                            sf_str += ( " " + "%s:%d" % (prepo_print[k], v) )
                        sfh.write("%s_%02d\t%s\n" % (imid, i, sf_str) )
                        for wp in wpairs:
                            pfh.write("%s_%02d\t%s %s\n" % (imid, i, wp[0], wp[1]) )
                else:
                    emtcnt += 1 # # of json with either caption or tag empty
                    
                if jcnt % 1000 == 0:
                    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                    print "%s %7d docs done" % (tt, jcnt)

    cfh.close()
        
    conn.close()
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s: %d docs with text, %d empty, %d failed" % (tt, jcnt, emtcnt, errcnt)
    #write_unigram_file(uni_dict, uni_file)
    #write_bigram_file(bg_dict, out_file)
    
    # DONE


def proc_caption(in_txt, prepo_list=[], vocab=[], cursor=None, addl_vocab=[]):
    soup = BeautifulSoup(in_txt)
    txt_nolink = soup.get_text()
    #print txt_nolink
    
    sents = sent_tokenizer.tokenize(txt_nolink)
    tokens = nltk.word_tokenize(txt_nolink)
    
    if len(tokens) >= 3:
        tt = map(lambda s: norm_tag(s.lower(), cursor, addl_vocab=addl_vocab,filter_stopword=1), tokens)
        tt = filter(lambda s: len(s)>1, tt)
        txt_cnter = Counter()
        for word in tt:
            txt_cnter[word] += 1
        
        cnt = reduce(lambda x, t: x+1.*(t in vocab), list(set(tt)), 0)
        if cnt < 3:
            return([], "", {}, txt_nolink)
        # else
        sent_feat = []
        wpairs = []
        for st in sents:
            cur_str = st;
            # tokenize and filter string, set wpairs
            tkn = nltk.word_tokenize(cur_str)
            tt = map(lambda s: \
                     norm_tag(s.lower(), cursor, addl_vocab=addl_vocab,filter_stopword=1), tkn)
            tt = filter(lambda s: len(s)>1, tt)
            tt = filter(lambda s: s in vocab, tt)
            tt = list(set(tt)) # unique tags in vocab
            for i in range(len(tt)):
                for j in range(i):
                    if tt[i]<tt[j]:
                        wpairs += [tt[i], tt[j]]
                    else:
                        wpairs += [tt[j], tt[i]]
            
            cur_feat = {}
            for p in prepo_list:
                num = cur_str.count(p)
                if num :
                    cur_feat[p] = num
                    cur_str.replace(p, "")
            if cur_feat:
                sent_feat += [cur_feat]
            
        return wpairs, sent_feat, txt_cnter, txt_nolink
    else:
        return([], "", [], {})

    
def norm_tag(in_tag, cur, addl_vocab=[],filter_stopword=0):
    out_tag = ''
    
    stmt = "SELECT string,word FROM dict WHERE string='%s'" % in_tag
    try:
        #for row in cur.execute(stmt):
        for row in cur.execute("SELECT string,word,is_stopword FROM dict WHERE string=?", [in_tag]):
            assert not out_tag, "one in_tag should have only one match!"
            out_tag = row[1]
            if filter_stopword and row[2]==1:
                out_tag = ""
    except:
        print 'err in SQL: "%s"' % stmt
        raise
    
    if not out_tag and addl_vocab and in_tag in addl_vocab:
        out_tag = in_tag
           
    return out_tag


def flickr_hash_dir(imgid, meta_dir, hash_level=2, chars_per_hash=2):
    hdir = []
    for i in range(hash_level):
        curs = i*chars_per_hash
        cure = curs + chars_per_hash
        hdir.append(imgid[curs:cure])
    meta_name = os.path.join(meta_dir, '/'.join(hdir), imgid+".json")
    return meta_name


if __name__ == '__main__':  
    argv = sys.argv 
    flickr_get_txt(argv)
    
    #if '--collect_tags' in argv:
    #    argv.remove('--collect_tags')
    #    collect_tags(argv)
    #elif '--collect_user' in argv:
    #    argv.remove('--collect_user')
    #    collect_user(argv)
        