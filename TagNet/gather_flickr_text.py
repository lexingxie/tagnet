
import sys, os
import json
import sqlite3
import codecs
import nltk
import re
from bs4 import BeautifulSoup

#import pickle
#from glob import glob
#from operator import itemgetter

from datetime import datetime
from optparse import OptionParser


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
        pp = r"|".join(prepo_list)
        prepo_re = re.compile(r'\b'+pp+'\b', re.IGNORECASE)
        print "read %d prepositions: %s" %(len(prepo_list), pp)
    else:
        prepo_list = []
        prepo_re = None

    print os.path.split(opts.in_dir.strip(os.sep)) 
    __, dir_name = os.path.split(opts.in_dir.strip(os.sep))
    #out_file = os.path.join(opts.out_dir, dir_name+".bigram")
    #uni_file = os.path.join(opts.out_dir, dir_name+".unigram")
    cache_file = os.path.join(opts.out_dir, dir_name+".tags")
    desc_file  = os.path.join(opts.out_dir, dir_name+".caption")
    if not os.path.exists(opts.out_dir):
        os.makedirs(opts.out_dir)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s processing dir %s" % (tt, opts.in_dir)
    
    """ get dictionary from SQlite """
    conn = sqlite3.connect(opts.db_file)
    cursor = conn.cursor()
    jcnt = 0
    errcnt = 0
    emtcnt = 0

    cfh = codecs.open(cache_file, encoding='utf-8', mode='wt')
    dfh = codecs.open(desc_file, encoding='utf-8', mode='wt')
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
                    cc = proc_caption(in_desc, prepo_re, vocab, cursor, addl_vocab)
                    print in_desc
                    print ">> " + cc
                    print ""
                else:
                    cc = ""
                    
                if len(tt) or len(cc):
                    cfh.write("%s\t%s\n" % (imid, ",".join(tt)))
                    dfh.write("%s\t%s\n" % (imid, cc) )
                else:
                    emtcnt += 1 # # of json with either caption or tag empty
                    
                if jcnt % 1000 == 0:
                    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                    print "%s %7d docs done" % (tt, jcnt)

    cfh.close()
        
    conn.close()
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s: %d docs with tags, %d empty, %d failed" % (tt, jcnt, emtcnt, errcnt)
    #write_unigram_file(uni_dict, uni_file)
    #write_bigram_file(bg_dict, out_file)
    
    # DONE


def proc_caption(in_txt, prepo_re=None, vocab=[], cursor=None, addl_vocab=[]):
    soup = BeautifulSoup(in_txt)
    txt_nolink = soup.get_text()
    
    tokens = nltk.word_tokenize(txt_nolink)
    if len(tokens) >= 3:
        #if prepo_re:
        #    m = prepo_re.match(in_txt)
        #    if not m:
        #        cc = " "
        #        return
        
        tt = map(lambda s: norm_tag(s.lower(), cursor, addl_vocab=addl_vocab,filter_stopword=1), tokens)
        tt = filter(lambda s: len(s)>1, tt)
        tt = list(set(tt)) #unique
        cnt = reduce(lambda x, t: x+1.*(t in vocab), tt, 0)
        print cnt, tt
        if cnt < 2:
            cc = " "
        else:
            cc = " ".join( filter(lambda s: s=='I' or len(s)>1, tokens) )
            
    else:
        cc = ' '
    return cc

    
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
        