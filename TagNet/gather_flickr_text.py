
import sys, os
import json
import sqlite3
import codecs
import nltk
#import re
import string
from glob import glob
from collections import Counter
from bs4 import BeautifulSoup
from bs4 import element as bs4_element# for type checking

#import itertools
#import pickle

#from operator import itemgetter

from datetime import datetime
from optparse import OptionParser

sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
SENT_TH = 1000

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
        #prepo_re = re.compile(r'\b'+pp+'\b', re.IGNORECASE)
        print "read %d prepositions: %s" % (len(prepo_list), pp)
    else:
        prepo_list = []
        #prepo_re = None
    
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
    desc_file  = os.path.join(opts.out_dir, dir_name+".sentence")
    #psnt_file  = os.path.join(opts.out_dir, dir_name+".pair-sent")
    fsnt_file  = os.path.join(opts.out_dir, dir_name+".sent-feat")
    ftxt_file  = os.path.join(opts.out_dir, dir_name+".txt-feat")

    cfh = codecs.open(tags_file, encoding='utf-8', mode='wt')
    dfh = codecs.open(desc_file, encoding='utf-8', mode='wt')
    #pfh = codecs.open(psnt_file, encoding='utf-8', mode='wt')
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
                in_txt = " . ".join([in_ttl, in_desc])
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
                    sents, sent_feat, txt_cnter = caption2sentence(in_txt, prepo_list, vocab, cursor, addl_vocab)
                    #  wpairs, sent_feat, txt_cnter, sents = proc_caption(in_desc, prepo_list, vocab, cursor, addl_vocab)
                    
                else:
                    sents = []
                    sent_feat = []
                
                # write the output    
                if len(sent_feat):
                    # tags
                    cfh.write("%s\t%s\n" % (imid, ",".join(tt)))
                    # boW overall
                    tfh.write("%s\t%s\n" % (imid, " ".join(map(lambda t:"%s:%d"%(t[0],t[1]), \
                                                           txt_cnter.iteritems()) ) ) )
                    # sentences
                    for i, sn in enumerate(sents):
                        dfh.write("%s_%02d\t%s\n" % (imid, i, sn) )
                      
                        # setence features
                        sf = sent_feat[i]
                        sf_str = ''
                        for k, v in sf.iteritems():
                            sf_str += ( " " + "%s:%d" % (prepo_print[k], v) )
                        sfh.write("%s_%02d\t%s\n" % (imid, i, sf_str) )
                        """
                            wp = wpairs[i]
                            for tp in wp:
                                pfh.write("%s_%02d\t%s %s\n" % (imid, i, tp[0], tp[1]) )
                        """
                else:
                    emtcnt += 1 # # of json with either caption or tag empty
                    
                if jcnt % 1000 == 0:
                    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                    print "%s %7d docs done" % (tt, jcnt)

    cfh.close()
        
    conn.close()
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s: %d docs with text, %d empty, %d failed" % (tt, jcnt, emtcnt, errcnt)
    
    # DONE

def trucate_sentence(orig_sent, TH=1000):
    """
        truncate insanely long sentences to its first TH (1000) chars
        SEMAFOR seem to have problems with really long stuff
        plus there's little point looking at captions that are just too long
    """
    wlist = orig_sent.split()
    curlen = 0
    for w in wlist:
        curlen += ( len(w)+1 )
        if curlen>TH:
            break
    
    trun_sent = " " 
    trun_sent += orig_sent[:curlen]
    return trun_sent

"""
    clean up caption (de-html, get rid of non ascii chars, etc)
    break into sentences, and compute bag-of-words (all) and bag-of-propositions (sentence)
"""  
def caption2sentence(in_txt, prepo_list=[], vocab=[], cursor=None, addl_vocab=[]):
    
    soup = BeautifulSoup(in_txt)
    txt_nolink = soup.get_text()  # does better than NLTK
    #txt_nolink = nltk.clean_html(in_txt) 
    txt_nolink = txt_nolink.replace("\n", " . ") #new line should trigger tokenizer too
    txt_nolink = txt_nolink.replace("\r", " ")
    txt_ascii = filter(lambda s: s in string.printable, txt_nolink)
    
    sents = sent_tokenizer.tokenize(txt_ascii)
    
    if len(txt_ascii.split()) >= 3:
        # counter for all word counts
        txt_cnter = Counter()
        
        #filter out sentences without alpha chars
        sents = filter(lambda st: len(set(st).intersection(list(string.ascii_letters)) ), sents)
        sent_feat = []
        #for k, st in enumerate(sents):
        for j, st in enumerate(sents):
            if len(st) > SENT_TH:
                sents[j] = trucate_sentence(st, SENT_TH)
                print "  truncate-sentence: from %d chars to %d chars" % (len(st), len(sents[j]))
            
            cur_str = sents[j];
            # tokenize and filter string, set wpairs
            tkn = nltk.word_tokenize(cur_str)
            tt = map(lambda s: \
                     norm_tag(s.lower(), cursor, addl_vocab=addl_vocab,filter_stopword=1), tkn)
            
            #for i, w in enumerate(tt):
            for w in tt:
                if len(w) and w in vocab:
                    txt_cnter[w] += 1
                    
            cur_feat = {}
            for p in prepo_list:
                num = cur_str.count(p)
                if num :
                    cur_feat[p] = num
                    cur_str.replace(p, "")
            
            sent_feat += [cur_feat]
            
        return sents, sent_feat, txt_cnter
    else:
        return([], [], {})


"""
    upgraded version of proc_caption above,
    now output sentence text for further extracting FrameNet features
"""  
def proc_caption(in_txt, prepo_list=[], vocab=[], cursor=None, addl_vocab=[]):
    
    soup = BeautifulSoup(in_txt)
    txt_nolink = soup.get_text()  # does better than NLTK
    #txt_nolink = nltk.clean_html(in_txt) 
    txt_nolink = txt_nolink.replace("\n", " ")
    txt_nolink = txt_nolink.replace("\r", " ")
    txt_ascii = filter(lambda s: s in string.printable, txt_nolink)
    
    sents = sent_tokenizer.tokenize(txt_ascii)
    #tokens = nltk.word_tokenize(txt_ascii)
    
    if len(txt_ascii.split()) >= 3:
        # counter for all word counts
        txt_cnter = Counter()
        
        # else
        sent_feat = []  
        wpairs = []      
        for k, st in enumerate(sents):
            cur_str = st;
            # tokenize and filter string, set wpairs
            tkn = nltk.word_tokenize(cur_str)
            tt = map(lambda s: \
                     norm_tag(s.lower(), cursor, addl_vocab=addl_vocab,filter_stopword=1), tkn)
            
            wpairs.append( [] )
            wpairs[k] = [(" ", "")]
            for i, w in enumerate(tt):
                if len(w) and w in vocab:
                    txt_cnter[w] += 1
                    
                    for j in range(i):
                        u = tt[j]
                        if len(u) and u in vocab:
                            if w < u:
                                wpairs[k] += [(w, u)]
                            else:
                                wpairs[k] += [(u, w)]
            wpairs[k].pop(0)
            
            cur_feat = {}
            for p in prepo_list:
                num = cur_str.count(p)
                if num :
                    cur_feat[p] = num
                    cur_str.replace(p, "")
            if cur_feat:
                sent_feat += [cur_feat]
            
        return wpairs, sent_feat, txt_cnter, sents
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

""" ----- END of process json generate sentence and BoW features """

def adhoc_translate(in_ascii_str):
    ss_nospace = in_ascii_str.translate(None, " ")
    ss_nospace = ss_nospace.replace("-LRB-", "(").replace("-RRB-", ")") # various brackets
    ss_nospace = ss_nospace.replace("-LSB-", "[").replace("-RSB-", "]")  
    ss_nospace = ss_nospace.replace("-LCB-", "{").replace("-RCB-", "}")  
    ss_nospace = ss_nospace.replace("``", '"').replace("''", '"') # quotes
    return ss_nospace


def gather_semafor_features(argv):    
    """
        parse semafor output that looks like the following
        output frame-wise features (bag-of-framenet), plus word pairs that are within a frame 
        
        <documents>
        ... 
        <sentence ID="5">
              <text>A slightly used Bronica SQai with a Sekonic Studio DeLuxe light meter .</text>
        ...
        <annotationset framename="Dimension" id="502">
         <layers>
          <layer id="50201" name="Target">
           <labels>
            <label end="62" id="5020101" name="Target" start="58">
            </label>
           </labels>
          </layer>
          <layer id="50202" name="FE">
           <labels>
            <label end="62" id="5020201" name="Dimension" start="58">
            </label>
            <label end="68" id="5020202" name="Object" start="64">
            </label>
           </labels>
          </layer>
         </layers>
        </annotationset>
        ...
        </sentence>
        ... ... 
    """
    if len(argv)<2:
        argv = ['-h']
    
    parser = OptionParser(description='parse flickr json files')
    parser.add_option("-i", "--sentence_dir", dest="sentence_dir", 
        default='', help="input dir containing sentence files")
    parser.add_option("-g", "--glob_str", dest="glob_str", 
        default='*.sentence', help="input file wild card string")
    parser.add_option("", "--sent_seg_str", dest="sent_seg_str", 
        default='.sent', help="additional suffix for sentence segment files")
    parser.add_option("-s", "--semafor_output", dest="semafor_output", 
        default='', help="input dir containing SEMAFOR xmls")
    parser.add_option("-o", "--out_dir", dest="out_dir", 
        default='', help="out dir for semafor features and word pairs")
    parser.add_option('-d', '--db_file', dest='db_file', 
        default='dict.db', help='file containing dictionary db')
    parser.add_option('-v', '--vocab', dest="vocab", default="")
    parser.add_option('-a', '--addl_vocab', dest="addl_vocab", default="")
    parser.add_option('-p', '--preposition_list', dest="preposition_list", default="")
    parser.add_option('-D', '--DEBUG', dest="DEBUG", type='int', default=0)
    
    opts, __ = parser.parse_args(argv)
    
    if opts.out_dir:
        out_dir = opts.out_dir
    else:
        out_dir = opts.sentence_dir
    
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
        print "read %d prepositions: %s \n" % (len(prepo_list), pp)
    else:
        prepo_list = []
    
    prepo_print = {}
    for p in prepo_list:
        prepo_print[p] = "_".join(p.split())
        
    conn = sqlite3.connect(opts.db_file)
    cursor = conn.cursor()
    
    #str_normalize = lambda s: s.translate(None, " ")
    
    in_file_list = glob(os.path.join(opts.sentence_dir, opts.glob_str))
    print os.path.join(opts.sentence_dir, opts.glob_str)
    print in_file_list
    
    for scnt, cur_file in enumerate(in_file_list):
        
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s processing sentence file %d-of-%d: %s \n" % (tt, scnt+1, len(in_file_list), cur_file)
        
        #_d, curn, _e = os.path.split(cur_file)  -- errs! 
        curn = os.path.splitext(os.path.split(cur_file)[1])[0]
        
        out_frame = os.path.join(out_dir, curn+".frame-feat")
        out_bagow = os.path.join(out_dir, curn+".frame-word")
        out_pairs = os.path.join(out_dir, curn+".frame-pair")
        ofh = open(out_frame, "wt")
        ofw = open(out_bagow, "wt")
        ofp = open(out_pairs, "wt") 
        
        fc = open(cur_file, "r")
        cur_fline = fc.readline().strip().split(None, 1)  # 603483869_01    By the tombstone.
        imgid,sentid = cur_fline[0].split("_")
        cur_sent = cur_fline[1].strip()
        
        # find files like 80.sent.[xx].xml
        xml_files = glob(os.path.join(opts.semafor_output, curn + opts.sent_seg_str + "*.xml"))
        xml_files.sort() 
        print xml_files
        
        frame_cnt = 0
        sents_cnt = 0
        empty_sent_cnt = 0
        mismatch_cnt = 0
        
        for semafor_file in xml_files:  # one file of n sentences
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s processing %s ..." % (tt, semafor_file)
                        
            soup = BeautifulSoup(open(semafor_file,'r').read())
            sentence_list = [c for c in soup.sentences.children]
            sentence_list = filter(lambda ss: type(ss) is bs4_element.Tag, sentence_list)
            
            for j, ss in enumerate(sentence_list): # foreach sentence
                
                #if not type(ss) is bs4_element.Tag:
                #    print "skip #%d:\n %s \t" % (j, ss)
                #    continue
                ss_semafor = ss.text.strip()
                
                # some ad-hoc processing to convert the two sentences to the same format
                ss_nospace = adhoc_translate ( ss_semafor.encode('ascii','ignore') )
                cs_translated = adhoc_translate( cur_sent )
                
                # check that this sentence is know           
                ind = ss_nospace.find( cs_translated )
                
                if ind < 0: # sentence do not match
                    mismatch_cnt += 1
                    print ("sentence #%d do not match:\n\t" % j) +\
                            cur_sent + '\n\t' + ss.text.strip() + "\n"
                elif ind > 0:
                    mismatch_cnt += 1
                    print ('sentence #%d err at loc %d:\n\t' % (j, ind)) + \
                            cur_sent + '\n\t' + ss.text.strip() + "\n"
                else: # ind == 0
                    annot_set = ss.find_all('annotationset')
                    if opts.DEBUG:
                        print "s#%d:\n" % j
                        print "\t" + cur_sent
                        print "\t" + ss.text.strip()
                    
                    """
                    [<layer id="50101" name="Target">
                    <labels>
                    <label end="68" id="5010101" name="Target" start="64"></label>
                    </labels>
                    </layer>,
                     <layer id="50102" name="FE">
                    <labels>
                    <label end="68" id="5010201" name="Unit" start="64"></label>
                    <label end="62" id="5010202" name="Count" start="58"></label>
                    </labels>
                    </layer>]
                    """
                    if not annot_set:
                        empty_sent_cnt += 1
                    
                    for k, cur_a in enumerate(annot_set):
                        framename = cur_a['framename']
                        label_list = cur_a.find_all('label')
                        lname  = map(lambda s: s['name'], label_list)
                        lstart = map(lambda s: int(s['start']), label_list)
                        lend   = map(lambda s: int(s['end'  ]), label_list)
                        fstart = min(lstart)
                        fend = max(lend)
                        cur_frame = ss_semafor[fstart : fend+1]
                        cur_frame_id = "%s_%s_%02d" % (imgid, sentid, k)
                        
                        tkn = nltk.word_tokenize(cur_frame)
                        ww = map(lambda s: norm_tag(s, cursor, addl_vocab=addl_vocab,filter_stopword=1), tkn)
                        ww = filter(lambda s: len(s)>1, ww)
                        ww = list(set(ww)) #unique words in vocab                       
                        
                        
                        cur_feat = {}
                        for p in prepo_list:
                            num = cur_frame.count(p)
                            if num :
                                cur_feat[p] = prepo_print[p]
                                cur_frame.replace(p, "")
                        """ write frame features to file: BoW, FE, word-pairs """
                        wlist = map(lambda s: s+":1", cur_feat.values() + ww)
                        ofw.write("%s %s\n" % (cur_frame_id, " ".join(wlist)) )
                        
                        fe_list = map(lambda s: s+":1", [framename] + lname )
                        ofh.write("%s %s\n" % (cur_frame_id, " ".join(fe_list)) )
                        
                        if len(ww) >= 2:
                            # extract bag-of-vocab-words and bag-of-prepositions
                            wpairs = [(" ", "")]
                            for i, w in enumerate(ww):
                                    for j in range(i):
                                        u = ww[j]
                                        if len(u) and u in vocab:
                                            if w < u:
                                                wpairs += [(w, u)]
                                            else:
                                                wpairs += [(u, w)]
                            wpairs.pop(0)
                            
                            for wp in wpairs:
                                ofp.write("%s %s,%s\n" % (cur_frame_id, wp[0], wp[1]) )
                            
                        else: # empty, no word-pairs
                            ofp.write("%s  \n" % cur_frame_id )
                        
                        frame_cnt += 1

                
                """ end of sentence loop
                    read newline from .sentence file 
                """
                sents_cnt += 1
                cur_fline = fc.readline().strip().split(None, 1)  # 603483869_01    By the tombstone.
                if cur_fline:
                    imgid,sentid = cur_fline[0].split("_")
                    cur_sent = cur_fline[1].strip()
                else:
                    break
                
                if sents_cnt % 500 == 0:
                    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                    print "%s %d sentences, %d mismatched, %d empty, %d frames found " % \
                        (tt, sents_cnt, mismatch_cnt, empty_sent_cnt, frame_cnt)
                    
                    if opts.DEBUG:
                        return
            
            # finish processing an xml file
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s done with xml file %s" % (tt, semafor_file)
            print "\t\t %d sentences, %d mismatched, %d empty, %d frames found \n" % \
                 (sents_cnt, mismatch_cnt, empty_sent_cnt, frame_cnt)
    
        ofh.close()
        ofw.close()
        ofp.close()
        
    return


def gather_wordpair_features(argv):
    return


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
    if '--collect_features' in argv:
        argv.remove('--collect_features')
        gather_wordpair_features(argv)
    elif '--collect_semafor' in argv:
        argv.remove('--collect_semafor')
        gather_semafor_features(argv)
    else:
        flickr_get_txt(argv)
    
    #if '--collect_tags' in argv:
    #    argv.remove('--collect_tags')
    #    collect_tags(argv)
    #elif '--collect_user' in argv:
    #    argv.remove('--collect_user')
    #    collect_user(argv)
        