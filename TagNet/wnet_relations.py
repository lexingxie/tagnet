


import sys,os
import sqlite3
#import codecs
import numpy as np
#import math
#import gzip

from scipy import io
from datetime import datetime
#from optparse import OptionParser

"""
    imports from own code, and NLTK/conceptnet that may fail on some machines
"""
from wnet_tag_topics import options_get_wnet_tag, get_wnet_tags,binary_mutual_info
from tag_popular_test import compile_synset_wordlist, print_synset_info
        
try: 
    #from imgtags import cache_flickr_info
    from count_bigram_flickr import norm_tag #read_unigram, read_bigram(src_file, bg_dict)
    from count_bigram_flickr import accumulate_bg, sort_bg
except:
    print "WARNING: some custom lib imports failed!"
    
try:        
    from nltk.corpus import wordnet as wn
    import conceptnet.models as cm
    en = cm.Language.get('en')
except:
    print "WARNING: some imports failed!"
    
"""
    aggregate over a few input synsets to get bigram
"""    
def construct_training_data(argv):
    opts, db_dict, _addl_vocab, _db_wn = options_get_wnet_tag(argv)
    addl_vocab = []
    dbcn = sqlite3.connect(db_dict) # dictionary db
    dbcsr = dbcn.cursor()
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s start processing '%s' " % (tt, opts.in_wnet_list)
    
    if os.path.isfile(opts.in_wnet_list):
        wnet_list = open(opts.in_wnet_list, 'rt').read().split()
    else:
        wnet_list = opts.in_wnet_list.split(",")
        
    imgid_list, usr_list, tag_dict = ([], [], {})
    
    for (iw, wn) in enumerate(wnet_list):
        if iw > opts.endnum:
            break
        
        wtag_file = os.path.join(opts.data_home, opts.wnet_list_dir+"_tags", wn+'.tags.txt')
        ilist, ulist, tdict = get_wnet_tags(wn, wtag_file, opts, False)
        imgid_list += ilist
        usr_list += ulist
        for u in tdict:
            tag_dict[u] = tdict[u] # imgid is unique 
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "\n%s read %d synsets, found %d imgs from %d unqiue usrs" % (tt, len(wnet_list), len(imgid_list), len(set(usr_list)) )
    tnum = map(lambda k: len(tag_dict[k]), tag_dict.iterkeys())
    snum = sum(tnum)
    print "%s %d tags in %d imgs, avg %0.4f tags per img" % (tt, snum, len(imgid_list), 1.*snum/len(tnum))
    
    #if 0:
    usr_tag = {}
    tag_cnt = {}
    empty_cnt = 0
    
    for (imid, u) in zip(imgid_list, usr_list):
        if u not in usr_tag:
            usr_tag[u] = {}
        
        vv = map(lambda s:norm_tag(s, dbcsr, addl_vocab,1), tag_dict[imid])
        vv = list(set(filter(len, vv)))
        for v in vv:
            if v in usr_tag[u]:
                usr_tag[u][v] += 1
            else:
                usr_tag[u][v] = 1 # first time seeing user u use tag v
                if v in tag_cnt:
                    tag_cnt[v] += 1
                else:
                    tag_cnt[v] = 1
        
    tag_list = filter(lambda t: tag_cnt[t]>1, tag_cnt.keys())
    tag_list.sort()
    tag_num = map(lambda t: tag_cnt[t], tag_list)
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s obtained %d usrs, %d out of %d tags with cnt>1, %d empty imgs " % (tt, len(usr_tag), len(tag_list), len(tag_cnt), empty_cnt)
    
    lcnt = 0
    bg_dict = {}
    #wn_tagf = open(os.path.join(opts.data_home, opts.wnet_out_dir, wn+".txt"), "wt")
    for u in usr_tag:
        vv = filter(lambda v: v in tag_list, usr_tag[u].keys())
        if vv:
            #wn_tagf.write(u+"\t"+ ",".join(vv) + "\n")
            accumulate_bg(vv, bg_dict, None, None, addl_vocab=[])
            lcnt += 1
            if lcnt<20: print u+"\t"+ ",".join(vv)
    
    #Nusr = len(usr_tag)
    lcnt = 0
    bg_tuples = sort_bg(bg_dict)
    #print bg_tuples
    bigram_list = []
    for u, v, c in bg_tuples:
        iu = tag_list.index(u)
        iv = tag_list.index(v)
        bigram_list += [iu, iv, c]
    
    wn_out_mat = os.path.split(opts.in_wnet_list)[1]
    wn_out_mat = os.path.splitext(wn_out_mat)[0]
    wn_out_mat = os.path.join(os.path.split(opts.in_wnet_list)[0], wn_out_mat+".mat")
    data = {'wnet_list': wnet_list, 'usr_list': ulist, 'tag_list': tag_list, 'tag_cnt':tag_num, 'bigram_list': bigram_list}
    io.savemat(wn_out_mat, data)
    
        
"""
    get bigram stats and other stats for each synset
"""    
def analyze_tag_pairs(argv):
    # parser = OptionParser(description='compile tags for all imgs in a wnet synset')
    opts, db_dict, addl_vocab, db_wn = options_get_wnet_tag(argv)
    
    tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "%s start processing '%s' " % (tt, opts.in_wnet_list)
    
    dbcn = sqlite3.connect(db_dict) # dictionary db
    dbcsr = dbcn.cursor()
    
    if os.path.isfile(opts.in_wnet_list):
        wnet_list = open(opts.in_wnet_list, 'rt').read().split()
    else:
        wnet_list = opts.in_wnet_list.split(",")
        
    for (iw, wn) in enumerate(wnet_list):
        if iw > opts.endnum:
            break
        
        wtag_file = os.path.join(opts.data_home, opts.wnet_list_dir+"_tags", wn+'.tags.txt')
        imgid_list, usr_list, tag_dict = get_wnet_tags(wn, wtag_file, opts, False)
        
        usr_tag = {}
        tag_cnt = {}
        empty_cnt = 0
        
        #for u, utuple in groupby(zip(usr_list, imgid_list), lambda x: x[0]):
        for (imid, u) in zip(imgid_list, usr_list):
            if u not in usr_tag:
                usr_tag[u] = {}
                    
            vv = map(lambda s:norm_tag(s, dbcsr, addl_vocab,1), tag_dict[imid])
            vv = list(set(filter(len, vv)))
            for v in vv:
                if v in usr_tag[u]:
                    usr_tag[u][v] += 1
                else:
                    usr_tag[u][v] = 1 # first time seeing user u use tag v
                    if v in tag_cnt:
                        tag_cnt[v] += 1
                    else:
                        tag_cnt[v] = 1
            
            #for v in list(set(usr_tag[u])):
                
        
        tag_list = filter(lambda t: tag_cnt[t]>1, tag_cnt.keys())
        tag_list.sort()
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s obtained %d usrs, %d tags non-trival, %d empty imgs " % (tt, len(usr_tag), len(tag_list), empty_cnt)
        
        ulist = []
        tcnt_mat = np.zeros( (len(usr_list), len(tag_list)))
        icnt = 0
        #out_dat_fh = open(os.path.join(opts.data_home, opts.wnet_out_dir, wn+".dat"), "wt")
        for u in list(set(usr_list)):
            jj = -1
            outstr = ""
            if usr_tag[u]:
                for t in usr_tag[u]:
                    if t in tag_list:
                        jj  = tag_list.index(t)
                        tcnt_mat[icnt, jj] += 1
                        outstr += "%d:%d " % (jj+1, usr_tag[u][t])
            else:
                pass # empty list of tags, skip
           
            if outstr:
                #out_dat_fh.write( outstr + "\n" )
                icnt += 1
                ulist.append(u)
        
        tcnt = tcnt_mat[:icnt, :]
        
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print '%s wnet "%s": %d unique tags by %d users\n' % (tt, wn, len(tag_list), len(ulist))
        
        # get and print synset info
        td = dict([ (k, tag_cnt[k]) for k in tag_list ])
        syn_info, wlist = compile_synset_wordlist(wn, len(ulist), None, db_wn, db_dict, addl_vocab, td)
        hi_words = [syn_info['self']['words']] + syn_info['ancestor'].values() + syn_info['descendant'].values()
        hi_depth = [0 ] + syn_info['ancestor'].keys() + syn_info['descendant'].keys()
        other_words = list(set(tag_list) - set(reduce(lambda a,b: a+b, hi_words, [])) )
        print_synset_info(syn_info, wn, wlist, len(ulist))
        
        
        print "\n tag occurrence:"
        hi_words = reduce(lambda a,b: a+b, hi_words, [])
        lcnt = 0
        bg_dict = {}
        wn_tagf = open(os.path.join(opts.data_home, opts.wnet_out_dir, wn+".txt"), "wt")
        for u in usr_tag:
            vv = filter(lambda v: v in tag_list, usr_tag[u].keys())
            if vv:
                wn_tagf.write(u+"\t"+ ",".join(vv) + "\n")
                accumulate_bg(vv, bg_dict, None, None, addl_vocab=[])
                lcnt += 1
                if lcnt<20: print u+"\t"+ ",".join(vv)
        wn_tagf.close()
        
        print "\nbigrams#\tMI \ttype\ttag1,tag2\trelations"
        Nusr = len(usr_tag)
        lcnt = 0
        bg_tuples = sort_bg(bg_dict)
        wn_bgf = open(os.path.join(opts.data_home, opts.wnet_out_dir, wn+".bigram.txt"), "wt")
        
        mi_list = [] 
        freq_list = []
        rel_list = []
        rel_flag = []
        for u, v, c in bg_tuples:
            iu = tag_list.index(u)
            iv = tag_list.index(v)
            assr = cm.Assertion.objects.filter(concept1__text=u, concept2__text=v,language=en)
            atxt = map(lambda a: str(a).strip("[]"), assr)
            atxt += map(lambda a: str(a).strip("[]"), 
                        cm.Assertion.objects.filter(concept1__text=v, concept2__text=u,language=en))
            if atxt:
                rlt_val = 1
            else:
                rlt_val = 0
                rel_list += [iu, iv, rlt_val]
                
            #if c<3: 
            #    if rlt_val:
            #        rel_list += [iu, iv, -rlt_val]  # has relation but no data
            #elif rlt_val:
            #    rel_list += [iu, iv, rlt_val]
                
            try:
                mi = binary_mutual_info(Nusr, tag_cnt[u], tag_cnt[v], c)
            except:
                print u, v, c, len(usr_tag)
                raise
            
            # record data
            mi_list += [iu, iv, mi]
            freq_list += [iu, iv, c]
            
            if u in other_words and v in other_words:
                btype = "OO"
            elif u in hi_words and v in hi_words:
                btype = "HH"
                rel_flag += [iu, iv]
                continue  # ignore two words that are already both in the hierarchy
            else:
                btype = "HO"
                
            outstr = "%s\t%0.5f\t%0.5f\t%s\t%s,%s\t%s" % \
                (syn_info['nltk_id'], 1.*c/Nusr, mi, btype, u, v, ";".join(atxt) if atxt else "None")
            wn_bgf.write(outstr +"\n")
            
            lcnt += 1
            if lcnt<50: print outstr
            
        wn_bgf.close()
        
        wn_out_mat = os.path.join(opts.data_home, opts.wnet_out_dir, wn+".mat")
        tcnt = tcnt_mat[:icnt, :]
        #mlab.save(wn_out_mat, 'img_list', 'tag_list', 'tcnt_mat')
        data = {'usr_list': ulist, 'tag_list': tag_list, 'tcnt':tcnt, 
                'other_words':other_words, 'hier_words':hi_words, 'hier_depth':hi_depth, 
                'mi': mi_list, 'freq': freq_list, 'csubnet':rel_list}
        io.savemat(wn_out_mat, data)
        
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print '%s saved to %s \n\n' % (tt, wn_out_mat)
        
        
if __name__ == '__main__':
    argv = sys.argv 
    if "--construct_training_data" in argv:
        argv.remove("--construct_training_data")
        construct_training_data(argv)
    else:
        analyze_tag_pairs(argv)
    """
    if "--analyze_tag_pairs" in argv:
        argv.remove("--analyze_tag_pairs")
        analyze_tag_pairs(argv)
    else:
        #map_wn_words(argv)
        pass        
    """  