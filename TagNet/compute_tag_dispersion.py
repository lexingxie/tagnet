

import sys,os
import shelve
import sqlite3
import pickle
from nltk.corpus import wordnet as wn

from datetime import datetime
from optparse import OptionParser

PRINT_FLAG = 1

def get_wn_similarity(shelve_name, wn_list, offset_dict):
    idx_list = gen_pairs_idx(len(wn_list))
    wn_sim = []
    cache_cnt = 0
    new_cnt = 0
    
    sim = shelve.open(shelve_name, writeback=False)
    for i, j in idx_list:
        if wn_list[i] < wn_list[j]:
            k1 = offset_dict[wn_list[i]]
            k2 = offset_dict[wn_list[j]]
        else:
            k1 = offset_dict[wn_list[j]]
            k2 = offset_dict[wn_list[i]]
         
        if k1 in sim and k2 in sim[k1]:
            s = sim[k1][k2]
            cache_cnt += 1
        else:
            try:
                w1 = wn.synset(k1)
                w2 = wn.synset(k2)
            except:
                print wn_list[i], wn_list[j]
                print k1, k2
            s = w1.path_similarity(w2)
            if k1 not in sim:
                sim[k1] = {}
            sim[k1][k2] = s
            new_cnt += 1
        
        wn_sim.append(s)
    
    sim.close()
    
    if PRINT_FLAG:
        tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        print "%s   %d synsets, %d pairs, %d cached, %d new" % (tt, len(wn_list), len(wn_sim), cache_cnt, new_cnt)
    
    return wn_sim

def gen_pairs_idx(num_items):
    idx_list = []
    for i in range(num_items):
        for j in range(i):
            idx_list.append((i,j))
    return idx_list

def store_wn_lookup():
    """
        build a wordnet offset-synset dict, 
        as suggested in this SO post
        http://stackoverflow.com/questions/8077641/wordnet-synset-offset/12378481#12378481
        .. this is a hidden function and actually executed from ipython
    """
    syns = list( wn.all_synsets() )
    #syn_str = map(lambda s: str(s).replace("Synset",'').strip('()').strip("'"), syns)
    syn_str = map(lambda s: str(s).replace("Synset",'').strip('()').strip("'").strip('"'), syns)
    #offsets_list = [("n%08d" % s.offset, s) for s in syns]
    olist = map(lambda a, b: ("n%08d" % a.offset, b), syns, syn_str)
    offset_dict = dict(olist)
    pickle.dump(offset_dict, open('/Users/xlx/Documents/proj/imgnet-flickr/db3/wn_offset_dict.pickle', 'wb'))

def compute_dispersion(argv):

    if len(argv)<2:
        argv = ['-h']
    
    parser = OptionParser(description='construct+compare conceptnet and flickr word similarities')
    parser.add_option('-d', '--db_dir', dest='db_dir', default="", help='dir containing sqlite db files')
    parser.add_option('-o', '--out_dir', dest='out_dir', default="", help='output dir')
    parser.add_option('-f', '--db_file', dest='db_file', default="wordnet_tag.db", help='dictionary')
    parser.add_option('-s', '--shelve_file', dest='shelve_file', default="tag_dispersion.shelve", help='name of the cache file')
    parser.add_option('', '--output_file', dest='output_file', default="tag_features.csv", help='name of the ouptut file')
    
    parser.add_option('', '--offset_file', dest='offset_file', default='wn_offset_dict.pickle' )
    
    parser.add_option('', '--start_idx', dest='start_idx', type='int', default=0, help='start index of tags')
    parser.add_option('', '--end_idx', dest='end_idx', type='int', default=10)
    parser.add_option('', '--topK', dest='topK', type='int', default=-1, help="only consider topK wn per tag, to speed things up")
    
    (opts, __args) = parser.parse_args(argv)
    
    if not opts.out_dir:
        opts.out_dir = opts.db_dir
    
    shelve_file = os.path.join(opts.out_dir, opts.shelve_file) 
    out_file = os.path.join(opts.out_dir, opts.output_file)
    fo = open(out_file, 'wt')
    
    offset_dict = pickle.load(open(os.path.join(opts.db_dir, opts.offset_file), "rb"))
    
    db_file = os.path.join(opts.db_dir, opts.db_file)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    tlist = cursor.execute("SELECT DISTINCT tag FROM wn_tag").fetchall() # list of tags
    
    cnt = 0
    nt = len(tlist)
    
    for tag in tlist:
        cnt += 1
        if cnt < opts.start_idx or cnt > opts.end_idx:
            continue
        # get the list of associated wns and their count
        tag = tag[0]
        stmt = "SELECT wnid, count FROM wn_tag WHERE tag='%s'" % tag
        tag_assoc = cursor.execute(stmt).fetchall()
        tag_assoc.sort(key=lambda x: x[1], reverse=True)
        if opts.topK > 0 and opts.topK<len(tag_assoc) :
            tag_assoc = tag_assoc[:opts.topK]
        
        tmp = zip(*tag_assoc) 
        wn_list = tmp[0]
        wn_count = tmp[1]
        
        PRINT_FLAG = (cnt % 20 == 0)
         
        idx_list = gen_pairs_idx(len(wn_list))
        if idx_list:
            wn_sim = get_wn_similarity(shelve_file, wn_list, offset_dict)
            
            pair_total = map(lambda u:  1.*wn_count[u[0]]*wn_count[u[1]], idx_list)
            npair = sum(pair_total)
            avg_sim = sum(map(lambda a,b: a*b, pair_total, wn_sim)) / npair
            
            if PRINT_FLAG:
                tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
                print "%s tag#%d/%d %s, sim: %0.4f, #pairs %d " % (tt, cnt, nt, tag, avg_sim, npair)
            fo.write("%s,%0.4f\n" % (tag, avg_sim) )
        else:
            tt = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
            print "%s tag#%d/%d %s, empty" % (tt, cnt, nt, tag)
    
    fo.close()
    conn.close()
    return


if __name__ == '__main__':
    argv = sys.argv  
    compute_dispersion(argv)
    