import flask
import numpy as np
import random
import pickle
import string
from gensim import corpora, models, similarities

###-------PULL LDA AND SIMILARITY MODELS FROM PICKLED FILES-------###
dictionary = corpora.Dictionary.load('../my_dict.dict')

with open('../documents.pkl', 'r') as f:
    documents = pickle.load(f)

with open('../successful_post_and_comment_dicts.pkl', 'r') as f:
    post_dict, comment_dict = pickle.load(f)

with open('../lda_models.pkl', 'r') as g:
    lda_models = pickle.load(g)

with open('../post_titles.pkl', 'r') as g:
    post_titles = pickle.load(g)

with open('../two_way_post_comment_dict.pkl', 'r') as h:
    com_to_post, post_to_com = pickle.load(h)

with open('../doc_indices_to_ids.pkl', 'r') as q:
    (index_to_post_id_dict, index_to_comment_id_dict) = pickle.load(q)

comment_id_to_index_dict = {com_id:index for index, com_id in index_to_comment_id_dict.items()}

indices = []

for i in [2, 10, 50, 100]:
    indices.append(similarities.MatrixSimilarity.load('../lda_corpus_with_'
                                                      +str(i)
                                                      +'_similarities.index'))



###-------END SETUP-------###

###-------SIMILARITY FUNCTION-------###

def get_av_sim_docs(s, threshold = .6):
    sims, av_sims, sim_posts, sim_comments = [], [], [], []
    for i in range(4):
        sims.append(indices[i][lda_models[i][dictionary.doc2bow(s.lower().split())]])
    for j in range(len(sims[0])):
        av_sims.append(np.mean([sims[k][j] for k in range(4)]))
    n_best = filter(lambda item: item[1] > threshold and item[0] < 513, sorted(enumerate(av_sims), key = lambda item: -item[1]))
    if n_best:
        return random.choice(n_best)
    return (-1, -1)

###-------END SIMILARITY FUNCTION-------###


app = flask.Flask(__name__)

@app.route('/')
def home_page():
    
    with open('index.html', 'r') as f:
        return f.read()

@app.route('/search', methods = ['POST'])
def do_search():
    
    data = flask.request.json
    
    print data

    s = data['example']

    s = ' '.join(map(lambda x: x.strip(string.punctuation + string.whitespace), 
                     s.split()))

    best_match = get_av_sim_docs(s)
    if best_match[0] == -1:
        json_to_return = {"results": ["No similar resolved arguments.", 
                                      "Try again!"]}
    else:
        results = [str(100 * best_match[1]) + '% match ', "Title: " + post_titles[best_match[0]],'OP: '+documents[best_match[0]]]
        ind = index_to_post_id_dict[best_match[0]]
        count = 1
        for com in post_to_com[ind]:
            results.append('Comment ' + str(count) + ': ' + documents[comment_id_to_index_dict[com]])
            count += 1
        json_to_return = {"results": results}

    return flask.jsonify(json_to_return)

app.run(debug = True)
