#%%
import numpy as np
from collections import Counter
import itertools
from visual import show_tfidf

# s = np.array([0.0170559,0.01167856,0.03971621,0.        , 0.04501733, 
# 				  0.02045529 ,0.         ,0.0137465  ,0.03262757, 0.02099458 ,
# 					0.03135861, 0.01749381 ,0.02331898, 0.00792785 ,0.06542736])
# print(s.argsort())
# print(s.argsort()[-3:])
# print(s.argsort()[::-1])
# print(s.argsort()[-3:][::-1])
#%%

docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]

docs_words = [d.replace(",", "").split(" ") for d in docs]

vocab = set(itertools.chain(*docs_words))
v2i = {v: i for i, v in enumerate(vocab)}
i2v = {i: v for v, i in v2i.items()}


def safe_log(x):
    mask = x != 0
    x[mask] = np.log(x[mask])
    return x

#%%
tf_methods = {
    "log" : lambda x : np.log(1 + x),
    "augmented" : lambda x : 0.5 + 0.5 * x / np.max(x, axis = 1, keepdims = True),
    "boolean" : lambda x : np.minimum(x, 1),
    "log_avg": lambda x : (1 + safe_log(x)) / (1 + safe_log(np.mean(x, axis = 1, keepdims = True)))
}
idf_methods = {
    "log" : lambda x : 1 + np.log(len(docs) / (1 + x)),
    "prob" : lambda x : np.maximum(0, np.log((len(docs) - x) / (1 + x))),
    "len_norm": lambda x : x / (np.sum(np.square(x)) + 1)
}

def get_tf(method = "log"):
    _tf = np.zeros((len(vocab), len(docs)), dtype = np.float64)
    for i, d in enumerate(docs_words):
        counter = Counter(d)
        # print('counter = ', counter)
        for v in counter.keys():
            # print('counter.most_common(1)[0][1] = ', counter.most_common(1)[0][1])
            # print('counter.most_common(1) = ', counter.most_common(1))
            # print('counter.most_common(1)[0] = ', counter.most_common(1)[0])
            _tf[v2i[v], i] = counter[v] / counter.most_common(1)[0][1]

    weighted_tf = tf_methods.get(method, None)
    if weighted_tf is None:
        raise ValueError
    return weighted_tf(_tf)

def get_idf(method = "log"):
    df = np.zeros((len(i2v), 1))
    for i in range(len(i2v)):
        d_count = 0
        for d in docs_words:
            d_count += 1 if i2v[i] in d else 0
            df[i, 0] = d_count
    idf_fn = idf_methods.get(method, None)
    if idf_fn is None:
        raise ValueError
    return idf_fn(df)

def cosine_similarity(q, _tf_idf):
    # print('_tf_idf : ',_tf_idf)
    # print('_tf_idf.shape : ',_tf_idf.shape)
    unit_q = q / np.sqrt(np.sum(np.square(q), axis = 0, keepdims = True))
    unit_ds = _tf_idf / np.sqrt(np.sum(np.square(_tf_idf), axis = 0, keepdims = True))
    similarity = unit_ds.T.dot(unit_q).ravel()
    return similarity

def docs_score(q, len_norm = True):
    q_words = q.replace(",", "").split(" ")

    unknown_v = 0
    for v in set(q_words):
        if v not in v2i:
            v2i[v] = len(v2i)
            i2v[len(v2i) - 1] = v
            unknown_v += 1
    if unknown_v > 0:
        _idf = np.concatenate((idf, np.zeros((unknown_v, 1), dtype = np.float)), axis = 0)
        _tf_idf = np.concatenate((tf_idf, np.zeros((unknown_v, tf_idf.shape[1]), dtype = np.float)), axis = 0)
        # print('_idf: ',_idf)
        # print('_idf.shape :', _idf.shape)
    else :
        _idf, _tf_idf = idf, tf_idf
    counter = Counter(q_words)
    q_tf = np.zeros((len(_idf), 1), dtype = np.float)
    for v in counter.keys():
        q_tf[v2i[v], 0] = counter[v]
        # print('counter[v]:', counter[v])

    q_vec = q_tf * _idf
    # print('q_tf: ', q_tf)
    # print('q_tf.shape: ', q_tf.shape)
    # print('q_vec: ', q_vec)
    # print('q_vec.shape: ', q_vec.shape)
    q_scores = cosine_similarity(q_vec, _tf_idf)
    # print('q_scores(pre): ', q_scores)
    # print('q_scores.shape: ', q_scores.shape)
    if len_norm:
        len_docs =[len(d) for d in docs_words]
        # print('len_docs:', len_docs)
        q_scores = q_scores / np.array(len_docs)
        # print('q_scores: ', q_scores)
        # print('q_scores.shape: ', q_scores.shape)
    return q_scores

def get_keywords(n = 2):
    for c in range(3):
        col = tf_idf[:, c]
        # print('col:', col)
        # print('col.shape:', col.shape)
        idx = np.argsort(col)[-n:]
        idx2 = np.argsort(col)[-n]
        # print('idx (1): ', idx)
        # print('idx (2):', idx2)
        # print("doc{}, top{} keywords {}".format(c, n, [i2v[i] for i in idx]))

tf = get_tf()
#%%
idf = get_idf()
tf_idf = tf * idf
# print('tf_idf: ', tf_idf)
# print('tf_idf.shape: ',tf_idf.shape)
# print("tf shape(vecb in each docs) : ", tf.shape)
# print("\ntf samples :\n", tf[:2])
# print("\nidf shape(vecb in all docs) : ", idf.shape)
# print("\nidf samples :\n", idf[:2])
# print("\ntf_idf shape : ", tf_idf.shape)
# print("\ntf_idf sample :\n", tf_idf[:2])

# Test
get_keywords()
q = "I get a coffee cup"
scores = docs_score(q)
print('scores: ',scores)
print('scores.shape: ',scores.shape)
d_ids = scores.argsort()[-3:][::-1]
print('d_ids: ', d_ids)
print("\ntop 3 docs for '{}' : \n {}".format(q, [docs[i] for i in d_ids]))
#%%
show_tfidf(tf_idf.T, [i2v[i] for i in range(tf_idf.shape[0])], "tfidf_matrix")