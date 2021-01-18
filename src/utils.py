import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
#Pre-process img
def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

#L2 distance
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

#Calculate embs
def calc_embs(model, imgs, batch_size):
    aligned_images = prewhiten(imgs)
    # print(aligned_images)
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))
    return embs

#cos distance
def most_similarity(embed_vecs, vec, labels):
    sim = cosine_similarity(embed_vecs, vec)
    sim = np.squeeze(sim, axis = 1)
    argmax = np.argsort(sim)[::-1][:10]
    label = [labels[idx] for idx in argmax]
    name = Counter(label)
    tmp_sum = sum(sorted(sim, reverse=True)[0:10])
    prob = tmp_sum/10
    return name.most_common(1)[0][0], prob

#get label from classify network
def get_label_classify(preds, lb):
    j = np.argmax(preds)
    proba = preds[j]
    a_label = [1 if n == proba else 0 for n in preds]
    return lb.classes_[j], proba