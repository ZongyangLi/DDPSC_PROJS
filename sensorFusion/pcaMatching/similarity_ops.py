import numpy as np
from numpy import matlib as mb # matlib must be imported separately

def compute_spatial_similarity(conv1,conv2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
    conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    im_similarity = np.zeros((conv1_normed.shape[0],conv1_normed.shape[0]))
    for zz in range(conv1_normed.shape[0]):
        repPx = mb.repmat(conv1_normed[zz,:],conv1_normed.shape[0],1)
        im_similarity[zz,:] = np.multiply(repPx,conv2_normed).sum(axis=1)
    similarity1 = np.reshape(np.sum(im_similarity,axis=1),out_sz)
    similarity2 = np.reshape(np.sum(im_similarity,axis=0),out_sz)
    return similarity1, similarity2


def combine_SVD(conv1,conv2):

    conv1 = np.reshape(conv1, (-1,2048))
    conv2 = np.reshape(conv2, (-1,2048))
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    conv1 = conv1 - pool1
    conv2 = conv2 - pool2
    allconv = np.vstack((conv1,conv2))
    allconv =(allconv.T-allconv.mean(axis=1)).T
    u, _, _ = np.linalg.svd(allconv, full_matrices=False)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))    
    r1 = np.reshape(u[:,0][0:conv1.shape[0]],out_sz)
    g1 = np.reshape(u[:,1][0:conv1.shape[0]],out_sz)
    b1 = np.reshape(u[:,2][0:conv1.shape[0]],out_sz)
    r2 = np.reshape(u[:,0][conv1.shape[0]:],out_sz)
    g2 = np.reshape(u[:,1][conv1.shape[0]:],out_sz)
    b2 = np.reshape(u[:,2][conv1.shape[0]:],out_sz)    
    r1 = 0.5+0.5*r1/(np.absolute(r1).max())
    g1 = 0.5+0.5*g1/(np.absolute(g1).max())
    b1 = 0.5+0.5*b1/(np.absolute(b1).max())
    res1 = np.array([r1,g1,b1])    
    r2 = 0.5+0.5*r2/(np.absolute(r2).max())
    g2 = 0.5+0.5*g2/(np.absolute(g2).max())
    b2 = 0.5+0.5*b2/(np.absolute(b2).max())
    res2 = np.array([r2,g2,b2])    
    res1 = np.ascontiguousarray(res1.transpose(1, 2, 0))*255
    res2 = np.ascontiguousarray(res2.transpose(1, 2, 0))*255
    return res1,res2

def SVD_whole(convlist):
    allconv = np.zeros(convlist[0].shape)
    leng = len(convlist)
    for convmap in convlist:
        pool = np.mean(convmap,axis=0)
        convmap = convmap / np.linalg.norm(pool) / convmap.shape[0]
        #convmap = (convmap.T-convmap.mean(axis=1)).T
        allconv = np.vstack((allconv,convmap))
        
    allconv = allconv[convlist[0].shape[0]:,:]
    allconv =(allconv-allconv.mean(axis=0))
    u, _, _ = np.linalg.svd(allconv, full_matrices=False)
#     u1, s, v= np.linalg.svd(conv1, full_matrices=False)
#     u2, _, _ = np.linalg.svd(conv2, full_matrices=False)
    out_sz = (int(np.sqrt(convmap.shape[0])),int(np.sqrt(convmap.shape[0])))
    #newconv1_r = u1[0]*s[0]*v[0]    
    reslist = []
    for i in range(0,leng):
        r = np.reshape(u[:,0][convmap.shape[0]*i:convmap.shape[0]*(i+1)],out_sz)
        g = np.reshape(u[:,1][convmap.shape[0]*i:convmap.shape[0]*(i+1)],out_sz)
        b = np.reshape(u[:,2][convmap.shape[0]*i:convmap.shape[0]*(i+1)],out_sz)
        # r = r-r.min()
#         r = r/r.max()
#         g = g-g.min()
#         g = g/g.max()
#         b = b-b.min()
#         b = b/b.max()
        r = 0.5+0.5*r/(np.absolute(r).max())
        g = 0.5+0.5*g/(np.absolute(g).max())
        b = 0.5+0.5*b/(np.absolute(b).max())
        res = np.array([r,g,b])
        #res = np.array([r,g,b])
        res = np.ascontiguousarray(res.transpose(1, 2, 0))*255
        reslist.append(res)    
    return reslist