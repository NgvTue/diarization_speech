import tensorflow as tf
from itertools import permutations
def faster_pit_loss_cross_entropy(k=4):
    perms = list(permutations(range(k)))
    N=len(perms)
    perms = [j for i in perms for j in i]
    perms = tf.constant(perms,dtype=tf.int32)
    @tf.function
    def batch_pit_loss(y_true, y_pred):
        """y_true: bs,len,num_classes
        y_pred: bs,len,num_classes
        """
        print("tracing mode\nThis warning shoudln't be raise more than one")
        y_true = tf.cast(y_true, y_pred.dtype)
        _,len,cl = y_true.shape 
        list_lost_per_st = tf.gather(y_true, perms, axis=-1)
        list_lost_per_st = tf.reshape(y_true, [-1,len,cl, N])
        y_pred = tf.expand_dims(y_pred,-1)
        y_pred = tf.tile(y_pred, [1,1,1,N])
        list_loss = tf.nn.sigmoid_cross_entropy_with_logits(list_lost_per_st, y_pred)
        list_loss = tf.math.reduce_mean(list_loss,2)
        list_loss = tf.math.reduce_mean(list_loss,1)
        return tf.math.reduce_mean(tf.math.reduce_min(list_loss,axis=-1))
    
    return batch_pit_loss
def pit_loss(loss_fn, k=4):
    perms = list(permutations(range(k)))
    # perms = tf.constant(perms,dtype=tf.int32)
    @tf.function
    def batch_pit_loss(y_true, y_pred):
        """y_true: bs,len,num_classes
        y_pred: bs,len,num_classes
        """
        print("tracing mode\nThis warning shoudln't be raise more than one")
        #y_true = tf.reshape(y_true,(-1,cl))
        #y_pred = tf.reshape(y_pred, (-1,cl))
        y_true = tf.cast(y_true, y_pred.dtype)
        list_lost_per=[]
        for perm in perms:
            print("tracing for loop permutation")
            y_pred_per = tf.gather(y_true, perm, axis=-1)
            list_lost_per.append(
                loss_fn(
                    y_pred_per,y_pred
                )
            )
        # loss_all = list_lost_per[0]
        # for lo in list_lost_per:
            # loss_all = tf.math.minimum(lo,loss_all)
        # return tf.math.reduce_mean(loss_all)
        list_lost_per_st = tf.stack(list_lost_per,-1)
        return tf.math.reduce_mean(tf.math.reduce_min(list_lost_per_st,axis=-1))
    
    return batch_pit_loss
    

def map_fn_cache_pit(k=5):
    perms = list(permutations(range(k)))
    @tf.function
    def fun(x,y):
        list_y=[]
        for perm in perms:
            # print("tracing for loop permutation")
            y_per = tf.gather(y, perm, axis=-1)
            list_y.append(
                y_per
            )
        return x,tf.stack(list_y,-1)
    return fun
def cache_pit(list_lost_per_st, y_pred):
    #list_lost_per_st = tf.stack(list_lost_per,-1)
    list_lost_per_st=tf.cast(list_lost_per_st, y_pred.dtype)
    y_pred = tf.expand_dims(y_pred,-1)
    y_pred = tf.tile(y_pred, [1,1,1,list_lost_per_st.shape[-1]])
    list_loss = tf.nn.sigmoid_cross_entropy_with_logits(list_lost_per_st, y_pred)
    list_loss = tf.math.reduce_mean(list_loss,2)
    list_loss = tf.math.reduce_mean(list_loss,1)
    return tf.math.reduce_mean(tf.math.reduce_min(list_loss,axis=-1))