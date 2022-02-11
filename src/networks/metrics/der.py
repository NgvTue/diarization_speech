
import tensorflow as tf
from itertools import permutations
def pit_per(k=4):
   perms = list(permutations(range(k)))
    # perms = tf.constant(perms,dtype=tf.int32)
   @tf.function
   def batch_pit_der(y_true, y_pred):
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
               der(
                  y_pred_per,y_pred
               )
         )
      # loss_all = list_lost_per[0]
      # for lo in list_lost_per:
         # loss_all = tf.math.minimum(lo,loss_all)
      # return tf.math.reduce_mean(loss_all)
      list_lost_per_st = tf.stack(list_lost_per,-1)
      return tf.math.reduce_min(list_lost_per_st,axis=-1)
   
   return batch_pit_der

@tf.function
def der(y_true, y_pred):
    """y_true: bs,len,s
       y_pred: bs,len,s
    """
    # permutation x
    y_true = tf.cast(y_true, tf.int32)
    decisions =tf.where(tf.nn.sigmoid(y_pred)> 0.5,1,0)
    decisions = tf.cast(decisions,tf.int32)
    n_ref =tf.math.reduce_sum(y_true,-1) #bs,len
    n_sys =tf.math.reduce_sum(decisions,-1)
    res = {}
    res['speech_scored'] =  tf.math.reduce_sum(tf.where(n_ref > 0,1,0),-1)
    res['speech_miss'] = tf.math.reduce_sum(tf.where( tf.logical_and (n_ref>0,n_sys == 0) , 1, 0), -1)
    res['speech_falarm'] =tf.math.reduce_sum(tf.where(tf.logical_and(n_ref == 0 , n_sys > 0), 1, 0 ), -1)
    res['speaker_scored'] = tf.cast(tf.math.reduce_sum(n_ref,-1),tf.float32)
    res['speaker_miss'] = tf.math.reduce_sum(tf.maximum(n_ref - n_sys, 0),-1)
    res['speaker_falarm'] = tf.math.reduce_sum(tf.maximum(n_sys - n_ref, 0),-1)
    n_map = tf.logical_and(y_true == 1, decisions==1)
    n_map = tf.where(n_map,1,0)
    n_map = tf.math.reduce_sum(n_map,-1)
    res['speaker_error'] =tf.math.reduce_sum(tf.minimum(n_ref, n_sys) - n_map,-1)
    res['correct'] = tf.reduce_sum(tf.where(y_true==decisions,1,0),-1) / y_true.shape[1]
    res['diarization_error'] = tf.cast(res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'], tf.float32)
    
    return tf.math.divide_no_nan(res['diarization_error'], res['speaker_scored'])


