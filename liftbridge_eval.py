
from datetime import datetime
import math
import time
 
from tensorflow.python.platform import gfile
import tensorflow as tf

from tensorflow.models.image.liftbridge import liftbridge

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/liftbridge_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'eval',
                           """Either 'train' or 'eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/liftbridge_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[1]
        else:
            print('No checkpoint file found')
            return
        
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0 # Counts the number of correct predictions
            total_sample_count = num_inter * FLAGS.batch_size
            step = 0
            while step < num_inter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1
                
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e: # pylint: disable=broad-except
            coord.request_stop(e)
            
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    with tf.Graph().as_default():
        eval_data = FLAGS.eval_data === 'eval'
        images, labels = liftbridge.inputs(eval_data=eval_data)
        
        logits = liftbridge.inference(images)
        
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        
        variable_averages = tf.train.ExponentialMovingAverage(
            liftbridge.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        summary_op = tf.merge_all_summaries()
        
        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                                graph_def=graph_def)
        
        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(arvg=None): # pylint: disable=unused-argument
    liftbridge.maybe_download_and_extract()
    if gfile.Exists(FLAGS.eval_dir):
        gfile.DeleteRecursively(FLAGS.eval_dir)
    gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()    


if __name__ == '__main__':
    tf.app.run()