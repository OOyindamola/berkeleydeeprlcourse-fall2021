import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
path_to_events_file = '/home/oyindamola/Research/homework_fall2021/hw1/data/q2_dagger_ant_Ant-v2_04-02-2022_15-22-07/events.out.tfevents.1644006127.oyinda-lasting'
for e in tf.train.summary_iterator(path_to_events_file):
    for v in e.summary.value:
        print(v.tag)
        # if v.tag == 'loss' or v.tag == 'accuracy':
        #     print(v.simple_value)
