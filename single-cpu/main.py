import tensorflow as tf
from absl import app
from absl import flags
import numpy as np

flags.DEFINE_integer('num', default=50, help='no')

class Net(object):
    def __init__(self, init_state: tf.Tensor):
        self.state = init_state
    def update(self, inp: tf.Tensor):
        self.state = tf.matmul(self.state, inp)
    def get_state(self):
        return self.state

def job(network, inp):
    network.update(inp)

def main(argv):
    del argv
    visible_devices = tf.config.get_visible_devices()
    print(visible_devices)
    print('Single CPU Job Started...')
    x = np.arange(100).reshape([10, 10])
    x = tf.constant(x, dtype=tf.float32)
    net = Net(x)
    current_state = net.state
    print('starting job...')
    for i in range(100):
        net.update(current_state)
        print(str(i) + ' iteration: ')
    print('Job Done!')

if __name__ == '__main__':
    app.run(main)

