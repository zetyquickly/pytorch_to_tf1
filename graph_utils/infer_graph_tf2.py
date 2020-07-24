import tensorflow.compat.v1 as tf
import numpy as np
import argparse

import pickle
import cv2 

from tensorflow.python.framework import tensor_util

# If load from pb, you may have to use get_tensor_by_name heavily.

IMAGE_SHAPE = [1, 3, 192, 192]

class Model(object):
    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath=self.model_filepath)

    def load_graph(self, model_filepath):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        self.graph = tf.Graph()

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        print('Check out the input placeholders:')
        nodes = [
            n.name + ' => ' + n.op for n in graph_def.node
            if n.op in ('Placeholder')
        ]
        for node in nodes:
            print(node)

        with self.graph.as_default():
            # Define input tensor
            self.input = tf.placeholder(np.float32,
                                        shape=IMAGE_SHAPE,
                                        name='image')
            tf.import_graph_def(graph_def, {
                'image:0': self.input,
            })

        self.graph.finalize()

        print('Model loading complete!')

        # Get layer names
        layers = [op.name for op in self.graph.get_operations()]
        for layer in layers:
            print(layer)
        """
        # Check out the weights of the nodes
        weight_nodes = [n for n in graph_def.node if n.op == 'Const']
        for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            # print("Value - " )
            # print(tensor_util.MakeNdarray(n.attr['value'].tensor))
        """

        # In this version, tf.InteractiveSession and tf.Session could be used interchangeably.
        # self.sess = tf.InteractiveSession(graph = self.graph)
        self.sess = tf.Session(graph=self.graph)

    def test(self, data):

        # Know your output node name
        for op in self.graph.get_operations():
            print(op.name)
        output_tensor = [
            self.graph.get_tensor_by_name("import/output/boxes:0"),
            self.graph.get_tensor_by_name("import/output/densepose_S:0"),
            self.graph.get_tensor_by_name("import/output/densepose_I:0"),
            self.graph.get_tensor_by_name("import/output/densepose_U:0"),
            self.graph.get_tensor_by_name("import/output/densepose_V:0")
        ]
        output = self.sess.run(output_tensor,
                               feed_dict={
                                   self.input: data,
                               })

        return output


def test_from_frozen_graph(model_filepath, image):

    tf.reset_default_graph()

    model = Model(model_filepath=model_filepath)
    image = np.random.rand(*IMAGE_SHAPE)

    test_prediction = model.test(data=image)
    print(test_prediction)
    with open('./1.jpg.pkl', 'wb') as f:
        pickle.dump(test_prediction, f)


def main():

    model_pb_filepath_default = './outV2.pb'
    input_file = './1.jpg'
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)

    # Argparser
    parser = argparse.ArgumentParser(
        description='Load and test model from frozen graph pb file.')

    parser.add_argument('--model_pb_filepath',
                        type=str,
                        help='model pb-format frozen graph file filepath',
                        default=model_pb_filepath_default)

    argv = parser.parse_args()

    model_pb_filepath = argv.model_pb_filepath

    test_from_frozen_graph(model_filepath=model_pb_filepath, image=img)


if __name__ == '__main__':
    main()