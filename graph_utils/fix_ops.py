import tensorflow as tf

graph_path = '/root/tensorpack/examples/FasterRCNN/out.pb'

graph_def = tf.GraphDef()
graph_def.ParseFromString(open(graph_path, 'rb').read())

for node in graph_def.node:
    if node.op == 'NonMaxSuppressionV3':
        node.op = 'NonMaxSuppressionV2'
        for item in node.input:
            if 'score_threshold' in item:
                node.input.remove(item)
    if node.op == 'Const':
        if 'score_threshold' in node.name:
            graph_def.node.remove(node)


with open('/root/tensorpack/examples/FasterRCNN/outV2.pb', 'wb') as f:
    f.write(graph_def.SerializeToString())
    