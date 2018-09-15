import tensorflow as tf
class GWData(object):
    """
    # usage: make a dataset by passing in a tfrecord. Format will need to line up with output produced in tf_record_writer.py
    # For a more easily workable example:  https://gist.github.com/trcook/9fc8698cf7dc848a953f8e7a7e5f1aad
    # Example

    dataset=GWData('./output.tfrecord')
    val_dataset=GWData('./validate.tfrecord')

    #make initializer that can be hot-swapped in keras
    it=tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)

    # make initializers for the iterator for dataset and val_dataset
    dataset.mk_init(it)
    val_dataset.mk_init(it)

    # get handle for next batch from the iterator
    next_item=it.get_next()
    # Produce X and Y handles that can be used as targets for keras
    X=next_item['X']
    Y=next_item['Y']
    # Into Keras:

    from keras import backend as K
    import keras as k
    from keras.layers import *

    inputs=Input(tensor=X)
    labs=Input(tensor=Y)

    net=Dense(.....)(inputs) 
    # ... rest of model goes here
    output=Dense(....)(net)
    model=k.Model(inputs,output)
    # setup loss, etc, anything else you do before normally calling to fit: 
    model.add_loss(loss)
    model.compile('adam')

    # initialize the iterator to the training data:
    sess.run(dataset.init_it)
    model.fit(.....)

    # now, hotswap dataset out for val_dataset:
    sess.run(val_dataset.init_it)
    model.evaluate(...)

    # This makes it easy enough to grab the outputs too: 
    sess.run(output)
    """

    def __init__(self,the_file):
        self.filename=the_file
        self.dataset=self.mk_data(self.filename)
        self.output_types=self.dataset.output_types
        self.output_shapes=self.dataset.output_shapes
        self.it_init=None

    def parse_label(lab):
        """ NOT IMPLEMENTED YET"""
        # probably put tensor through something like word-to-vec, but that may make more sense *before* we pack into tfrecord. 
        return lab

    def parse_function(ex):
       
        feature_map = {
        "image": tf.FixedLenFeature((),tf.string),
        "label":tf.FixedLenFeature((),tf.string),
        "X": tf.FixedLenFeature((),tf.float32),
        "Y":tf.FixedLenFeature((),tf.float32)}

        pex = tf.parse_single_example(ex,feature_map)
        X=tf.cast(pex['X'],tf.int32)
        Y=tf.cast(pex['Y'],tf.int32)
        D=tf.constant(3)
        img=tf.decode_raw(pex['image'],tf.int8)
        img=tf.cast(img,tf.float32)
        img=tf.reshape(img,tf.stack([X,Y,D]))
        label=parse_label(pex['label'])
        return {'X':img,'Y':label}

    def mk_data(the_file):
        if not isinstance(the_file,list):
            the_file=[the_file]
        # point the dataset at the tfrecord we created
        dataset=tf.data.TFRecordDataset(the_file)
        # Parse the record into tensors.
        dataset = dataset.map(parse_function)
        # Shuffle the dataset
        dataset = dataset.shuffle(buffer_size=1)
        # Repeat the input indefinitly
        dataset = dataset.repeat()  
        # Generate batches
        dataset = dataset.batch(3)
        return dataset

    def mk_init(self,it):
        dataset=self.dataset
        #generate initializer for the iterator that is specific to *dataset*
        self.it_init=it.make_initializer(dataset)
        
        return X,Y



