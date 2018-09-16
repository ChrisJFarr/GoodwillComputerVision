"""
This file provides a class for constructing datasets from a tfrecord file that can then be used in a hot-swappable way with an iterator
tf.record reading code influenced heavily by:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py

https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map
https://www.tensorflow.org/guide/datasets

Additional sources consulted:
https://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
http://machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html


"""

import tensorflow as tf
class GWData(object):

    """

    # usage: make a dataset by passing in a tfrecord. Format will need to line up with output produced in tf_record_writer.py
    # For a more easily workable example:  https://gist.github.com/trcook/9fc8698cf7dc848a953f8e7a7e5f1aad

    :Example:

    ::

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

    def __init__(self,the_file:str):
        self.filename=the_file
        self.dataset=self.mk_data(self.filename)
        self.output_types=self.dataset.output_types
        self.output_shapes=self.dataset.output_shapes
        self.it_init=None

    def parse_label(self,lab:tf.string):
        """ NOT IMPLEMENTED YET"""
        # probably put tensor through something like word-to-vec, but that may make more sense *before* we pack into tfrecord. 
        return lab

    def parse_function(self, ex: tf.train.Example) -> dict:

        """
        This is the function that is mapped over the examples in the tfrecord file to parse an example back into a usable set of tensors.

        """

        # define a feature map that identifies the type of feature and dtype.
        # The keys in the dict should correspond to the names of features (i.e. keys) in the example.

        feature_map = {
        "image": tf.FixedLenFeature((),tf.string),
        "label":tf.FixedLenFeature((),tf.string),
        "X": tf.FixedLenFeature((),tf.float32),
        "Y":tf.FixedLenFeature((),tf.float32)}

        # parse the example -- tensorflow applies the dict to the example.
        # The resulting dict-like will have tensors corresponding to the values unpacked from the example.
        pex = tf.parse_single_example(ex,feature_map)

        # re-cast our dimension parameters to integers
        X=tf.cast(pex['X'],tf.int32)
        Y=tf.cast(pex['Y'],tf.int32)
        D=tf.constant(3)
        image_dims=tf.stack([X,Y,D])

        # decode image data from byte-encoded back into integers.
        # It has a type of tf.int8 because the data is byte-encoded -- 1 byte is 8-bits, hence a 8-bit integer.
        img=tf.decode_raw(pex['image'],tf.int8)

        # cast the image data back to float32 -- we do this because its easier
        # to deal with data in a model if the inputs are floats (less type errors).
        img=tf.cast(img,tf.float32)

        # reshape the image data back to its original shape
        img=tf.reshape(img,image_dims)
        # Apply a parsing routine to the label data. If not implemented,
        label=self.parse_label(pex['label'])
        return {'X':img,'Y':label}

    def mk_data(self,the_file:"str or list"):
        # add some logic so we can add one file or a list of files
        if not isinstance(the_file,list):
            the_file=[the_file]

        # point the dataset at the tfrecord we created
        dataset=tf.data.TFRecordDataset(the_file)
        # Parse the record into tensors.
        dataset = dataset.map(self.parse_function)
        # Shuffle the dataset
        dataset = dataset.shuffle(buffer_size=1)
        # Repeat the input indefinitly
        dataset = dataset.repeat()  
        # Generate batches
        dataset = dataset.batch(3)
        return dataset

    def mk_init(self,it:tf.data.Iterator):
        '''
        This function is what creates the button that lets us switch the source feeding an iterator.
        Once run, the class instance will populate a property called in_init.

        :Example:

        ::

            import tensorflow as tf

            dat=GWData('file.tfrecord')
            it=tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
            dat.mk_init(it)
            sess=tf.Session()
            # point iterator 'it' at dataset in 'dat':
            sess.run(dat.it_init)
            # tell iterator 'it' to get a (batch) example from the dataset in 'dat'
            sess.run(it.get_next())


        '''
        dataset=self.dataset
        #generate initializer for the iterator that is specific to *dataset*
        self.it_init=it.make_initializer(dataset)
        




import numpy as np
np.concatenate