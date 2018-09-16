#! /usr/bin/env python
# This file writes records into tensorflow

# Based heavily on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py

import tensorflow as tf
import cv2
import glob
import os
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("-o",'--output',help='output name', default='output.tfrecord',dest='output')
parser.add_argument("-g","--glob",help='glob pattern to use for finding images',default="./**/*.[Jj][Pp][Gg]",dest='globber')


    


def parse_path(x):
    raise NotImplementedError


def get_image(img,sized=(1800,1200)):
    x=cv2.imread(img)
    xs=x.shape
    i=sized[0]-xs[0]
    j=sized[1]-xs[1]
    if i>0:
        x=np.pad(x,((0,0),(i//2,i-i//2),(0,0)),'constant')
    if i<0:
        end=xs[0]-(i//2)
        start=i//2
        x=x[start:end,:,:]
    if j>0:
        x=np.pad(x,((j//2,j-j//2),(0,0),(0,0)),'constant')
    if j<0:
        end=xs[0]-(j//2)
        start=j//2
        x=x[start:end,:,:]      
    x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)

    return x


int_feature=lambda x: tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))
byte_feature=lambda x: tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))
float_feature=lambda x: tf.train.Feature(float_list=tf.train.FloatList(value=[x]))


def make_example_from_image(img,path):
    x,y,z=img.shape
    payload={
        "image":byte_feature(img.tobytes()),
        "label":byte_feature(bytes(path,'utf-8')),
        "X":float_feature(x),
        "Y":float_feature(y)
        }
    ex=tf.train.Example(features=tf.train.Features(feature=payload))
    return ex



def record_from_image_paths(files,output_file):
    w = tf.python_io.TFRecordWriter(output_file)
    for path in tqdm(files):
        img=get_image(path)
        ex=make_example_from_image(img,path)
        w.write(ex.SerializeToString())
    w.close()

    

if __name__=='__main__':
    args=parser.parse_args()
    output=args.output
    globber=args.globber
    print(globber)
    files=glob.glob(globber)
    print(files)
    record_from_image_paths(files,output)
