#!/bin/bash
# Dataset init code from https://github.com/geopavlakos/object3d

mkdir -p tmp

# Download some necessary data
if [ ! -f tmp/extra_data.tar ]; then
    wget -P tmp http://visiondata.cis.upenn.edu/object3d/data/extra_data.tar
fi
tar -xf tmp/extra_data.tar

# Download PASCAL 3D
if [ ! -f tmp/PASCAL3D+_release1.1.zip ]; then
    wget -P tmp ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip
fi
unzip tmp/PASCAL3D+_release1.1.zip
mv PASCAL3D+_release1.1 data/PASCAL3D

# move all imagenet images in PASCAL3D+ in one folder and rename *.JPEG to *.jpg
mkdir -p data/pascal3d/images
for x in $(ls data/PASCAL3D/Images | grep imagenet); do mv data/PASCAL3D/Images/$x/*.JPEG data/pascal3d/images/; done
for file in data/pascal3d/images/*.JPEG; do mv "$file" data/pascal3d/images/"`basename "$file" .JPEG`.jpg"; done
rm -r data/PASCAL3D

# Download PASCAL VOC
if [ ! -f tmp/VOCtrainval_11-May-2012.tar ]; then
    wget -P tmp http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
fi
tar -xf tmp/VOCtrainval_11-May-2012.tar
mv VOCdevkit data/

# move all pascal images in one folder
for file in data/VOCdevkit/VOC2012/JPEGImages/*.jpg; do mv "$file" data/pascal3d/images/; done
rm -r data/VOCdevkit

# create symbolic links for training and testing code
mkdir -p pose-hg/pose-hg-train/data
mkdir -p pose-hg/pose-hg-demo/data
ln -s $PWD/data/pascal3d $PWD/pose-hg/pose-hg-train/data
ln -s $PWD/data/pascal3d $PWD/pose-hg/pose-hg-demo/data

# get PASCAL3D pretrained model

if [ ! -f tmp/pose-hg-pascal3d.t7 ]; then
    wget -P tmp http://visiondata.cis.upenn.edu/object3d/models/pose-hg-pascal3d.t7
fi
cp tmp/pose-hg-pascal3d.t7 pose-hg/pose-hg-demo/
