#!/bin/bash

DIR="/Users/guangyang/Desktop/istn-custom-master/data/Gianna"

echo "src,trg" > train.csv
echo "src,trg" > train.seg.csv

echo "src,trg" > val.csv
echo "src,trg" > val.seg.csv

for N in {0..99}
  do
    echo "$DIR/Train_Images_Unaligned/$N.png,$DIR/Train_Images/$N.png" >> train.csv
    echo "$DIR/Train_Masks_Unaligned/$N.png,$DIR/Train_Masks/$N.png" >> train.seg.csv
  done

for N in {0..54}
  do
    echo "$DIR/Test_Images_Unaligned/$N.png,$DIR/Test_Images/$N.png" >> val.csv
    echo "$DIR/Test_Masks_Unaligned/$N.png,$DIR/Test_Masks/$N.png" >> val.seg.csv
  done
