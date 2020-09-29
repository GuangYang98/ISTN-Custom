#!/bin/bash

DIR="/Users/guangyang/Desktop/istn-custom-master/data/big_mess_300300"

echo "src,trg" > train.csv
echo "src,trg" > train.seg.csv

echo "src,trg" > val.csv
echo "src,trg" > val.seg.csv

for N in {1..109}
    do
a=`expr $N - 1`
echo "$DIR/Train_Images_Unaligned/$N.png,$DIR/Train_Images_Unaligned/${a}.png" >> train.csv
echo "$DIR/Train_Masks_Unaligned/$N.png,$DIR/Train_Masks_Unaligned/${a}.png" >> train.seg.csv
    done

for N in {1..54}
    do
a=`expr $N - 1`
echo "$DIR/Test_Images_Unaligned/$N.png,$DIR/Test_Images_Unaligned/${a}.png" >> val.csv
echo "$DIR/Test_Masks_Unaligned/$N.png,$DIR/Test_Masks_Unaligned/${a}.png" >> val.seg.csv
    done
