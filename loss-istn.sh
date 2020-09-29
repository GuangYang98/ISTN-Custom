#!/bin/bash
#SBATCH --job-name=A1
#SBATCH --partition=TITAN
#SBATCH -n 1
#SBATCH --gres=gpu 
#SBATCH -t 7-12:00
#SBATCH --mem=32000
#SBATCH --error=loss_istn.err
#SBATCH --output=loss_istn.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yahiya.hussain001@umb.edu

## $1, $2, and $3 are commandline arguments to the bash script
## example: ./Loss_istn.sh u testRun_u data/synth2d
## or on gibbs cluster: sbatch Loss_istn.sh u testRun_u data/synth2d 
 
## can be s, u, e, or i, for supervised, unsupervised, explicit, or implicit from the ISTN paper respectively
LOSS="$1"
## name of the outputted file location: for example: if $2 == testRun then files will be at 
    ## istn-master/output/testRun 
NAME="$2"
## path to the data from the current istn-master working directory: for example data/synth2d
PATHTODATA="$3"


source activate istn
python istn-reg.py \
 --config $PATHTODATA/config.json \
 --transformation affine \
 --loss $LOSS \
 --out output/$NAME \
 --model output/$NAME/train/model \
 --train $PATHTODATA/train.csv \
 --train_seg $PATHTODATA/train.seg.csv \
 --val $PATHTODATA/val.csv \
 --val_seg $PATHTODATA/val.seg.csv \
 --test $PATHTODATA/val.csv \
 --test_seg $PATHTODATA/val.seg.csv
