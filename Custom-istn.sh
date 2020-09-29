#!/bin/bash
#SBATCH --job-name=A1
#SBATCH --partition=TITAN
#SBATCH -n 1
#SBATCH --gres=gpu 
#SBATCH -t 7-12:00
#SBATCH --mem=32000
#SBATCH --error=A1.err
#SBATCH --output=A1.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=Guang.Yang003@umb.edu
 

LOSS="$1"
NAME="$2_$LOSS"
PATHTODATA="$3"

## basically the same as Loss-istn.sh but calls custom-istn-reg.py which is made to output
    ## .dat files storing the affine matrix transformations used in transforming images
# source activate istn
python3 custom-istn-reg.py \
 --config $PATHTODATA/config.json \
 --transformation affine \
 --loss $LOSS \
 --out output/$NAME \
 --model output/$NAME/train/model \
 --val $PATHTODATA/val.csv \
 --val_seg $PATHTODATA/val.seg.csv \
 --test $PATHTODATA/val.csv \
 --test_seg $PATHTODATA/val.seg.csv \
 --train $PATHTODATA/train.csv \
 --train_seg $PATHTODATA/train.seg.csv

#python3 custom-istn-reg.py --config $PATHTODATA/config.json --transformation affine --loss $LOSS --out output/$NAME --model output/$NAME/train/model --test $PATHTODATA/val.csv --test_seg $PATHTODATA/val.seg.csv

# echo "Name: $NAME Data: $PATHTODATA DThis file is for haehnassignment1," > README_$LOSS
# echo "100 epochs, batchsize 10, loss is $LOSS, maxdegree rotation is 60, folder structure is more clear" >> README_$LOSS
# cp README output/$NAME
# cp -r $PATHTODATA output/$NAME
# rm README
