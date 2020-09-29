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
#SBATCH --mail-user=yahiya.hussain001@umb.edu
 

LOSS="$1"
NAME="$2_$LOSS"
PATHTODATA="$3"


source activate istn
python istn-reg.py --config $PATHTODATA/config.json --transformation affine --loss $LOSS --out output/$NAME --model output/$NAME/train/model --test $PATHTODATA/val.csv --test_seg $PATHTODATA/val.seg.csv

echo "Name: $NAME Data: $PATHTODATA DThis file is for haehnassignment1," > README_$LOSS
echo "100 epochs, batchsize 10, loss is $LOSS, maxdegree rotation is 60, folder structure is more clear" >> README_$LOSS
# cp README output/$NAME
# cp -r $PATHTODATA output/$NAME
# rm README
