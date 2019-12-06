EP_LIST='0.01 0.03 0.1 0.3 1 3'
FUSION_LIST='block blocktucker mutan tucker mlb mfb mfh'

for EP in $EP_LIST; do
    for FUSION in $FUSION_LIST; do
        sbatch slurm_train.sh $EP $FUSION
	sleep 2
    done
done
