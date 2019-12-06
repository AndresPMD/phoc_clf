EP_LIST='0.01 0.03 0.1 0.3 1 3'

for EP in $EP_LIST; do
    sbatch -A fashion -p fashion --cpus-per-task 8 strain_rr.sh $EP
done