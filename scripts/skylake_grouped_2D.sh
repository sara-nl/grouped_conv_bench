#!/bin/bash

for chan in 32 64 128
do
    for batch in 16 32
    do
        for groups_log in $(seq 0 $(echo "l($chan)/l(2)" | bc -l))
	do
	   groups_num=$(echo $((2**groups_log)))
	   echo "batch: ${batch}, chan: ${chan}, groups: ${groups_num}"
           KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=16 python pytorch_benchmark.py --batch-size ${batch} --ndims 2 --in-channels ${chan} --out-channels ${chan} --kernel-size 3 --input-dim 512 512 --num-batches-per-iter 2 --num-iter 3 --ngroups=${groups_num} --output='runs/skylake_2D.txt'
	done
    done
done
