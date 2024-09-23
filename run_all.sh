
for BATCH_SIZE in 1 2 4 8 16 32 64 128
do
        for LIN in 256 512 1024
        do
		python3 -m flexgen.flex_opt --model meta-llama/Meta-Llama-3-70B --path __DUMMY__ --gpu-batch-size ${BATCH_SIZE} --prompt-len ${LIN} --percent 50 50 0 100 100 0 
		python3 -m flexgen.flex_opt --model meta-llama/Meta-Llama-3-70B --path __DUMMY__ --gpu-batch-size ${BATCH_SIZE} --prompt-len ${LIN} --percent 50 50 0 100 100 0 --cpu-cache-compute
	done
done
