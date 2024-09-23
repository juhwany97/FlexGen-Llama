import os
import pandas as pd
import argparse
import re

def print_average(size, tbt, ttft):
    # Define the batch sizes and sequence lengths
    batch_sizes = [1,2,4,8,16,32,64,128]
    sequence_lengths = [256, 512, 1024]

    # Initialize a dictionary to store the results
    results = {}

    # Iterate over all batch sizes and sequence lengths
    for b in batch_sizes:
        for seq_len in sequence_lengths:
            # Construct the filename
            #file_name = f"./{name}_{size}b_xft_b{b}_i{seq_len}"
            file_name = f"fo-{size}b-gbs{b}-ngbs1-prompt{seq_len}-gen32-percent-55-45-0-100-100-0-gpu-cache.log"
            #file_name = f"./{name}_{size}b_xft_b{b}_i{seq_len}_rev1"
            #file_name = f"./{name}_{size}b_xft_b{b}_i{seq_len}_INT8_rev1"
            
            prefill_latency = None
            decode_latency = None
            if os.path.exists(file_name):
                with open(file_name, 'r') as log_file:
                    log_content = log_file.read()
                    prefill_latency_match = re.search(r'prefill latency:\s+([\d.]+)\s+s', log_content)
                    decode_latency_match = re.search(r'decode latency:\s+([\d.]+)\s+s', log_content)
    
                    if prefill_latency_match and decode_latency_match:
                        prefill_latency = float(prefill_latency_match.group(1)) * 1000
                        decode_latency = float(decode_latency_match.group(1)) / 32 * 1000
                    
                    results[(b, seq_len)] = {'TTFT_avg': prefill_latency, 'TBT_avg': decode_latency}
            else:
                print(f"File not found: {file_name}")

    # print(f"{'Batch Size':<10} {'Seq Length':<10} {'TTFT Avg (ms)':<15} {'TBT Avg (ms)':<15}")
    if tbt == True:
        print(f"{'TBT Avg (s)':<15}")
        for(b, seq_len), averages in results.items():
            #print(f"{b:<10} {seq_len:<10} {averages['TTFT_avg']:<15.2f} {averages['TBT_avg']:<15.2f}")
            print(f"{averages['TBT_avg']:<15.2f}")

    if ttft == True:
        print(f"{'TTFT Avg (ms)':<15}")
        for(b, seq_len), averages in results.items():
            #print(f"{b:<10} {seq_len:<10} {averages['TTFT_avg']:<15.2f} {averages['TBT_avg']:<15.2f}")
            print(f"{averages['TTFT_avg']:<15.2f}")
def main():
    parser = argparse.ArgumentParser(description='Filter CSV file for kernel names and durations.')
    parser.add_argument('-b', '--msize', type=int, help='model size')
    parser.add_argument('-tbt', action='store_true', help ="print tbt")
    parser.add_argument('-ttft', action='store_true', help ="print ttft")

    args = parser.parse_args()
    print_average(args.msize, args.tbt, args.ttft)

if __name__ == "__main__":
    main()