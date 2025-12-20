use the following command to summarize kernels:
```bash
sed -i '/^## Kernels/,$d' README.md
echo -e "## Kernels\n" >> README.md
for f in softmax-v*.cu; do
  v=$(echo $f | grep -oP 'v\d+')
  comment=$(grep -m1 '^// ' $f | sed 's/^\/\/ *//') 
  [ -z "$comment" ] && comment="(no description)"
  echo "- \`kernel_$v\`: $comment" >> README.md
done
```

### Benchmark

The kernels are running with random generated input data of (nrow, NUM_WARP_IN_BLOCK * NUM_THREAD_IN_WARP).
In which `NUM_WARP_IN_BLOCK` is 4 and `NUM_THREAD_IN_WARP` is 32.

## Kernels

- `kernel_v1`: each block handles one row, 1 thread, naive serial implementation
- `kernel_v2`: each block handles one row with ncol threads, smem tree reduction, ncol must equal block size
- `kernel_v3`: each block handles one row with ncol threads, CUB BlockReduce
- `kernel_v4`: each block handles one row, use NUM_THREAD_IN_WARP threads
- `kernel_v5`: (no description)
- `kernel_v6`: (no description)
