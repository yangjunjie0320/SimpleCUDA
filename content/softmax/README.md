use the following command to summarize kernels:
```bash
sed -i '/^## Kernels/,$d' README.md
echo -e "## Kernels\n" >> README.md

for f in softmax-v*.cu; do
  v=$(echo $f | grep -oP 'v\d+')
  comment=$(awk '
    /^\/\// { lines = lines (lines ? " " : "") substr($0, 4); next }
    /^__global__|^template/ { print lines; exit }
    { lines = "" }
  ' "$f")
  [ -z "$comment" ] && comment="(no description)"
  echo "- \`kernel_$v\`: $comment" >> README.md
done
```

### Note

The kernels are running with random generated input data of (nrow, NUM_WARP_IN_BLOCK * NUM_THREAD_IN_WARP).
In which `NUM_WARP_IN_BLOCK` is 4 and `NUM_THREAD_IN_WARP` is 32.

- SM是 GPU 的基本执行单元，有固定资源限制：最大一起活跃的 warp 数、block/thread 数、使用的寄存器与共享内存都有限制。
- occupancy = 活跃 warp 数 / SM 支持的最大 warp 数，反映 SM 并行利用程度。
- SM 同时能驻留的 blocks 数由这些资源共同决定：最大 block 限制、寄存器使用、共享内存使用等，取最小的约束值。
- Block 太小时，每个 block 的 warp 数少，驻留 block 数量被最大 block 限制；活跃 warp 总数小，occupancy 低。
- Block 太大时，内部同步或依赖会造成 warp 等待，这种执行延迟不会反映在 occupancy 上，但会降低性能。
- 寄存器使用量、共享内存使用量直接影响可驻留的 thread/warp 数，高资源需求会降低并发度。
- 高占用率有助于隐藏延迟，但真正性能还依赖于内存访问效率、分支发散等因素。
- 每个线程的 workload 应均匀且适中：太小浪费调度资源，太大则可能占用过多资源降低并发。

### Further Optimization Directions
- Online Softmax: 一次遍历同时算 max 和 sum，减少 memory pass
- Persistent Kernel: Block 常驻动态领任务，减少 launch 开销和负载不均
- 混合精度 (FP16/BF16): 半精度读写、单精度计算，带宽减半
- Warp Specialization: 不同 warp 分工（load/compute/store），流水线并行

## Kernels

- `kernel_v1`: each block handles one row, 1 thread, naive serial implementation
- `kernel_v2`: each block handles one row with ncol threads, smem tree reduction, ncol must equal block size
- `kernel_v3`: each block handles one row with ncol threads, CUB BlockReduce
- `kernel_v4`: each block handles one row, use NUM_THREAD_IN_WARP threads
- `kernel_v5`: each warp handles one row NUM_THREAD_IN_WARP threads; each block contains NUM_WARP_IN_BLOCK warps
- `kernel_v6`: Each warp processes num_rows_per_access contiguous rows per iteration, striding by row_step across multiple iterations. Each thread handles num_cols_per_thread non-contiguous columns (stride = NUM_THREAD_IN_WARP). Each block contains NUM_WARP_IN_BLOCK warps.
