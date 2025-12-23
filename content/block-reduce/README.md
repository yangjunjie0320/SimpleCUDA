use the following command to summarize kernels:
```bash
kernel_name="block-_educe"
sed -i '/^## Kernels/,$d' README.md
echo -e "## Kernels\n" >> README.md

for f in ${kernel_name}-v*.cu; do
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

We will implement the block reduce kernel. The input is a 2D array of size (nrow, NUM_WARP_IN_BLOCK * NUM_THREAD_IN_WARP).
and the output is a 2D array of size (nrow, ). 
nrow blocks will be launched, and each block will reduce the input array along the second dimension.

## Kernels
