import time
import cudf
import cugraph
import os
import sys

# get the input TSV (undirected format) from:
# https://graphchallenge.mit.edu/data-sets#SNAPDatasets
write_comms = False
if os.path.isfile(str(sys.argv[1])) is False:
    print('File not found: ', sys.argv[1])
    sys.exit()
else:
    print('Input file found, require file in edge-list tab-separated format.')
    if len(sys.argv) > 2:
        write_comms = True
        print('To write output communities on another file.')

# read data into a cuDF DataFrame using read_csv
t0 = time.time()

#cu_M = cudf.read_csv(str(sys.argv[1]), delimiter='\t', names=["src", "dst", "wgt"], dtype=["int32", "int32", "float32"])
cu_M = cudf.read_csv(str(sys.argv[1]), delimiter='\t', names=["src", "dst", "wgt"], dtype=["int64", "int64", "float64"])
G = cugraph.Graph()
G.from_cudf_edgelist(cu_M, source='src', destination='dst', renumber=False)

t3 = time.time()

print('Time taken (s) to read the native graph into a CUDA dataframe: ', (t3-t0))

t1 = time.time()

count_per_vertex = cugraph.triangle_count(G)
count = count_per_vertex["counts"].sum()

t2 = time.time()

nv = G.number_of_vertices()
print('Graph #vertices/#edges: ', nv, ',', G.number_of_edges())
print('Time taken (s) for counting triangles: ', (t2-t1))
print('#Triangles: ', count/3)
