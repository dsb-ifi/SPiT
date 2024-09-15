extern "C" __global__
void dpcc_roots_kernel(long long n, long long* P) {
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int thread_count = gridDim.x * blockDim.x;
    bool flag = true;
    while (flag) {
        flag = false;
        for (long long v = thread_id; v < n; v += thread_count) {
            long long root = P[v];
            while (root != P[root]) {
                root = P[root];
            }
            P[v] = root;
            if (P[v] != P[P[v]]) {
                flag = true;
            }
        }
        __syncthreads();
    }
}