__device__ __forceinline__ unsigned long long to_packed_ull(float sim, unsigned int index) {
    unsigned short score = static_cast<unsigned short>(round(sim * 65535));
    unsigned long long packed_value;
    packed_value = (static_cast<unsigned long long>(score) << 48) | static_cast<unsigned long long>(index);
    return packed_value;
}

extern "C" __global__
void argmax_cosine_kernel(
    const unsigned long long m, 
    const unsigned long long d, 
    const unsigned long long n,
    const float *vertices, 
    const unsigned long long *u, 
    const unsigned long long *v,
    unsigned long long *packed, 
    const float* muptr, 
    const float* stdptr,
    const float *size
) {    
    const unsigned long long thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long thread_cnt = gridDim.x * blockDim.x;
    const float mu = muptr[0];
    const float std = stdptr[0];
    
    for (long long tid = thread_idx; tid < m; tid += thread_cnt) {
        unsigned long long i = u[tid];
        unsigned long long j = v[tid];
        float sim = 0;
        float norm_i = 0;
        float norm_j = 0;

        if (i == j) {
            sim = (size[i] - mu) / std;
            sim = sim < -.75 ? -.75 : sim > .75 ? .75 : sim;
        }
        else {
            for (unsigned long long k = 0; k < d; k++) {
                unsigned long long idx_i = i * d + k;
                unsigned long long idx_j = j * d + k;

                sim += vertices[idx_i] * vertices[idx_j];
                norm_i += vertices[idx_i] * vertices[idx_i];
                norm_j += vertices[idx_j] * vertices[idx_j];                
            }
            sim = sim / (sqrtf(norm_i) * sqrtf(norm_j));
        }
        sim = (sim + 1.0f) / 2;        
        atomicMax((unsigned long long *)&packed[i], to_packed_ull(sim, j));        
        atomicMax((unsigned long long *)&packed[j], to_packed_ull(sim, i));
    }
}

extern "C" __global__
void argmax_cosine_kernel_bbox(
    const unsigned long long m, 
    const unsigned long long d, 
    const unsigned long long n,
    const float *vertices, 
    const unsigned long long *u, 
    const unsigned long long *v,
    unsigned long long *packed, 
    const float* muptr, 
    const float* stdptr,
    const float* cmixptr,
    const float *size,
    const float *ymin,
    const float *xmin,
    const float *ymax,
    const float *xmax
) {    
    const unsigned long long thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long thread_cnt = gridDim.x * blockDim.x;
    const float mu = muptr[0];
    const float std = stdptr[0];
    const float cmix = cmixptr[0];
    
    for (long long tid = thread_idx; tid < m; tid += thread_cnt) {
        unsigned long long i = u[tid];
        unsigned long long j = v[tid];
        float sim = 0;
        float cpw = 0;
        float per = 0;
        float norm_i = 0;
        float norm_j = 0;

        if (i == j) {
            sim = (size[i] - mu) / std;
            sim = sim < -0.75f ? -0.75f : sim > 0.75f ? 0.75f : sim;
            per = ymax[i] - ymin[i] + xmax[i] - xmin[i] + 2.0f;
            cpw = 4.0f * size[i] / (per * per);
        }
        else {
            per = (
                (max(ymax[i], ymax[j]) - min(ymin[i], ymin[j])) + 
                (max(xmax[i], xmax[j]) - min(xmin[i], xmin[j])) + 2.0f
            );
            cpw = 4.0f * (size[i] + size[j]) / (per * per);

            for (unsigned long long k = 0; k < d; k++) {
                unsigned long long idx_i = i * d + k;
                unsigned long long idx_j = j * d + k;
                sim += vertices[idx_i] * vertices[idx_j];
                norm_i += vertices[idx_i] * vertices[idx_i];
                norm_j += vertices[idx_j] * vertices[idx_j];                
            }
            sim = sim / (sqrtf(norm_i) * sqrtf(norm_j));
        }
        cpw = cpw < 0 ? 0 : cpw > 1 ? 1 : cpw;
        sim = cmix * cpw + (1.0f - cmix) * (sim + 1.0f) / 2.0f;
        atomicMax((unsigned long long *)&packed[i], to_packed_ull(sim, j));        
        atomicMax((unsigned long long *)&packed[j], to_packed_ull(sim, i));
    }
}
