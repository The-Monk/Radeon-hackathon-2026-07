// repro_harness.cu — grade an agent-written low-bit kernel against the golden.
// Harness owns inputs, reference and comparison. Agent supplies agent_kernel.cuh.
#include "quant_common.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "agent_kernel.cuh"

static std::vector<char> slurp(const char* p, size_t n) {
    FILE* f = fopen(p,"rb"); if(!f){fprintf(stderr,"missing %s\n",p);exit(2);}
    std::vector<char> v(n); if (fread(v.data(),1,n,f)!=n){fprintf(stderr,"short %s\n",p);exit(2);}
    fclose(f); return v;
}
int main() {
    const int N = 2048, Kb = 32;
    char pw[64],pa[64],po[64];
    snprintf(pw,sizeof pw,"golden_%s_w.bin",FMT_NAME);
    snprintf(pa,sizeof pa,"golden_%s_a.bin",FMT_NAME);
    snprintf(po,sizeof po,"golden_%s_o.bin",FMT_NAME);
    auto hW = slurp(pw,(size_t)N*Kb*sizeof(block_lowbit));
    auto hA = slurp(pa,(size_t)Kb*QK_LOWBIT*4);
    auto hG = slurp(po,(size_t)N*4);
    const float* golden = (const float*)hG.data();

    block_lowbit* dW; float *dA,*dO;
    hipMalloc(&dW,hW.size()); hipMalloc(&dA,hA.size()); hipMalloc(&dO,(size_t)N*4);
    hipMemcpy(dW,hW.data(),hW.size(),hipMemcpyHostToDevice);
    hipMemcpy(dA,hA.data(),hA.size(),hipMemcpyHostToDevice);
    agent_kernel<<<N,64,64*sizeof(float)>>>(dW,dA,dO,(int64_t)Kb);
    hipError_t e=hipDeviceSynchronize();
    if(e!=hipSuccess){printf("LAUNCH FAILED: %s\n",hipGetErrorString(e));return 1;}
    std::vector<float> got(N); hipMemcpy(got.data(),dO,(size_t)N*4,hipMemcpyDeviceToHost);
    double worst=0; int bad=0;
    for(int i=0;i<N;++i){
        if(!std::isfinite(got[i])){bad++;worst=INFINITY;continue;}
        double d=fabs((double)got[i]-golden[i]), s=fabs((double)golden[i]);
        double rel = s>1e-6 ? d/s : d; if(rel>worst) worst=rel;
    }
    printf("fmt=%s rows=%d non_finite=%d max_rel_err=%.6e\n",FMT_NAME,N,bad,worst);
    printf("golden[0]=%.6f agent[0]=%.6f\n", golden[0], got[0]);
    return 0;
}
