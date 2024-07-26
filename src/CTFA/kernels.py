import cupy as cp

BLOCKSIZE = 1024
BLOCKDIM = lambda x : (x - 1) // BLOCKSIZE + 1
F4 = lambda x : cp.float32(x)

kerSample = cp.RawKernel(r'''
extern "C" __global__ void sample(
    const float* spectrum,  // (n, n)
    float* polargram,       // (nBeta, nTheta + 1)
    int n,
    int nBeta,
    int nTheta,
    float theta0,
    float theta1,
    float a0,
    float a1,
    float ast,
    float b,
    float phi
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nBeta * (nTheta + 1)) {
        int i = tid / (nTheta + 1);
        int j = tid % (nTheta + 1);
        float pi = 3.1415926535897932384626;
        float beta = pi / nBeta * i;
        float theta = theta0 + (theta1 - theta0) / nTheta * j - phi;
        float s, c;
        sincosf(beta - ast, &s, &c);
        float a = a0 * c * c + a1 * s * s;
        float delta = a * a + 4 * b * theta;
        float f = delta >= 0 ? sqrtf(2 * theta / (a + sqrtf(delta))) : 1;
        sincosf(beta, &s, &c);
        float dx = f * c * n + n / 2;
        float dy = f * s * n + n / 2;
        int x = floorf(dx);
        int y = floorf(dy);
        dx -= x;
        dy -= y;
        if (0 <= x     && x     < n && 0 <= y     && y     < n) atomicAdd(polargram + tid, spectrum[(y    ) * n + (x    )] * (1 - dx) * (1 - dy));
        if (0 <= x     && x     < n && 0 <= y + 1 && y + 1 < n) atomicAdd(polargram + tid, spectrum[(y + 1) * n + (x    )] * (1 - dx) * (    dy));
        if (0 <= x + 1 && x + 1 < n && 0 <= y     && y     < n) atomicAdd(polargram + tid, spectrum[(y    ) * n + (x + 1)] * (    dx) * (1 - dy));
        if (0 <= x + 1 && x + 1 < n && 0 <= y + 1 && y + 1 < n) atomicAdd(polargram + tid, spectrum[(y + 1) * n + (x + 1)] * (    dx) * (    dy));
    }
}''', 'sample')

def sample(spectrum, nBeta, nTheta, theta0, theta1, a0, a1, ast, b, phi):
    n = spectrum.shape[0]
    polargram = cp.zeros((nBeta, nTheta + 1), dtype = cp.float32)
    kerSample(
        (BLOCKDIM(nBeta * (nTheta + 1)), ),
        (BLOCKSIZE, ),
        (spectrum, polargram, n, nBeta, nTheta, F4(theta0), F4(theta1), F4(a0), F4(a1), F4(ast), F4(b), F4(phi))
    )
    return polargram

kerEquiphaseAverage = cp.RawKernel(r'''
extern "C" __global__ void equiphaseAverage(
    const float* spectrum,  // (ns, n, n)
    const float* da,        // (ns, )
    float* g,               // (nBeta, nTheta + 1)
    float* wt,              // (nBeta, nTheta + 1)
    int ns,
    int n,
    int nBeta,
    int nTheta,
    float theta0,
    float theta1,
    float minf,
    float a0,
    float a1,
    float ast,
    float b,
    float phi
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < ns * n * n) {
        int l = tid / n / n;
        int i = tid / n % n;
        int j = tid % n;
        float y = (float)i / n - 0.5;
        float x = (float)j / n - 0.5;
        float f = hypotf(y, x);
        float beta = atan2f(y, x);
        if (f >= minf) {
            float s, c;
            sincosf(beta - ast, &s, &c);
            float a = a0 * c * c + a1 * s * s;
            float theta = (a + da[l]) * powf(f, 2) + b * powf(f, 4) + phi;
            float pi = 3.1415926535897932384626;
            float dm = (beta < 0 ? beta + pi : beta) / pi * nBeta;
            float dk = (theta - theta0) / (theta1 - theta0) * nTheta;
            int m = floorf(dm);
            int k = floorf(dk);
            dm -= m;
            dk -= k;
            if (0 <= m     && m     < nBeta && 0 <= k     && k     <= nTheta) { int pid = (m    ) * (nTheta + 1) + (k    ); float wid = (1 - dm) * (1 - dk); atomicAdd(g + pid, spectrum[tid] * wid); atomicAdd(wt + pid, wid); }
            if (0 <= m     && m     < nBeta && 0 <= k + 1 && k + 1 <= nTheta) { int pid = (m    ) * (nTheta + 1) + (k + 1); float wid = (1 - dm) * (    dk); atomicAdd(g + pid, spectrum[tid] * wid); atomicAdd(wt + pid, wid); }
            if (0 <= m + 1 && m + 1 < nBeta && 0 <= k     && k     <= nTheta) { int pid = (m + 1) * (nTheta + 1) + (k    ); float wid = (    dm) * (1 - dk); atomicAdd(g + pid, spectrum[tid] * wid); atomicAdd(wt + pid, wid); }
            if (0 <= m + 1 && m + 1 < nBeta && 0 <= k + 1 && k + 1 <= nTheta) { int pid = (m + 1) * (nTheta + 1) + (k + 1); float wid = (    dm) * (    dk); atomicAdd(g + pid, spectrum[tid] * wid); atomicAdd(wt + pid, wid); }
        }
    }
}''', 'equiphaseAverage')

def equiphaseAverage(spectrum, da, nBeta, nTheta, theta0, theta1, minf, a0, a1, ast, b, phi):
    ns, n = spectrum.shape[0], spectrum.shape[1]
    g = cp.zeros((nBeta, nTheta + 1), dtype = cp.float32)
    wt = cp.full((nBeta, nTheta + 1), 1e-6, dtype = cp.float32)

    kerEquiphaseAverage(
        (BLOCKDIM(ns * n * n), ),
        (BLOCKSIZE, ),
        (spectrum, da, g, wt, ns, n, nBeta, nTheta, F4(theta0), F4(theta1), F4(minf), F4(a0), F4(a1), F4(ast), F4(b), F4(phi))
    )
    return g / wt

kerEquation = cp.RawKernel(r'''
extern "C" __global__ void equation(
    const float* theta,     // (nBeta, nTheta + 1)
    const float* amp,       // (nBeta, nTheta + 1)
    float* f2,              // (nBeta, nTheta + 1)
    float* A,               // (n, 4)
    float* B,               // (n, )
    int nBeta,
    int nTheta,
    int lft,
    float theta0,
    float theta1,
    float maxf,
    float a0,
    float a1,
    float ast,
    float b,
    float phi
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n = nBeta * (nTheta + 1 - lft);
    if (tid < n) {
        int i = tid / (nTheta + 1 - lft);
        int j = tid % (nTheta + 1 - lft) + lft;
        int idx = i * (nTheta + 1) + j;
        float pi = 3.1415926535897932384626;
        float beta = pi / nBeta * i;
        float t = theta0 + (theta1 - theta0) / nTheta * j - phi;
        float s, c;
        sincosf(beta - ast, &s, &c);
        float a = a0 * c * c + a1 * s * s;
        float f2_ = 2 * t / (a + sqrtf(a * a + 4 * b * t));
        f2[idx] = f2_;
        if (f2_ <= maxf * maxf) {
            sincosf(2 * beta, &s, &c);
            float wt = amp[idx];
            A[tid * 4 + 0] = wt * f2_;
            A[tid * 4 + 1] = wt * f2_ * c;
            A[tid * 4 + 2] = wt * f2_ * s;
            A[tid * 4 + 3] = wt;
            B[tid] = wt * (theta[idx] - b * f2_ * f2_);
        }
    }
}''', 'equation')

def normalEquation(theta, amp, lft, theta0, theta1, maxf, a0, a1, ast, b, phi):
    nBeta, nTheta = theta.shape[0], theta.shape[1] - 1
    n = nBeta * (nTheta + 1 - lft)
    A = cp.zeros((n, 4), dtype = cp.float32)
    B = cp.zeros(n, dtype = cp.float32)
    f2 = cp.empty((nBeta, nTheta + 1), dtype = cp.float32)
    kerEquation(
        (BLOCKDIM(n), ),
        (BLOCKSIZE, ),
        (theta, amp, f2, A, B, nBeta, nTheta, lft, F4(theta0), F4(theta1), F4(maxf), F4(a0), F4(a1), F4(ast), F4(b), F4(phi))
    )
    return A.T @ A, A.T @ B, f2

kerCC = cp.RawKernel(r'''
extern "C" __global__ void getAB(
    const float* f2,        // (nBeta, nTheta + 1)
    const float* signal,    // (nBeta, nTheta + 1)
    float* A,               // (n, )
    float* B,               // (n, )
    int nBeta,
    int nTheta,
    int lft,
    float maxf,
    float a0,
    float a1,
    float ast,
    float b,
    float phi
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n = nBeta * (nTheta + 1 - lft);
    if (tid < n) {
        int i = tid / (nTheta + 1 - lft);
        int j = tid % (nTheta + 1 - lft) + lft;
        int idx = i * (nTheta + 1) + j;
        if (f2[idx] <= maxf * maxf) {
            float pi = 3.1415926535897932384626;
            float beta = pi / nBeta * i;
            float s, c;
            sincosf(beta - ast, &s, &c);
            float a = a0 * c * c + a1 * s * s;
            A[tid] = signal[idx];
            B[tid] = sin(a * f2[idx] + b * f2[idx] * f2[idx] + phi);
        }
    }
}''', 'getAB')

def computeCC(f2, signal, lft, maxf, a0, a1, ast, b, phi):
    nBeta, nTheta = f2.shape[0], f2.shape[1] - 1
    n = nBeta * (nTheta + 1 - lft)
    A = cp.zeros(n, dtype = cp.float32)
    B = cp.zeros(n, dtype = cp.float32)

    kerCC(
        (BLOCKDIM(n), ),
        (BLOCKSIZE, ),
        (f2, signal, A, B, nBeta, nTheta, lft, F4(maxf), F4(a0), F4(a1), F4(ast), F4(b), F4(phi))
    )
    cc = cp.dot(A, B) / cp.sqrt(cp.dot(A, A) * cp.dot(B, B))
    return cc.get()
