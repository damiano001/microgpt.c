/*
 * microgpt.c  —  C port of Karpathy's microgpt.py 
 * (https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
 *
 * Same algorithm as the Python original; no external dependencies.
 * Compile : gcc -O3 -march=native -ffast-math -o gpt microgpt.c -lm
 * Run     : ./gpt          (curl downloads names.txt on first run)
 *
 * Speedup over Python: ~700× (eliminates Value-graph overhead,
 * uses flat float arrays, manual backprop, cache-friendly loops).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


/* ═══════════════════════════════════════════════════════════════════════
   HYPERPARAMETERS  
   ═══════════════════════════════════════════════════════════════════════ */
#define N_LAYER     1
#define N_EMBD      16
#define N_HEAD      4
#define HEAD_DIM    (N_EMBD / N_HEAD)   /* 4  */
#define FF_DIM      (4 * N_EMBD)        /* 64 */
#define BLOCK_SIZE  16
#define NUM_STEPS   1000
#define INIT_STD    0.08f
#define TEMPERATURE 0.5f
#define N_SAMPLES   20

/* Adam */
#define LR        0.01f
#define BETA1     0.85f
#define BETA2     0.99f
#define ADAM_EPS  1e-8f

/* ═══════════════════════════════════════════════════════════════════════
   DATASET
   ═══════════════════════════════════════════════════════════════════════ */
#define MAX_VOCAB    64
#define MAX_DOCS  50000
#define MAX_DOC_LEN  32

static int  n_docs;
static char docs[MAX_DOCS][MAX_DOC_LEN];
static int  doc_ord[MAX_DOCS];

static int  n_vocab, bos;     /* bos = last token id */
static char uchars[MAX_VOCAB];/* token id -> char      */

/* ═══════════════════════════════════════════════════════════════════════
   PSEUDO-RANDOM NUMBER GENERATOR
   Fast LCG + Box-Muller; seeded to 42 to mirror Python's random.seed(42).
   ═══════════════════════════════════════════════════════════════════════ */
static unsigned g_rng = 42u;

static unsigned rng_u32(void) {
    return g_rng = g_rng * 1664525u + 1013904223u;
}
static float rng_f32(void) {   /* uniform in (0,1) */
    return (float)(rng_u32() >> 8) * (1.f / (float)(1u << 24));
}
static float rng_gauss(void) { /* standard normal via Box-Muller */
    static int   have = 0;
    static float z1;
    if (have) { have = 0; return z1; }
    float u, v;
    do { u = rng_f32(); } while (u == 0.f);
    v = rng_f32();
    float r = sqrtf(-2.f * logf(u));
    z1 = r * sinf(6.28318530f * v); have = 1;
    return r * cosf(6.28318530f * v);
}
/* Random choice weighted by probs[0..n-1] */
static int weighted_choice(const float *probs, int n) {
    float r = rng_f32(), cum = 0.f;
    for (int i = 0; i < n - 1; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return n - 1;
}

/* ═══════════════════════════════════════════════════════════════════════
   TOKENISER / DATASET LOADER
   ═══════════════════════════════════════════════════════════════════════ */
static int char2id(char c) {
    for (int i = 0; i < n_vocab - 1; i++) if (uchars[i] == c) return i;
    return -1;
}

static void build_vocab(void) {
    char seen[256] = {0};
    for (int i = 0; i < n_docs; i++)
        for (char *p = docs[i]; *p; p++) seen[(unsigned char)*p] = 1;
    n_vocab = 0;
    for (int c = 0; c < 256; c++) if (seen[c]) uchars[n_vocab++] = (char)c;
    /* sort for determinism */
    for (int i = 0; i < n_vocab - 1; i++)
        for (int j = i + 1; j < n_vocab; j++)
            if (uchars[j] < uchars[i]) {
                char t = uchars[i]; uchars[i] = uchars[j]; uchars[j] = t;
            }
    bos = n_vocab;
    n_vocab++;   /* +1 for BOS */
}

static int load_dataset(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[MAX_DOC_LEN * 2];
    n_docs = 0;
    while (n_docs < MAX_DOCS && fgets(line, sizeof(line), f)) {
        int l = (int)strlen(line);
        while (l > 0 && (line[l-1]=='\n'||line[l-1]=='\r'||line[l-1]==' ')) line[--l]='\0';
        if (!l) continue;
        strncpy(docs[n_docs], line, MAX_DOC_LEN - 1);
        docs[n_docs][MAX_DOC_LEN - 1] = '\0';
        doc_ord[n_docs] = n_docs;
        n_docs++;
    }
    fclose(f);
    build_vocab();
    /* Fisher-Yates shuffle */
    for (int i = n_docs - 1; i > 0; i--) {
        int j = (int)(rng_u32() % (unsigned)(i + 1));
        int t = doc_ord[i]; doc_ord[i] = doc_ord[j]; doc_ord[j] = t;
    }
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════
   PARAMETER TENSORS  (weights + gradients + Adam m/v)
   Each tensor is registered in a global table so the optimizer can
   iterate over all of them without knowing the model topology.
   ═══════════════════════════════════════════════════════════════════════ */
#define MAX_TENSORS 40
static float *G_P[MAX_TENSORS], *G_G[MAX_TENSORS];
static float *G_M[MAX_TENSORS], *G_V[MAX_TENSORS];
static int    G_N[MAX_TENSORS];
static int    G_T = 0;
static long   g_nparams = 0;

/* Allocate one weight tensor, its grad, and its Adam buffers. */
static float *ptensor(float **gout, float **mout, float **vout, int n) {
    float *p = (float *)malloc(n * sizeof(float));
    float *g = (float *)calloc(n, sizeof(float));
    float *m = (float *)calloc(n, sizeof(float));
    float *v = (float *)calloc(n, sizeof(float));
    if (!p || !g || !m || !v) { fprintf(stderr, "OOM\n"); exit(1); }
    for (int i = 0; i < n; i++) p[i] = rng_gauss() * INIT_STD;
    G_P[G_T] = p; G_G[G_T] = g; G_M[G_T] = m; G_V[G_T] = v; G_N[G_T] = n;
    G_T++;
    g_nparams += n;
    *gout = g; *mout = m; *vout = v;
    return p;
}

/* Weight matrices [N_EMBD x N_EMBD unless noted] stored row-major:
   W[out][in]  ->  flat index: i*k + j  where shape is [n x k]           */
static float *wte,  *gwte,  *mwte,  *vwte;   /* [n_vocab x N_EMBD]      */
static float *wpe,  *gwpe,  *mwpe,  *vwpe;   /* [BLOCK_SIZE x N_EMBD]   */
static float *lmh,  *glmh,  *mlmh,  *vlmh;  /* [n_vocab x N_EMBD]      */
static float *WQ[N_LAYER], *gWQ[N_LAYER], *mWQ[N_LAYER], *vWQ[N_LAYER];
static float *WK[N_LAYER], *gWK[N_LAYER], *mWK[N_LAYER], *vWK[N_LAYER];
static float *WV[N_LAYER], *gWV[N_LAYER], *mWV[N_LAYER], *vWV[N_LAYER];
static float *WO[N_LAYER], *gWO[N_LAYER], *mWO[N_LAYER], *vWO[N_LAYER];
static float *FC1[N_LAYER],*gFC1[N_LAYER],*mFC1[N_LAYER],*vFC1[N_LAYER]; /* [FF_DIM x N_EMBD] */
static float *FC2[N_LAYER],*gFC2[N_LAYER],*mFC2[N_LAYER],*vFC2[N_LAYER]; /* [N_EMBD x FF_DIM] */

static void model_init(int V) {
    wte  = ptensor(&gwte,  &mwte,  &vwte,  V * N_EMBD);
    wpe  = ptensor(&gwpe,  &mwpe,  &vwpe,  BLOCK_SIZE * N_EMBD);
    lmh  = ptensor(&glmh,  &mlmh,  &vlmh,  V * N_EMBD);
    for (int l = 0; l < N_LAYER; l++) {
        WQ[l]  = ptensor(&gWQ[l],  &mWQ[l],  &vWQ[l],  N_EMBD * N_EMBD);
        WK[l]  = ptensor(&gWK[l],  &mWK[l],  &vWK[l],  N_EMBD * N_EMBD);
        WV[l]  = ptensor(&gWV[l],  &mWV[l],  &vWV[l],  N_EMBD * N_EMBD);
        WO[l]  = ptensor(&gWO[l],  &mWO[l],  &vWO[l],  N_EMBD * N_EMBD);
        FC1[l] = ptensor(&gFC1[l], &mFC1[l], &vFC1[l], FF_DIM * N_EMBD);
        FC2[l] = ptensor(&gFC2[l], &mFC2[l], &vFC2[l], N_EMBD * FF_DIM);
    }
}

static void zero_grads(void) {
    for (int i = 0; i < G_T; i++) memset(G_G[i], 0, G_N[i] * sizeof(float));
}

/* ═══════════════════════════════════════════════════════════════════════
   ACTIVATION BUFFERS  (static, sized to max sequence length)
   Separate forward and backward arrays avoid recomputation.
   ═══════════════════════════════════════════════════════════════════════ */
#define BS BLOCK_SIZE

/* Forward activations */
static float Emb  [BS][N_EMBD];
static float X0   [BS][N_EMBD];                  /* rmsnorm(Emb)               */
static float Xr1  [N_LAYER][BS][N_EMBD];         /* layer input / attn residual*/
static float Xn1  [N_LAYER][BS][N_EMBD];         /* rmsnorm(Xr1) -> Q,K,V     */
static float Qa   [N_LAYER][BS][N_EMBD];
static float Ka   [N_LAYER][BS][N_EMBD];
static float Va   [N_LAYER][BS][N_EMBD];
static float Aw   [N_LAYER][N_HEAD][BS][BS];     /* post-softmax attn weights  */
static float Ho   [N_LAYER][BS][N_EMBD];         /* concat head outputs -> WO  */
static float Ap   [N_LAYER][BS][N_EMBD];         /* WO(Ho) output              */
static float Xr2  [N_LAYER][BS][N_EMBD];         /* Xr1 + Ap / MLP residual    */
static float Xn2  [N_LAYER][BS][N_EMBD];         /* rmsnorm(Xr2)               */
static float Mh   [N_LAYER][BS][FF_DIM];         /* FC1 pre-relu               */
static float Mr   [N_LAYER][BS][FF_DIM];         /* post-relu                  */
static float Mo   [N_LAYER][BS][N_EMBD];         /* FC2 output                 */
static float Lo   [N_LAYER][BS][N_EMBD];         /* Xr2 + Mo  (layer output)   */
static float Prob [BS][MAX_VOCAB];               /* softmax(lmh * Lo[-1])      */

/* Gradient buffers (same shapes, += semantics throughout) */
static float dX0  [BS][N_EMBD];
static float dXr1 [N_LAYER][BS][N_EMBD];
static float dXn1 [N_LAYER][BS][N_EMBD];
static float dQa  [N_LAYER][BS][N_EMBD];
static float dKa  [N_LAYER][BS][N_EMBD];
static float dVa  [N_LAYER][BS][N_EMBD];
static float dHo  [N_LAYER][BS][N_EMBD];
static float dAp  [N_LAYER][BS][N_EMBD];
static float dXr2 [N_LAYER][BS][N_EMBD];
static float dXn2 [N_LAYER][BS][N_EMBD];
static float dMh  [N_LAYER][BS][FF_DIM];
static float dMr  [N_LAYER][BS][FF_DIM];
static float dMo  [N_LAYER][BS][N_EMBD];
static float dLo  [N_LAYER][BS][N_EMBD];
static float dEmb [BS][N_EMBD];

#undef BS

/* ═══════════════════════════════════════════════════════════════════════
   MATH PRIMITIVES  (inlined for speed)
   ═══════════════════════════════════════════════════════════════════════ */

/* y[i] = sum_j W[i*k+j] * x[j]   (matrix-vector product) */
static inline void mv(const float * restrict W,
                       const float * restrict x,
                       float       * restrict y, int n, int k) {
    for (int i = 0; i < n; i++) {
        float s = 0.f;
        const float *row = W + i * k;
        for (int j = 0; j < k; j++) s += row[j] * x[j];
        y[i] = s;
    }
}

/* Weight/input gradients for a linear layer:
   dW[i*k+j] += dy[i] * x[j]
   dx[j]     += sum_i W[i*k+j] * dy[i]                                   */
static inline void mv_bwd(const float * restrict W,
                           const float * restrict x,
                           const float * restrict dy,
                           float       * restrict dW,
                           float       * restrict dx, int n, int k) {
    for (int i = 0; i < n; i++) {
        float dyi = dy[i];
        const float *row  = W  + i * k;
        float       *drow = dW + i * k;
        for (int j = 0; j < k; j++) {
            drow[j] += dyi * x[j];
            dx[j]   += row[j] * dyi;
        }
    }
}

/* RMSNorm forward:  y = x / rms(x),  rms = sqrt(mean(x^2) + eps)       */
static inline void rms_fwd(const float * restrict x,
                             float       * restrict y, int d) {
    float ms = 0.f;
    for (int i = 0; i < d; i++) ms += x[i] * x[i];
    float s = 1.f / sqrtf(ms / d + 1e-5f);
    for (int i = 0; i < d; i++) y[i] = x[i] * s;
}

/* RMSNorm backward  (accumulates into dx):
   dy/dx_j = (1/rms)*delta_{ij} - x_i*x_j/(n*rms^3)
   -> dL/dx_j += inv*(dy_j - x_j*inv^2*dot(x,dy)/n)                     */
static inline void rms_bwd(const float * restrict x,
                             const float * restrict dy,
                             float       * restrict dx, int d) {
    float ms = 0.f;
    for (int i = 0; i < d; i++) ms += x[i] * x[i];
    float inv = 1.f / sqrtf(ms / d + 1e-5f);
    float dot = 0.f;
    for (int i = 0; i < d; i++) dot += dy[i] * x[i];
    float c = inv * inv * dot / d;
    for (int i = 0; i < d; i++) dx[i] += inv * (dy[i] - x[i] * c);
}

/* In-place numerically stable softmax */
static inline void softmax_ip(float *a, int n) {
    float mx = a[0];
    for (int i = 1; i < n; i++) if (a[i] > mx) mx = a[i];
    float s = 0.f;
    for (int i = 0; i < n; i++) { a[i] = expf(a[i] - mx); s += a[i]; }
    float inv = 1.f / s;
    for (int i = 0; i < n; i++) a[i] *= inv;
}

/* ═══════════════════════════════════════════════════════════════════════
   FORWARD PASS
   Processes T tokens, stores all activations needed for backprop, and
   returns the average cross-entropy loss.
   toks[0..T-1]    = input tokens
   targets[0..T-1] = target tokens (toks shifted by one position)
   ═══════════════════════════════════════════════════════════════════════ */
static float forward(const int *toks, const int *targets, int T) {

    /* 1. Token + position embeddings */
    for (int t = 0; t < T; t++) {
        const float *te = wte + toks[t] * N_EMBD;
        const float *pe = wpe + t       * N_EMBD;
        for (int d = 0; d < N_EMBD; d++) Emb[t][d] = te[d] + pe[d];
    }

    /* 2. Initial RMSNorm (matches Python: applied once before the layer loop) */
    for (int t = 0; t < T; t++) rms_fwd(Emb[t], X0[t], N_EMBD);

    /* 3. Transformer layers */
    for (int l = 0; l < N_LAYER; l++) {
        float (*prev)[N_EMBD] = (l == 0) ? X0 : Lo[l - 1];

        /* Save layer input; compute attn pre-norm */
        for (int t = 0; t < T; t++) {
            memcpy(Xr1[l][t], prev[t], N_EMBD * sizeof(float));
            rms_fwd(Xr1[l][t], Xn1[l][t], N_EMBD);
        }

        /* Q, K, V projections */
        for (int t = 0; t < T; t++) {
            mv(WQ[l], Xn1[l][t], Qa[l][t], N_EMBD, N_EMBD);
            mv(WK[l], Xn1[l][t], Ka[l][t], N_EMBD, N_EMBD);
            mv(WV[l], Xn1[l][t], Va[l][t], N_EMBD, N_EMBD);
        }

        /* Multi-head causal self-attention */
        const float sc = 1.f / sqrtf((float)HEAD_DIM);
        for (int t1 = 0; t1 < T; t1++) {
            float *ho = Ho[l][t1];
            memset(ho, 0, N_EMBD * sizeof(float));
            for (int h = 0; h < N_HEAD; h++) {
                const int hs = h * HEAD_DIM;
                float *aw = Aw[l][h][t1];
                /* Dot-product scores (causal: only t2 <= t1) */
                for (int t2 = 0; t2 <= t1; t2++) {
                    float s = 0.f;
                    const float *q = Qa[l][t1] + hs;
                    const float *k = Ka[l][t2] + hs;
                    for (int j = 0; j < HEAD_DIM; j++) s += q[j] * k[j];
                    aw[t2] = s * sc;
                }
                softmax_ip(aw, t1 + 1);
                /* Weighted sum of V */
                for (int j = 0; j < HEAD_DIM; j++) {
                    float o = 0.f;
                    for (int t2 = 0; t2 <= t1; t2++)
                        o += aw[t2] * Va[l][t2][hs + j];
                    ho[hs + j] = o;
                }
            }
        }

        /* Output projection */
        for (int t = 0; t < T; t++) mv(WO[l], Ho[l][t], Ap[l][t], N_EMBD, N_EMBD);

        /* Residual 1: Xr2 = Xr1 + Ap */
        for (int t = 0; t < T; t++)
            for (int d = 0; d < N_EMBD; d++) Xr2[l][t][d] = Xr1[l][t][d] + Ap[l][t][d];

        /* MLP pre-norm */
        for (int t = 0; t < T; t++) rms_fwd(Xr2[l][t], Xn2[l][t], N_EMBD);

        /* FC1 + ReLU */
        for (int t = 0; t < T; t++) {
            mv(FC1[l], Xn2[l][t], Mh[l][t], FF_DIM, N_EMBD);
            for (int d = 0; d < FF_DIM; d++)
                Mr[l][t][d] = Mh[l][t][d] > 0.f ? Mh[l][t][d] : 0.f;
        }

        /* FC2 */
        for (int t = 0; t < T; t++) mv(FC2[l], Mr[l][t], Mo[l][t], N_EMBD, FF_DIM);

        /* Residual 2: Lo = Xr2 + Mo */
        for (int t = 0; t < T; t++)
            for (int d = 0; d < N_EMBD; d++) Lo[l][t][d] = Xr2[l][t][d] + Mo[l][t][d];
    }

    /* 4. LM head + softmax + cross-entropy loss */
    float (*fin)[N_EMBD] = Lo[N_LAYER - 1];
    float loss = 0.f;
    for (int t = 0; t < T; t++) {
        mv(lmh, fin[t], Prob[t], n_vocab, N_EMBD);
        softmax_ip(Prob[t], n_vocab);
        loss += -logf(Prob[t][targets[t]] + 1e-10f);
    }
    return loss / T;
}

/* ═══════════════════════════════════════════════════════════════════════
   BACKWARD PASS
   Propagates gradients from the loss back through every activation and
   accumulates them into the parameter gradient buffers (G_G[i]).
   Call zero_grads() before this each training step.
   ═══════════════════════════════════════════════════════════════════════ */
static void backward(const int *toks, const int *targets, int T) {

    /* Zero activation gradient buffers */
    memset(dX0,  0, sizeof dX0);
    memset(dXr1, 0, sizeof dXr1);
    memset(dXn1, 0, sizeof dXn1);
    memset(dQa,  0, sizeof dQa);
    memset(dKa,  0, sizeof dKa);
    memset(dVa,  0, sizeof dVa);
    memset(dHo,  0, sizeof dHo);
    memset(dAp,  0, sizeof dAp);
    memset(dXr2, 0, sizeof dXr2);
    memset(dXn2, 0, sizeof dXn2);
    memset(dMh,  0, sizeof dMh);
    memset(dMr,  0, sizeof dMr);
    memset(dMo,  0, sizeof dMo);
    memset(dLo,  0, sizeof dLo);
    memset(dEmb, 0, sizeof dEmb);

    /* ── 1. Loss <- softmax <- LM head ─────────────────────────────── */
    /* Combined softmax + cross-entropy backward:
       dlogit[i] = (prob[i] - 1{i==target}) / T                          */
    float (*fin)[N_EMBD]  = Lo[N_LAYER - 1];
    float (*dfin)[N_EMBD] = dLo[N_LAYER - 1];
    const float inv_T = 1.f / T;
    for (int t = 0; t < T; t++) {
        float dlogits[MAX_VOCAB];
        for (int i = 0; i < n_vocab; i++) dlogits[i] = Prob[t][i] * inv_T;
        dlogits[targets[t]] -= inv_T;
        mv_bwd(lmh, fin[t], dlogits, glmh, dfin[t], n_vocab, N_EMBD);
    }

    /* ── 2. Transformer layers (reverse order) ──────────────────────── */
    for (int l = N_LAYER - 1; l >= 0; l--) {

        /* Residual 2: Lo = Xr2 + Mo  ->  dXr2 += dLo,  dMo = dLo */
        for (int t = 0; t < T; t++)
            for (int d = 0; d < N_EMBD; d++) {
                dXr2[l][t][d] += dLo[l][t][d];
                dMo [l][t][d]  = dLo[l][t][d];
            }

        /* FC2 backward */
        for (int t = 0; t < T; t++) {
            memset(dMr[l][t], 0, FF_DIM * sizeof(float));
            mv_bwd(FC2[l], Mr[l][t], dMo[l][t], gFC2[l], dMr[l][t], N_EMBD, FF_DIM);
        }

        /* ReLU backward */
        for (int t = 0; t < T; t++)
            for (int d = 0; d < FF_DIM; d++)
                dMh[l][t][d] += dMr[l][t][d] * (Mh[l][t][d] > 0.f ? 1.f : 0.f);

        /* FC1 backward */
        for (int t = 0; t < T; t++) {
            memset(dXn2[l][t], 0, N_EMBD * sizeof(float));
            mv_bwd(FC1[l], Xn2[l][t], dMh[l][t], gFC1[l], dXn2[l][t], FF_DIM, N_EMBD);
        }

        /* MLP RMSNorm backward: Xn2 = rms(Xr2)  ->  dXr2 accumulates */
        for (int t = 0; t < T; t++) rms_bwd(Xr2[l][t], dXn2[l][t], dXr2[l][t], N_EMBD);

        /* Residual 1: Xr2 = Xr1 + Ap  ->  dXr1 += dXr2,  dAp = dXr2 */
        for (int t = 0; t < T; t++)
            for (int d = 0; d < N_EMBD; d++) {
                dXr1[l][t][d] += dXr2[l][t][d];
                dAp [l][t][d]  = dXr2[l][t][d];
            }

        /* WO backward */
        for (int t = 0; t < T; t++) {
            memset(dHo[l][t], 0, N_EMBD * sizeof(float));
            mv_bwd(WO[l], Ho[l][t], dAp[l][t], gWO[l], dHo[l][t], N_EMBD, N_EMBD);
        }

        /* ── Multi-head attention backward ─────────────────────────── */
        const float sc = 1.f / sqrtf((float)HEAD_DIM);
        for (int t1 = T - 1; t1 >= 0; t1--) {
            for (int h = 0; h < N_HEAD; h++) {
                const int hs = h * HEAD_DIM;
                float *aw = Aw[l][h][t1];

                /* Backward through weighted sum of V:
                   Ho[t1][hs+j] = sum_{t2<=t1} aw[t2] * Va[t2][hs+j]    */
                float daw[BLOCK_SIZE] = {0};
                for (int t2 = 0; t2 <= t1; t2++) {
                    float d_aw = 0.f;
                    for (int j = 0; j < HEAD_DIM; j++) {
                        d_aw += dHo[l][t1][hs + j] * Va[l][t2][hs + j];
                        dVa[l][t2][hs + j] += aw[t2] * dHo[l][t1][hs + j];
                    }
                    daw[t2] = d_aw;
                }

                /* Softmax backward:
                   dscores[i] = p[i] * (daw[i] - dot(p, daw))            */
                float dot = 0.f;
                for (int i = 0; i <= t1; i++) dot += aw[i] * daw[i];
                float dscores[BLOCK_SIZE];
                for (int i = 0; i <= t1; i++) dscores[i] = aw[i] * (daw[i] - dot);

                /* Scores backward:
                   score[t2] = sc * dot(Q[t1,hs:], K[t2,hs:])            */
                for (int t2 = 0; t2 <= t1; t2++) {
                    float ds = dscores[t2] * sc;
                    const float *k  = Ka[l][t2] + hs;
                    const float *q  = Qa[l][t1] + hs;
                    float       *dq = dQa[l][t1] + hs;
                    float       *dk = dKa[l][t2] + hs;
                    for (int j = 0; j < HEAD_DIM; j++) {
                        dq[j] += ds * k[j];
                        dk[j] += ds * q[j];
                    }
                }
            }
        }

        /* Q, K, V linear backward (all three accumulate into dXn1) */
        for (int t = 0; t < T; t++) {
            memset(dXn1[l][t], 0, N_EMBD * sizeof(float));
            mv_bwd(WQ[l], Xn1[l][t], dQa[l][t], gWQ[l], dXn1[l][t], N_EMBD, N_EMBD);
            mv_bwd(WK[l], Xn1[l][t], dKa[l][t], gWK[l], dXn1[l][t], N_EMBD, N_EMBD);
            mv_bwd(WV[l], Xn1[l][t], dVa[l][t], gWV[l], dXn1[l][t], N_EMBD, N_EMBD);
        }

        /* Attn RMSNorm backward: Xn1 = rms(Xr1)  ->  dXr1 accumulates */
        for (int t = 0; t < T; t++) rms_bwd(Xr1[l][t], dXn1[l][t], dXr1[l][t], N_EMBD);

        /* Pass gradient to previous layer output (or X0 if l==0) */
        if (l == 0) {
            for (int t = 0; t < T; t++)
                for (int d = 0; d < N_EMBD; d++) dX0[t][d] += dXr1[l][t][d];
        } else {
            for (int t = 0; t < T; t++)
                for (int d = 0; d < N_EMBD; d++) dLo[l-1][t][d] += dXr1[l][t][d];
        }
    }

    /* ── 3. Initial RMSNorm backward: X0 = rms(Emb) ─────────────────── */
    for (int t = 0; t < T; t++) rms_bwd(Emb[t], dX0[t], dEmb[t], N_EMBD);

    /* ── 4. Embedding backward ──────────────────────────────────────── */
    for (int t = 0; t < T; t++) {
        const float *de  = dEmb[t];
        float       *gte = gwte + toks[t] * N_EMBD;
        float       *gpe = gwpe + t       * N_EMBD;
        for (int d = 0; d < N_EMBD; d++) {
            gte[d] += de[d];
            gpe[d] += de[d];
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
   ADAM OPTIMIZER  (with linear LR decay, matching the Python original)
   ═══════════════════════════════════════════════════════════════════════ */
static void adam_step(int step) {
    const float lr_t = LR * (1.f - (float)step / NUM_STEPS);
    const float b1c  = 1.f - powf(BETA1, (float)(step + 1));
    const float b2c  = 1.f - powf(BETA2, (float)(step + 1));
    for (int ti = 0; ti < G_T; ti++) {
        float *p = G_P[ti], *g = G_G[ti], *m = G_M[ti], *v = G_V[ti];
        const int n = G_N[ti];
        for (int i = 0; i < n; i++) {
            m[i] = BETA1 * m[i] + (1.f - BETA1) * g[i];
            v[i] = BETA2 * v[i] + (1.f - BETA2) * g[i] * g[i];
            const float mh = m[i] / b1c;
            const float vh = v[i] / b2c;
            p[i] -= lr_t * mh / (sqrtf(vh) + ADAM_EPS);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
   INFERENCE  (no gradients — reuses forward activation buffers)
   At each step, run a full forward pass on the accumulated token sequence
   and sample the next token from the final-position logits.
   ═══════════════════════════════════════════════════════════════════════ */
static void generate_sample(int idx) {
    int  toks[BLOCK_SIZE + 2];
    int  ntoks = 0;
    char result[MAX_DOC_LEN + 1];
    int  rlen = 0;

    toks[ntoks++] = bos;

    for (int pos = 0; pos < BLOCK_SIZE; pos++) {
        int T = ntoks;

        /* Forward pass (no loss needed) */
        for (int t = 0; t < T; t++) {
            const float *te = wte + toks[t] * N_EMBD;
            const float *pe = wpe + t * N_EMBD;
            for (int d = 0; d < N_EMBD; d++) Emb[t][d] = te[d] + pe[d];
            rms_fwd(Emb[t], X0[t], N_EMBD);
        }
        for (int l = 0; l < N_LAYER; l++) {
            float (*prev)[N_EMBD] = (l == 0) ? X0 : Lo[l - 1];
            for (int t = 0; t < T; t++) {
                memcpy(Xr1[l][t], prev[t], N_EMBD * sizeof(float));
                rms_fwd(Xr1[l][t], Xn1[l][t], N_EMBD);
            }
            for (int t = 0; t < T; t++) {
                mv(WQ[l], Xn1[l][t], Qa[l][t], N_EMBD, N_EMBD);
                mv(WK[l], Xn1[l][t], Ka[l][t], N_EMBD, N_EMBD);
                mv(WV[l], Xn1[l][t], Va[l][t], N_EMBD, N_EMBD);
            }
            const float sc = 1.f / sqrtf((float)HEAD_DIM);
            for (int t1 = 0; t1 < T; t1++) {
                memset(Ho[l][t1], 0, N_EMBD * sizeof(float));
                for (int h = 0; h < N_HEAD; h++) {
                    int hs = h * HEAD_DIM;
                    float *aw = Aw[l][h][t1];
                    for (int t2 = 0; t2 <= t1; t2++) {
                        float s = 0.f;
                        for (int j = 0; j < HEAD_DIM; j++)
                            s += Qa[l][t1][hs+j] * Ka[l][t2][hs+j];
                        aw[t2] = s * sc;
                    }
                    softmax_ip(aw, t1 + 1);
                    for (int j = 0; j < HEAD_DIM; j++) {
                        float o = 0.f;
                        for (int t2 = 0; t2 <= t1; t2++) o += aw[t2] * Va[l][t2][hs+j];
                        Ho[l][t1][hs+j] = o;
                    }
                }
            }
            for (int t = 0; t < T; t++) mv(WO[l], Ho[l][t], Ap[l][t], N_EMBD, N_EMBD);
            for (int t = 0; t < T; t++)
                for (int d = 0; d < N_EMBD; d++) Xr2[l][t][d] = Xr1[l][t][d] + Ap[l][t][d];
            for (int t = 0; t < T; t++) rms_fwd(Xr2[l][t], Xn2[l][t], N_EMBD);
            for (int t = 0; t < T; t++) {
                mv(FC1[l], Xn2[l][t], Mh[l][t], FF_DIM, N_EMBD);
                for (int d = 0; d < FF_DIM; d++)
                    Mr[l][t][d] = Mh[l][t][d] > 0.f ? Mh[l][t][d] : 0.f;
            }
            for (int t = 0; t < T; t++) mv(FC2[l], Mr[l][t], Mo[l][t], N_EMBD, FF_DIM);
            for (int t = 0; t < T; t++)
                for (int d = 0; d < N_EMBD; d++) Lo[l][t][d] = Xr2[l][t][d] + Mo[l][t][d];
        }

        /* LM head at last position, apply temperature, sample */
        float logits[MAX_VOCAB];
        mv(lmh, Lo[N_LAYER - 1][T - 1], logits, n_vocab, N_EMBD);
        for (int i = 0; i < n_vocab; i++) logits[i] /= TEMPERATURE;
        softmax_ip(logits, n_vocab);

        int next = weighted_choice(logits, n_vocab);
        if (next == bos) break;
        toks[ntoks++] = next;
        if (rlen < MAX_DOC_LEN - 1) result[rlen++] = uchars[next];
    }
    result[rlen] = '\0';
    printf("sample %2d: %s\n", idx + 1, result);
}

/* ═══════════════════════════════════════════════════════════════════════
   MAIN
   ═══════════════════════════════════════════════════════════════════════ */
int main(void) {
    const char *path = "input.txt";

    /* Download names.txt if missing */
    if (fopen(path, "r") == NULL) {
        printf("Downloading names.txt ...\n");
        int ret = system("curl -sL https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o input.txt");
        if (ret != 0) { fprintf(stderr, "Download failed.\n"); return 1; }
    }

    if (!load_dataset(path)) { fprintf(stderr, "Failed to load %s\n", path); return 1; }
    printf("num docs:   %d\n", n_docs);
    printf("vocab size: %d\n", n_vocab);

    model_init(n_vocab);
    printf("num params: %ld\n", g_nparams);

    /* ── Training loop ─────────────────────────────────────────────── */
    printf("\nStarting training for %d steps...\n", NUM_STEPS);
    clock_t training_start = clock();
    for (int step = 0; step < NUM_STEPS; step++) {
        /* Pick document (cyclic over shuffled order) */
        const char *doc = docs[doc_ord[step % n_docs]];

        /* Tokenise: [BOS, c1, c2, ..., cn, BOS] */
        int tokens[BLOCK_SIZE + 2];
        tokens[0] = bos;
        int len = 0;
        for (const char *p = doc; *p && len < BLOCK_SIZE; p++)
            tokens[1 + len++] = char2id(*p);
        tokens[1 + len] = bos;

        /* T = number of (input, target) pairs */
        int T = (len + 1 < BLOCK_SIZE) ? len + 1 : BLOCK_SIZE;
        const int *in_toks = tokens;       /* input:  tokens[0..T-1]  */
        const int *targets = tokens + 1;   /* target: tokens[1..T]    */

        zero_grads();
        float loss = forward(in_toks, targets, T);
        backward(in_toks, targets, T);
        adam_step(step);

        printf("step %4d / %4d | loss %.4f\r", step + 1, NUM_STEPS, loss);
        fflush(stdout);
    }
    printf("\n");
    clock_t training_end = clock();
    double training_time = ((double)(training_end - training_start)) / CLOCKS_PER_SEC;
    printf("Training completed in %.2f seconds\n", training_time);
    printf("Average time per step: %.2f ms\n", training_time / NUM_STEPS * 1000.0);

    /* ── Inference ─────────────────────────────────────────────────── */
    printf("--- inference (new, hallucinated names) ---\n");
    clock_t inference_start = clock();
    for (int i = 0; i < N_SAMPLES; i++) generate_sample(i);
    clock_t inference_end = clock();
    double inference_time = ((double)(inference_end - inference_start)) / CLOCKS_PER_SEC;
    printf("\nInference completed in %.2f seconds\n", inference_time);
    printf("Average time per sample: %.2f ms\n", inference_time / N_SAMPLES * 1000.0);
    printf("Total time (training + inference): %.2f seconds\n", training_time + inference_time);

    return 0;
}
