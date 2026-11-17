/* LSTM int8 + testa HL per l'inferenza riga per riga (1 thread), chiamata da Python con ctypes.
 *
 * Per strato: gates = sx * sw[n] * (Wx_q . x_q) + sh * sw[n] * (Wh_q . h_q) + b      (porte i, f, g, o come PyTorch)
 *   pesi int8 simmetrici per canale d'uscita; attivazioni int8 simmetriche per vettore (scala dinamica max|v|/127),
 *   portate a u8 (+128) per l'istruzione VNNI vpdpbusd (u8 x s8 -> int32), con la correzione 128 * somma dei pesi.
 * Testa: Linear con pesi int8 per colonna -> 2 x K logit -> softmax -> E[|c| c] / E[|c|] sui centri c dei bin, x scala.
 * Insieme di modelli: media delle uscite (pesi dati), x scala finale.
 *
 * Percorso veloce AVX-512 VNNI (Zen4, Xeon recenti) scelto a runtime; altrimenti C semplice (stesso risultato numerico
 * salvo l'approssimazione di exp).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#define NB 128                     /* uscite per blocco del prodotto matrice-vettore (8 registri zmm da 16) */
#define MAXL 6

typedef struct {
    int in, in4, N, Np;            /* ingresso, ingresso arrotondato a 4, 4H, 4H arrotondato a NB */
    int8_t *W;                     /* blocchi: [Np/NB][in4/4][NB][4] */
    int32_t *rs;                   /* somma dei pesi per uscita (Np) */
    float *sw;                     /* scala dei pesi per uscita (Np) */
} QMat;

typedef struct {
    int L, F, K;                   /* strati, feature, bin per target */
    int H[MAXL];
    float *in_scale;               /* F: moltiplicatore per feature (le colonne dei pesi sono già divise) */
    QMat Wx[MAXL], Wh[MAXL];
    float *b[MAXL];                /* 4H */
    float *h[MAXL], *c[MAXL];      /* stato */
    int8_t *Woq; float *sco;       /* testa int8 per colonna (H[L-1] x 2K) e scale (2K) */
    float *bo;                     /* 2K */
    float *centers;                /* K */
    float scale[2];
    /* memoria di lavoro */
    float *z, *xin, *lg;
    uint8_t *u;
    int32_t *acc;
} Model;

static void *amalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, (n + 63) / 64 * 64)) return NULL;
    memset(p, 0, (n + 63) / 64 * 64); return p;
}

static int g_vnni = -1, g_avx2 = -1;
static int has_avx2(void) {
    if (g_avx2 < 0) { __builtin_cpu_init(); g_avx2 = __builtin_cpu_supports("avx2"); }
    return g_avx2;
}
static int has_vnni(void) {
    if (g_vnni < 0) {
        __builtin_cpu_init();
        g_vnni = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl")
                 && __builtin_cpu_supports("avx512vnni");
    }
    return g_vnni;
}
int lstm_has_vnni(void) { return has_vnni(); }

/* w: (N, in) int8 riga per uscita; sw: (N) */
static void qmat_set(QMat *q, int in, int N, const int8_t *w, const float *sw) {
    q->in = in; q->in4 = (in + 3) / 4 * 4; q->N = N; q->Np = (N + NB - 1) / NB * NB;
    int K4 = q->in4 / 4;
    q->W = (int8_t *)amalloc((size_t)q->Np * q->in4);
    q->rs = (int32_t *)amalloc(sizeof(int32_t) * q->Np);
    q->sw = (float *)amalloc(sizeof(float) * q->Np);
    for (int n = 0; n < N; n++) {
        int32_t s = 0;
        int blk = n / NB, j = n % NB;
        for (int k = 0; k < in; k++) {
            int8_t v = w[(size_t)n * in + k];
            s += v;
            q->W[(((size_t)blk * K4 + k / 4) * NB + j) * 4 + k % 4] = v;
        }
        q->rs[n] = s; q->sw[n] = sw[n];
    }
}

/* attivazioni -> u8 (q + 128), ritorna la scala; pad a 4 con 128 (q = 0) */
static float quant_u8(const float *v, int n, int n4, uint8_t *u) {
    float m = 0.f;
    for (int i = 0; i < n; i++) { float a = fabsf(v[i]); m = a > m ? a : m; }
    float s = m > 0.f ? m / 127.f : 1.f, inv = 1.f / s;
    for (int i = 0; i < n; i++) { int q = (int)lrintf(v[i] * inv); q = q > 127 ? 127 : (q < -127 ? -127 : q); u[i] = (uint8_t)(q + 128); }
    for (int i = n; i < n4; i++) u[i] = 128;
    return s;
}

/* acc[n] = sum_k u[k] * W[n,k] (u8 x s8) */
static void matvec_generic(const QMat *q, const uint8_t *u, int32_t *acc) {
    int K4 = q->in4 / 4;
    for (int blk = 0; blk < q->Np / NB; blk++) {
        int32_t a[NB]; memset(a, 0, sizeof a);
        const int8_t *W = q->W + (size_t)blk * K4 * NB * 4;
        for (int k4 = 0; k4 < K4; k4++) {
            const uint8_t *uu = u + 4 * k4;
            const int8_t *w = W + (size_t)k4 * NB * 4;
            for (int j = 0; j < NB; j++)
                a[j] += uu[0] * w[4 * j] + uu[1] * w[4 * j + 1] + uu[2] * w[4 * j + 2] + uu[3] * w[4 * j + 3];
        }
        memcpy(acc + blk * NB, a, sizeof a);
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
static void matvec_vnni(const QMat *q, const uint8_t *u, int32_t *acc) {
    int K4 = q->in4 / 4;
    for (int blk = 0; blk < q->Np / NB; blk++) {
        __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        const int8_t *W = q->W + (size_t)blk * K4 * NB * 4;
        for (int k4 = 0; k4 < K4; k4++) {
            int32_t ub; memcpy(&ub, u + 4 * k4, 4);
            __m512i x = _mm512_set1_epi32(ub);
            const __m512i *w = (const __m512i *)(W + (size_t)k4 * NB * 4);
            a0 = _mm512_dpbusd_epi32(a0, x, _mm512_load_si512(w + 0));
            a1 = _mm512_dpbusd_epi32(a1, x, _mm512_load_si512(w + 1));
            a2 = _mm512_dpbusd_epi32(a2, x, _mm512_load_si512(w + 2));
            a3 = _mm512_dpbusd_epi32(a3, x, _mm512_load_si512(w + 3));
            a4 = _mm512_dpbusd_epi32(a4, x, _mm512_load_si512(w + 4));
            a5 = _mm512_dpbusd_epi32(a5, x, _mm512_load_si512(w + 5));
            a6 = _mm512_dpbusd_epi32(a6, x, _mm512_load_si512(w + 6));
            a7 = _mm512_dpbusd_epi32(a7, x, _mm512_load_si512(w + 7));
        }
        __m512i *o = (__m512i *)(acc + blk * NB);
        _mm512_store_si512(o + 0, a0); _mm512_store_si512(o + 1, a1); _mm512_store_si512(o + 2, a2); _mm512_store_si512(o + 3, a3);
        _mm512_store_si512(o + 4, a4); _mm512_store_si512(o + 5, a5); _mm512_store_si512(o + 6, a6); _mm512_store_si512(o + 7, a7);
    }
}


/* AVX2 senza VNNI: q = u ^ 0x80 (s8), maddubs(|q|, sign(w, q)) non satura (|q|, |w| <= 127); ritorna sum q*w + 128 * rs
 * per restare coerente con la convenzione u8 dei chiamanti */
__attribute__((target("avx2")))
static void matvec_avx2(const QMat *q, const uint8_t *u, int32_t *acc) {
    int K4 = q->in4 / 4;
    const __m256i ones = _mm256_set1_epi16(1), flip = _mm256_set1_epi8((char)0x80);
    for (int blk = 0; blk < q->Np / NB; blk++) {
        const int8_t *W = q->W + (size_t)blk * K4 * NB * 4;
        for (int half = 0; half < 2; half++) {
            __m256i a[8];
            for (int r = 0; r < 8; r++) a[r] = _mm256_setzero_si256();
            for (int k4 = 0; k4 < K4; k4++) {
                int32_t ub; memcpy(&ub, u + 4 * k4, 4);
                __m256i x = _mm256_xor_si256(_mm256_set1_epi32(ub), flip);
                __m256i ax = _mm256_abs_epi8(x);
                const __m256i *w = (const __m256i *)(W + (size_t)k4 * NB * 4 + half * 256);
                for (int r = 0; r < 8; r++) {
                    __m256i ws = _mm256_sign_epi8(_mm256_load_si256(w + r), x);
                    a[r] = _mm256_add_epi32(a[r], _mm256_madd_epi16(_mm256_maddubs_epi16(ax, ws), ones));
                }
            }
            for (int r = 0; r < 8; r++) {
                int base = blk * NB + half * 64 + r * 8;
                __m256i rs = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(q->rs + base)), 7);
                _mm256_storeu_si256((__m256i *)(acc + base), _mm256_add_epi32(a[r], rs));
            }
        }
    }
}

/* exp vettoriale (Cephes: 2^n * p(r), errore relativo ~2e-7) */
__attribute__((target("avx512f")))
static inline __m512 exp512(__m512 x) {
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.f)), _mm512_set1_ps(88.f));
    __m512 fx = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693359375f), x);
    x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.12194440e-4f), x);
    __m512 y = _mm512_set1_ps(1.9875691500E-4f);
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507E-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073E-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894E-2f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201E-1f));
    y = _mm512_fmadd_ps(y, _mm512_mul_ps(x, x), _mm512_add_ps(x, _mm512_set1_ps(1.f)));
    return _mm512_scalef_ps(y, fx);
}

__attribute__((target("avx512f")))
static void cell_vnni(int H, const float *z, float *c, float *h) {
    const __m512 one = _mm512_set1_ps(1.f), two = _mm512_set1_ps(2.f);
    for (int j = 0; j < H; j += 16) {
        __m512 zi = _mm512_loadu_ps(z + j), zf = _mm512_loadu_ps(z + H + j), zg = _mm512_loadu_ps(z + 2 * H + j), zo = _mm512_loadu_ps(z + 3 * H + j);
        __m512 si = _mm512_div_ps(one, _mm512_add_ps(one, exp512(_mm512_sub_ps(_mm512_setzero_ps(), zi))));
        __m512 sf = _mm512_div_ps(one, _mm512_add_ps(one, exp512(_mm512_sub_ps(_mm512_setzero_ps(), zf))));
        __m512 so = _mm512_div_ps(one, _mm512_add_ps(one, exp512(_mm512_sub_ps(_mm512_setzero_ps(), zo))));
        __m512 tg = _mm512_sub_ps(_mm512_div_ps(two, _mm512_add_ps(one, exp512(_mm512_mul_ps(_mm512_set1_ps(-2.f), zg)))), one);
        __m512 cn = _mm512_fmadd_ps(sf, _mm512_loadu_ps(c + j), _mm512_mul_ps(si, tg));
        __m512 tc = _mm512_sub_ps(_mm512_div_ps(two, _mm512_add_ps(one, exp512(_mm512_mul_ps(_mm512_set1_ps(-2.f), cn)))), one);
        _mm512_storeu_ps(c + j, cn);
        _mm512_storeu_ps(h + j, _mm512_mul_ps(so, tc));
    }
}

static inline float sigm(float x) { return 1.f / (1.f + expf(-x)); }
static void cell_generic(int H, const float *z, float *c, float *h) {
    for (int j = 0; j < H; j++) {
        float cn = sigm(z[H + j]) * c[j] + sigm(z[j]) * tanhf(z[2 * H + j]);
        c[j] = cn; h[j] = sigm(z[3 * H + j]) * tanhf(cn);
    }
}

void *lstm_create(int L, int F, const int *H, int K) {
    if (L > MAXL) return NULL;
    Model *m = (Model *)calloc(1, sizeof(Model));
    m->L = L; m->F = F; m->K = K;
    int mx = F;
    for (int l = 0; l < L; l++) { m->H[l] = H[l]; if (4 * H[l] + NB > mx) mx = 4 * H[l] + NB; if (H[l] > mx) mx = H[l]; }
    if (2 * K > mx) mx = 2 * K;
    mx += 64;
    m->z = (float *)amalloc(sizeof(float) * mx); m->xin = (float *)amalloc(sizeof(float) * mx);
    m->lg = (float *)amalloc(sizeof(float) * mx);
    m->u = (uint8_t *)amalloc(mx); m->acc = (int32_t *)amalloc(sizeof(int32_t) * mx);
    m->in_scale = (float *)amalloc(sizeof(float) * F);
    for (int i = 0; i < F; i++) m->in_scale[i] = 1.f;
    for (int l = 0; l < L; l++) {
        int Hp = (H[l] + 15) / 16 * 16;
        m->h[l] = (float *)amalloc(sizeof(float) * Hp); m->c[l] = (float *)amalloc(sizeof(float) * Hp);
        m->b[l] = (float *)amalloc(sizeof(float) * 4 * H[l]);
    }
    return m;
}

void lstm_set_input_scale(void *mp, const float *s) { Model *m = (Model *)mp; memcpy(m->in_scale, s, sizeof(float) * m->F); }

/* wx: (4H, in) int8, swx: (4H); wh: (4H, H) int8, swh: (4H); b: (4H) = b_ih + b_hh */
void lstm_set_layer(void *mp, int l, const int8_t *wx, const float *swx, const int8_t *wh, const float *swh, const float *b) {
    Model *m = (Model *)mp;
    int in = l == 0 ? m->F : m->H[l - 1], N = 4 * m->H[l];
    qmat_set(&m->Wx[l], in, N, wx, swx);
    qmat_set(&m->Wh[l], m->H[l], N, wh, swh);
    memcpy(m->b[l], b, sizeof(float) * N);
}

/* testa lineare int8: woq (Hl, 2K) int8 (trasposta di head.weight), sco (2K) scale per colonna, bo (2K) bias;
   centers (K) centri dei bin, s0/s1 scala di calibrazione dei due target */
void lstm_set_head_q(void *mp, const int8_t *woq, const float *sco, const float *bo, const float *centers, float s0, float s1) {
    Model *m = (Model *)mp;
    int Hl = m->H[m->L - 1], K2 = 2 * m->K;
    m->Woq = (int8_t *)amalloc((size_t)Hl * K2); memcpy(m->Woq, woq, (size_t)Hl * K2);
    m->sco = (float *)amalloc(sizeof(float) * K2); memcpy(m->sco, sco, sizeof(float) * K2);
    m->bo = (float *)amalloc(sizeof(float) * K2); memcpy(m->bo, bo, sizeof(float) * K2);
    m->centers = (float *)amalloc(sizeof(float) * m->K); memcpy(m->centers, centers, sizeof(float) * m->K);
    m->scale[0] = s0; m->scale[1] = s1;
}

static void axpy_rows_q(const float *x, int n_in, const int8_t *Wq, int n_out, float *y) {
    for (int k = 0; k < n_in; k++) {
        float a = x[k];
        if (a == 0.f) continue;
        const int8_t *w = Wq + (size_t)k * n_out;
        for (int j = 0; j < n_out; j++) y[j] += a * (float)w[j];
    }
}
__attribute__((target("avx512f")))
static void axpy_rows_q512(const float *x, int n_in, const int8_t *Wq, int n_out, float *y) {
    int n16 = n_out / 16 * 16;
    for (int k = 0; k < n_in; k++) {
        float a = x[k];
        if (a == 0.f) continue;
        __m512 av = _mm512_set1_ps(a);
        const int8_t *w = Wq + (size_t)k * n_out;
        int j = 0;
        for (; j < n16; j += 16) {
            __m512 wf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(w + j))));
            _mm512_storeu_ps(y + j, _mm512_fmadd_ps(av, wf, _mm512_loadu_ps(y + j)));
        }
        for (; j < n_out; j++) y[j] += a * (float)w[j];
    }
}

void lstm_reset(void *mp) {
    Model *m = (Model *)mp;
    for (int l = 0; l < m->L; l++) { int Hp = (m->H[l] + 15) / 16 * 16; memset(m->h[l], 0, sizeof(float) * Hp); memset(m->c[l], 0, sizeof(float) * Hp); }
}

static void model_step(Model *m, const float *x, float *out) {
    int v = has_vnni();
    const float *in = m->xin;
    for (int i = 0; i < m->F; i++) m->xin[i] = x[i] * m->in_scale[i];
    for (int l = 0; l < m->L; l++) {
        int H = m->H[l], N = 4 * H;
        QMat *qx = &m->Wx[l], *qh = &m->Wh[l];
        float sx = quant_u8(in, qx->in, qx->in4, m->u);
        if (v) matvec_vnni(qx, m->u, m->acc); else if (has_avx2()) matvec_avx2(qx, m->u, m->acc); else matvec_generic(qx, m->u, m->acc);
        for (int n = 0; n < N; n++) m->z[n] = m->b[l][n] + sx * qx->sw[n] * (float)(m->acc[n] - 128 * qx->rs[n]);
        float sh = quant_u8(m->h[l], H, qh->in4, m->u);
        if (v) matvec_vnni(qh, m->u, m->acc); else if (has_avx2()) matvec_avx2(qh, m->u, m->acc); else matvec_generic(qh, m->u, m->acc);
        for (int n = 0; n < N; n++) m->z[n] += sh * qh->sw[n] * (float)(m->acc[n] - 128 * qh->rs[n]);
        if (v && H % 16 == 0) cell_vnni(H, m->z, m->c[l], m->h[l]); else cell_generic(H, m->z, m->c[l], m->h[l]);
        in = m->h[l];
    }
    int Hl = m->H[m->L - 1], K = m->K, K2 = 2 * K;
    const float *hin = m->h[m->L - 1];
    memset(m->lg, 0, sizeof(float) * K2);
    if (v) axpy_rows_q512(hin, Hl, m->Woq, K2, m->lg); else axpy_rows_q(hin, Hl, m->Woq, K2, m->lg);
    for (int j = 0; j < K2; j++) m->lg[j] = m->bo[j] + m->sco[j] * m->lg[j];
    for (int t = 0; t < 2; t++) {                 /* softmax sui bin -> E[|c| c] / E[|c|] */
        float *g = m->lg + t * K, mx = g[0];
        for (int k = 1; k < K; k++) mx = g[k] > mx ? g[k] : mx;
        double num = 0, den = 0;
        for (int k = 0; k < K; k++) {
            float p = expf(g[k] - mx), cc = m->centers[k];
            float a = fabsf(cc);
            num += (double)p * a * cc; den += (double)p * a;
        }
        out[t] = (float)(num / (den > 1e-6 * 1 ? den : 1e-6)) * m->scale[t];
    }
}

/* un passo per tutti i modelli dell'insieme; out = (sum_i w_i * out_i) * fin_scale */
void ens_step(void **ms, int n, const float *w, const float *x, float *out, float fs0, float fs1) {
    float acc0 = 0.f, acc1 = 0.f, o[2];
    for (int i = 0; i < n; i++) { model_step((Model *)ms[i], x, o); acc0 += w[i] * o[0]; acc1 += w[i] * o[1]; }
    out[0] = acc0 * fs0; out[1] = acc1 * fs1;
}
void ens_reset(void **ms, int n) { for (int i = 0; i < n; i++) lstm_reset(ms[i]); }
