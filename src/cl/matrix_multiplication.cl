#define TS 16
#define TW 16

__kernel void matrix_multiplication_global(__global const float *a, __global const float *b, __global float *c,
                                           const unsigned int M, const unsigned int N, const unsigned int K) {
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if (i >= N || j >= M)
        return;

    float buf = 0.f;

    for (unsigned int k = 0; k < K; ++k)
        buf += a[j * M + k] * b[k * N + i];

    c[j * N + i] = buf;
}

__kernel void matrix_multiplication_local(__global const float *a, __global const float *b, __global float *c,
                                          const unsigned int M, const unsigned int N, const unsigned int K) {
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    const unsigned int i = local_i + get_group_id(0);
    const unsigned int j = local_j + get_group_id(1);

    if (i >= M && j >= N)
        return;

    __local float abuf[TS][TS];
    __local float bbuf[TS][TS];

    float buf = 0.f;

    const unsigned int num = K / TS;

    for (unsigned int t = 0; t < num; ++t) {
        const unsigned int tiled_i = TS * t + local_i;
        const unsigned int tiled_j = TS * t + local_j;

        abuf[local_j][local_i] = a[tiled_j * M + i];
        bbuf[local_j][local_i] = b[j * K + tiled_i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int p = 0; p < TS; p++) {
            buf += abuf[p][local_i] * bbuf[local_j][p];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[j * M + i] = buf;
}

__kernel void matrix_multiplication_more_work(__global const float *a, __global const float *b, __global float *c,
                                              const unsigned int M, const unsigned int N, const unsigned int K) {

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);
    const unsigned int global_i = get_group_id(0) * TS + local_i;
    const unsigned int global_j = get_group_id(1) * TS + local_j;

    const unsigned int RTS = TS / TW;

    __local float abuf[TS][TS];
    __local float bbuf[TS][TS];

    float buf[TW];
    for (int t = 0; t < TW; ++t)
        buf[t] = 0.0;

    for (unsigned int t = 0; t < K; t += TS) {
        unsigned int tile_i = t + local_i;
        unsigned int tile_j = t + local_j;

        for (unsigned int w = 0; w < TS; w += RTS) {
            abuf[local_j + w][local_i] = 0;
            bbuf[local_j + w][local_i] = 0;

            if (tile_i < K && global_j + w < M)
                abuf[local_j + w][local_i] = a[(global_j + w) * K + tile_i];
            if (global_i < N && tile_j + w < K)
                bbuf[local_j + w][local_i] = b[(tile_j + w) * N + global_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int k = 0; k < TS; k++)
            for (unsigned int w = 0; w < TW; w++)
                buf[w] += abuf[local_j + RTS * w][k] * bbuf[k][local_i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (unsigned int w = 0; w < TW; w++)
        if ((global_j + RTS * w) < M && global_i < N)
            c[(global_j + RTS * w) * N + global_i] = buf[w];
}
