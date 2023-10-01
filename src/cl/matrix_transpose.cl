#define TS 16

__kernel void matrix_transpose(__global const float *as, __global float *res, unsigned int width, unsigned int height) {
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    __local float buf[TS][TS];

    if (i >= width || j >= height)
        return;

    buf[local_i][local_j] = as[j * width + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    res[i * height + j] = buf[local_i][local_j];
}
