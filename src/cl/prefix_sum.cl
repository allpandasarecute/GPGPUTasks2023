__kernel void prefix_sum_naive(__global const unsigned int *as, __global unsigned int *bs, const unsigned int offset,
                               const unsigned int n) {
    const unsigned int gid = get_global_id(0);

    if (gid >= offset)
        bs[gid] = as[gid] + as[gid - offset];
    else
        bs[gid] = as[gid];
}

__kernel void prefix_sum_next(__global unsigned int *as, const unsigned int off, const unsigned int n) {
    const unsigned int gid = get_global_id(0);

    if (gid >= n)
        return;

    if (!((gid + 1) % (off << 1))) {
        as[gid] = as[gid] + as[gid - off];
    }
}

__kernel void prefix_sum_down(__global unsigned int *as, const unsigned int off, const unsigned int n) {
    const unsigned int gid = get_global_id(0);

    if (gid >= n)
        return;

    unsigned int tmp;
    if (!((gid + 1) % (off << 1))) {
        tmp = as[gid];
        as[gid] = tmp + as[gid - off];
        as[gid - off] = tmp;
    }
}

__kernel void prefix_sum_normal(__global const unsigned int *as, __global unsigned int *bs, const unsigned int total,
                                const unsigned int n) {
    const unsigned int gid = get_global_id(0);

    if (gid >= n)
        return;

    bs[gid] = (gid + 1 == n) ? total : as[gid + 1] - total;
}
