#define WORKGROUP_SIZE 128

__kernel void reduce(__global const unsigned int *as, __global unsigned int *bs, unsigned int n, unsigned int i) {
    const unsigned int global_id = get_global_id(0);

    if (global_id >= n)
        return;

    if (((global_id + 1) >> i) & 1) {
        bs[global_id] += as[((global_id + 1) >> i) - 1];
    }
}

__kernel void prefix_sum(__global const unsigned int *as, __global unsigned int *res, unsigned int n) {
    const unsigned int i = get_global_id(0);

    if (i >= n)
        return;

    res[i] = as[i * 2] + as[i * 2 + 1];
}

__kernel void radix_counters(__global const unsigned int *as, __global unsigned int *counters, unsigned int i,
                             unsigned int n) {
    const unsigned int global_id = get_global_id(0);

    if (global_id * WORKGROUP_SIZE >= n)
        return;

    const unsigned int segment_begin = global_id * WORKGROUP_SIZE;
    const unsigned int segment_end = (global_id + 1) * WORKGROUP_SIZE;
    unsigned int counter = 0;

    for (int j = segment_begin; j < segment_end; j++)
        counter += (as[j] >> i) & 1;

    counters[global_id] = counter;
}

__kernel void radix_sort(__global const unsigned int *counters, __global unsigned int *as, __global unsigned int *bs,
                         unsigned int offset, unsigned int n) {
    const unsigned int i = get_global_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int local_id = get_local_id(0);

    __local unsigned int counter_sum;
    __local unsigned int counter;
    __local unsigned int tmp[WORKGROUP_SIZE];

    if (i >= n)
        return;

    const unsigned int zero_all_count = n - counters[n / WORKGROUP_SIZE - 1];
    tmp[local_id] = as[i];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i == group_id * WORKGROUP_SIZE) {
        if (group_id == 0)
            counter_sum = 0;
        else
            counter_sum = counters[group_id - 1];

        unsigned int tmp_counter = 0;
        unsigned int tmp_copy[WORKGROUP_SIZE];

        for (unsigned int ik = 0; ik < WORKGROUP_SIZE; ++ik) {
            tmp_copy[ik] = tmp[ik];
            tmp_counter += ((tmp_copy[ik] >> offset) & 1);
        }

        unsigned int ik = 0, j = WORKGROUP_SIZE - tmp_counter;
        for (unsigned int k = 0; k < WORKGROUP_SIZE; ++k)
            if ((tmp_copy[k] >> offset) & 1)
                tmp[j++] = tmp_copy[k];
            else
                tmp[ik++] = tmp_copy[k];

        counter = tmp_counter;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int new_pos = local_id < (WORKGROUP_SIZE - counter)
                                         ? group_id * WORKGROUP_SIZE - counter_sum + local_id
                                         : zero_all_count + counter_sum + (local_id - WORKGROUP_SIZE + counter);

    barrier(CLK_GLOBAL_MEM_FENCE);

    bs[new_pos] = tmp[local_id];
}
