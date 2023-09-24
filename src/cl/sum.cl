#define VALUES_PER_WORKITEM 32
#define WORKGROUP_SIZE 128

__kernel void sum_global_atomic(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n) {
    const unsigned int globalId = get_global_id(0);
    if (globalId >= n)
        return;

    atomic_add(sum, arr[globalId]);
}


__kernel void sum_loop_noncoalesced(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n) {
    const unsigned int globalId = get_global_id(0);

    unsigned int res = 0;

    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = globalId * VALUES_PER_WORKITEM + i;

        if (idx < n)
            res += arr[idx];
    }

    atomic_add(sum, res);
}

__kernel void sum_loop_coalesced(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n) {
    const unsigned int localId = get_local_id(0);
    const unsigned int groupId = get_group_id(0);
    const unsigned int localSize = get_local_size(0);

    int res = 0;

    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = groupId * localSize * VALUES_PER_WORKITEM + i * localSize + localId;
        if (idx < n)
            res += arr[idx];
    }

    atomic_add(sum, res);
}

__kernel void sum_local_mem(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n) {
    const unsigned int globalId = get_global_id(0);
    const unsigned int localId = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[localId] = globalId < n ? arr[globalId] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId != 0)
        return;

    unsigned int group_res = 0;

    for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i)
        group_res += buf[i];

    atomic_add(sum, group_res);
}

__kernel void sum_tree_atomic(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n) {
    const unsigned int globalId = get_global_id(0);
    const unsigned int localId = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[localId] = globalId < n ? arr[globalId]: 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int val = WORKGROUP_SIZE; val > 1; val /= 2) {
        if (2 * localId < val) {
            unsigned int a = buf[localId];
            unsigned int b = buf[localId + val / 2];
            buf[localId] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId != 0)
        return;

    atomic_add(sum, buf[0]);
}