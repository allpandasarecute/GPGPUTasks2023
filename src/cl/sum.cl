#define WORKGROUP_SIZE 32
#define VALUES_PER_WORKITEM 128

__kernel void sum_global_atomic(__global const unsigned int *arr, volatile __global unsigned int *res, unsigned int n) {
    const unsigned int index = get_global_id(0);

	if (index >= n)
		return;

	atomic_add(res, arr[index]);
}

__kernel void sum_loop_noncoalesced(__global const unsigned int *arr, volatile  __global unsigned int *res, unsigned int n) {
	const unsigned int globalId = get_global_id(0);

	unsigned int localRes = 0;
	for (unsigned int i = 0; i < VALUES_PER_WORKITEM; ++i) {
		int idx = globalId * VALUES_PER_WORKITEM + i;

		if (idx < n)
			localRes += arr[idx];
	}

	atomic_add(res, localRes);
}

__kernel void sum_loop_coalesced(__global const unsigned int *arr, volatile  __global unsigned int *res, unsigned int n) {
	const unsigned int localId = get_local_id(0);
	const unsigned int groupId = get_group_id(0);
	const unsigned int localSize = get_local_size(0);

	unsigned int localRes = 0;
	for (unsigned int i = 0; i < VALUES_PER_WORKITEM; ++i) {
		int idx = groupId * localSize * VALUES_PER_WORKITEM + i * localSize + localId;

		if (idx < n)
			localRes += arr[idx];
	}

	atomic_add(res, localRes);
}

__kernel void sum_local_mem(__global const unsigned int *arr, __global unsigned int *res, unsigned int n) {
    const unsigned int globalId = get_global_id(0);
    const unsigned int localId = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    if (globalId < n)
        buf[localId] = arr[globalId];
    else
        buf[localId] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) {
        unsigned int localRes = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i)
            localRes += buf[i];
        atomic_add(res, localRes);
    }
}

__kernel void sum_tree_atomic(__global const unsigned int *arr, __global unsigned int *res, unsigned int n) {
    const unsigned int globalId = get_global_id(0);
    const unsigned int localId = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[localId] = globalId < n ? arr[globalId]: 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * localId < nValues) {
            unsigned int a = buf[localId];
            unsigned int b = buf[localId + nValues / 2];
            buf[localId] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0)
        atomic_add(res, buf[0]);
}
