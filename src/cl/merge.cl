inline bool comp(float a, float b, bool eq) {
    return eq ? a <= b : a < b;
}

int binary_search(const __global float *a, int n, float e, bool eq) {
    int l = 0, r = n;
    int m;

    while (r - l > 1) {
        m = (l + r) / 2;

        if (comp(a[m], e, eq))
            l = m;
        else
            r = m;
    }

    return l;
}

__kernel void merge(const __global float *as, __global float *bs, const int len) {
    const int i = get_global_id(0);

    int left = (i % (2 * len)) < len;

    int lower_bound = binary_search(as + (i / (2 * len)) * (2 * len) + left * len, len, as[i], left);

    bs[i - len * !left + lower_bound] = as[i];
}
