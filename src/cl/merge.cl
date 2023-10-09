inline bool comp(float a, float b, bool eq) {
    return eq ? a <= b : a < b;
}

int binary_search(const __global float *a, int n, float val, bool eq) {
    int l = -1, r = n;
    int m;

    while (r - l > 1) {
        m = (l + r) / 2;

        if (comp(a[m], val, eq))
            l = m;
        else
            r = m;
    }

    return l + 1;
}

__kernel void merge(const __global float *as, __global float *bs, unsigned int len) {
    const unsigned int i = get_global_id(0);
    const unsigned int block = i / (2 * len);

    unsigned int lower_bound;
    if ((i / len) % 2)
        lower_bound = i - len + binary_search(as + block * 2 * len, len, as[i], false);
    else
        lower_bound = i + binary_search(as + (block * 2 + 1) * len, len, as[i], true);

    bs[lower_bound] = as[i];
}
