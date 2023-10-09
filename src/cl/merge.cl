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

__kernel void merge(const __global float *a, __global float *c, unsigned int len) {
    int i = get_global_id(0);
    int block = i / (2 * len);

    int lower_bound;
    if ((i / len) % 2)
        lower_bound = i - len + binary_search(a + block * 2 * len, len, a[i], false);
    else
        lower_bound = i + binary_search(a + (block * 2 + 1) * len, len, a[i], true);

    c[lower_bound] = a[i];
}
