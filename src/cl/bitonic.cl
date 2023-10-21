__kernel void bitonic(__global float *as, const unsigned int window, const unsigned int arrow, const unsigned int n) {
    unsigned int id = get_global_id(0);

    if ((id << 1) >= n)
        return;

    unsigned int begin = ((id << 1) / window);
    const int dec = begin & 1;

    begin = begin * window;
    id = begin + (id - id / arrow * arrow) + ((id << 1) - begin) / (arrow << 1) * (arrow << 1);

    const float max = as[id] > as[id + arrow] ? as[id] : as[id + arrow];
    const float min = as[id] < as[id + arrow] ? as[id] : as[id + arrow];

    as[id] = (dec) ? max : min;
    as[id + arrow] = (dec) ? min : max;
}
