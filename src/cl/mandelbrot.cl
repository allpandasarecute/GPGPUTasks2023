#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *res, unsigned int width, unsigned int height, float startX, float startY, float sizeX, float sizeY, unsigned int maxIter, const unsigned int smooth) {
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    float x0 = startX + (i + 0.5f) * sizeX / width;
    float y0 = startY + (j + 0.5f) * sizeY / height;

    float x = x0, y = y0;

    int iter;
    for (iter = 0; iter < maxIter; ++iter) {
        float prevX = x, prevY = y;

        x = x * x - y * y + x0;
        y = 2.0f * prevX * y + y0;

        if ((x * x + y * y) > threshold2)
            break;
    }

    float localRes = iter;

    if (iter && smooth != maxIter)
        localRes -= log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);

    res[j * width + i] = 1.0f * localRes / maxIter;
}
