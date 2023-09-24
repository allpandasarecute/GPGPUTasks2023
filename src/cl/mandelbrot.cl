#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* results, unsigned int width, unsigned int height,
                         float startX, float startY, float sizeX, float sizeY,
                         unsigned int maxIter, int smooth, int antiAlias) {
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен

    unsigned int idX = get_global_id(0) % width;
    unsigned int idY = get_global_id(0) / width;

    if (idY >= height)
        return;

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    float x0, y0, x, y;

    float res, localRes = 0.0f;
    float step = 1.0f / ((float)(antiAlias) + 1.0f);
    
    int iter;
    for (int j = 0; j < antiAlias; ++j) {
        for (int i = 0; i < antiAlias; ++i) {
            x = (x0 = startX + (idX + step * ((float)i + 1.0f)) * sizeX / width);
            y = (y0 = startY + (idY + step * ((float)j + 1.0f)) * sizeY / height);

            for (iter = 0; iter < maxIter; ++iter) {
                float prevX = x;
                x = x * x - y * y + x0;
                y = 2.0f * prevX * y + y0;
                if ((x * x + y * y) > threshold2) {
                    break;
                }
            }
            res = iter;
            if (smooth && iter != maxIter)
                res = res - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
            localRes += 1.0f * res / (float)maxIter;
        }
    }
    results[idY * width + idX] = localRes / (float)(antiAlias * antiAlias);
}
