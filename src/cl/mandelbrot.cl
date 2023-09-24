#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *res, unsigned int width, unsigned int height, float startX, float startY, float sizeX, float sizeY, unsigned int maxIter, int smooth, const unsigned int antiAlias) {
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    unsigned int idX = get_global_id(0);
    unsigned int idY = get_global_id(1);

    float x0, y0, x, y;
    float step = 1.0f / ((float)antiAlias + 1.0f);

    float localRes;

    int iter;
    for (int i = 0; i < antiAlias; ++i) {
        for (int j = 0; j < antiAlias; ++j) {

            x0 = startX + (idX + step * ((float)i + 1.0f)) * sizeX / width;
            y0 = startY + (idY + step * ((float)j + 1.0f)) * sizeY / height;
            x = x0;
            y = y0;
            for (iter = 0; iter < maxIter; ++iter) {
                float prevX = x;

                x = x * x - y * y + x0;
                y = 2.0f * prevX * y + y0;

                if ((x * x + y * y) > threshold2)
                    break;
            }

            localRes = (float)iter;

    //    if (iter && smooth != maxIter)
    //        localRes -= log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
        }
    }
    res[idY * width + idX] = 1.0f * localRes / maxIter / (float)(antiAlias * antiAlias);
}
