#ifndef UTILS_H
#define UTILS_H

#include <immintrin.h> // Include intrinsic functions
#include <time.h>
#include "../search/search_ops.h"

int64_t get_current_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)(ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL);
}

bool should_stop_search(const SearchContext* ctx) {
    if (ctx->params.time_limit > 0) {
        int64_t elapsed = get_current_time() - ctx->start_time;
        if (elapsed >= ctx->params.time_limit) {
            return true;
        }
    }
    return ctx->stop_search;
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Perform population count on a __m256i vector of 64-bit integers.
 *
 * Counts the number of `1` bits in each 64-bit element of the input __m256i vector.
 *
 * @param v Input vector (__m256i) containing four 64-bit integers.
 * @return A __m256i vector where each 64-bit integer is the population count of the corresponding element in the input vector.
 */
static inline __m256i avx2_popcnt_epi64(__m256i v);

#ifdef __cplusplus
}
#endif

#endif // UTILS_H
