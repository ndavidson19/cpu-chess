#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h> // Include intrinsic functions
#include "../search/search_ops.h"

int64_t get_current_time();
bool should_stop_search(const SearchContext* ctx);
int min(int a, int b);

// Game state checking functions
bool is_draw(const Position* pos);
bool is_insufficient_material(const Position* pos);
bool is_stalemate(const Position* pos);
bool is_checkmate(const Position* pos);
bool is_square_attacked(const Position* pos, int square, int attacker_color);


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
static inline __m256i avx2_popcnt_epi64(__m256i v) {
    __m256i count = _mm256_setzero_si256();

    for (int i = 0; i < 4; ++i) {
        uint64_t val = _mm256_extract_epi64(v, i);
        int cnt = __builtin_popcountll(val);
        count = _mm256_insert_epi64(count, cnt, i);
    }

    return count;
}

#ifdef __cplusplus
}
#endif

#endif // UTILS_H
