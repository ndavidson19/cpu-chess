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

// Helper function to get minimum of two integers
int min(int a, int b) {
    return a < b ? a : b;
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


bool is_insufficient_material(const Position* pos) {
    // Count total material excluding kings
    int white_material = 0;
    int black_material = 0;
    
    // Count all pieces except pawns and kings
    for (int piece = KNIGHT; piece <= QUEEN; piece++) {
        white_material += __builtin_popcountll(pos->pieces[WHITE][piece]);
        black_material += __builtin_popcountll(pos->pieces[BLACK][piece]);
    }
    
    // Count pawns
    int white_pawns = __builtin_popcountll(pos->pieces[WHITE][PAWN]);
    int black_pawns = __builtin_popcountll(pos->pieces[BLACK][PAWN]);
    
    if (white_pawns > 0 || black_pawns > 0) return false;
    if (white_material == 0 && black_material == 0) return true;
    
    // King + minor piece vs King
    if ((white_material == 1 && black_material == 0) ||
        (white_material == 0 && black_material == 1)) {
        Bitboard minor_pieces = pos->pieces[WHITE][KNIGHT] | pos->pieces[WHITE][BISHOP] |
                               pos->pieces[BLACK][KNIGHT] | pos->pieces[BLACK][BISHOP];
        if (__builtin_popcountll(minor_pieces) == 1) return true;
    }
    
    // King + Bishop vs King + Bishop (same colored squares)
    if (white_material == 1 && black_material == 1) {
        Bitboard white_bishops = pos->pieces[WHITE][BISHOP];
        Bitboard black_bishops = pos->pieces[BLACK][BISHOP];
        if (white_bishops && black_bishops) {
            Bitboard light_squares = 0x55AA55AA55AA55AAULL;
            bool white_on_light = white_bishops & light_squares;
            bool black_on_light = black_bishops & light_squares;
            if (white_on_light == black_on_light) return true;
        }
    }
    
    return false;
}

bool is_stalemate(const Position* pos) {
    if (in_check(pos)) return false;
    
    SearchMove moves[256];
    int move_count = generate_moves(pos, moves);
    
    // Try each move to see if it's legal
    for (int i = 0; i < move_count; i++) {
        Position temp_pos = *pos;
        make_move(&temp_pos, moves[i].move);
        if (!in_check(&temp_pos)) {
            return false;  // Found at least one legal move
        }
    }
    
    return true;  // No legal moves
}

bool is_draw(const Position* pos) {
    // Check for insufficient material
    if (is_insufficient_material(pos)) return true;
    
    // Check for threefold repetition
    if (pos->repetition_count >= 3) return true;
    
    // Check for fifty move rule
    if (pos->fifty_move_count >= 100) return true;
    
    // Check for stalemate
    if (is_stalemate(pos)) return true;
    
    return false;
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
