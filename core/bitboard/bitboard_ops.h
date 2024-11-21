#ifndef BITBOARD_OPS_H
#define BITBOARD_OPS_H

#include <stdint.h>
#include "../common/types.h"

// Architecture-specific includes
#ifdef USE_ARM_NEON
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

// Type definitions
typedef uint64_t Bitboard;

// Board representation constants 
#define FILE_A 0x0101010101010101ULL
#define FILE_B 0x0202020202020202ULL
#define FILE_C 0x0404040404040404ULL
#define FILE_D 0x0808080808080808ULL
#define FILE_E 0x1010101010101010ULL
#define FILE_F 0x2020202020202020ULL
#define FILE_G 0x4040404040404040ULL
#define FILE_H 0x8080808080808080ULL

// Magic Bitboard constants 
#define ROOK_MAGIC_COUNT 64
#define BISHOP_MAGIC_COUNT 64
#define ROOK_BITS 12
#define BISHOP_BITS 9
#define ROOK_TABLE_SIZE (1ULL << ROOK_BITS)
#define BISHOP_TABLE_SIZE (1ULL << BISHOP_BITS)

// Structure for magic bitboard data 
typedef struct {
    Bitboard mask;
    Bitboard magic;
    int shift;
    Bitboard* attacks;
} MagicEntry;

// Architecture-specific SIMD structures
#ifdef USE_ARM_NEON
typedef struct {
    uint64x2_t positions;
    uint64x2_t weights;
    uint64x2_t psqt_values;
} EvalBatch;

// NEON vector operations
static inline uint64x2_t neon_horizontal_add(uint64x2_t v) {
    return vpaddq_u64(v, v);
}

#else
typedef struct {
    __m256i positions;
    __m256i weights;
    __m256i psqt_values;
} EvalBatch;

// x86 SIMD utilities
static inline __m256i horizontal_sum(__m256i x) {
    __m256i sum1 = _mm256_hadd_epi32(x, x);
    __m256i sum2 = _mm256_hadd_epi32(sum1, sum1);
    __m128i sum3 = _mm256_extracti128_si256(sum2, 1);
    __m128i sum4 = _mm256_castsi256_si128(sum2);
    __m128i sum5 = _mm_add_epi32(sum3, sum4);
    return _mm256_castsi128_si256(sum5);
}
#endif

// Global lookup tables 
extern MagicEntry ROOK_MAGICS[64];
extern MagicEntry BISHOP_MAGICS[64];
extern Bitboard PAWN_ATTACKS_TABLE[2][64];
extern Bitboard KNIGHT_ATTACKS_TABLE[64];
extern Bitboard KING_ATTACKS_TABLE[64];

// Core function declarations 
void init_attack_tables(void);
void init_magic_bitboards(void);
void cleanup_magic_bitboards(void);

// Attack generation functions 
Bitboard get_pawn_attacks(int square, int color);
Bitboard get_knight_attacks(int square);
Bitboard get_king_attacks(int square);
Bitboard get_rook_attacks(int square, Bitboard occupancy);
Bitboard get_bishop_attacks(int square, Bitboard occupancy);
Bitboard get_queen_attacks(int square, Bitboard occupancy);

// Architecture-specific evaluation functions
#ifdef USE_ARM_NEON
int evaluate_position_neon(const int* positions, const int* weights, const int* psqt, int length);
int evaluate_pawns_neon(Bitboard wp, Bitboard bp, const int* weights);
#else
int evaluate_position_simd(const int* positions, const int* weights, const int* psqt, int length);
int evaluate_pawns_simd(Bitboard wp, Bitboard bp, const int* weights);
#endif

// Helper functions 
static inline Bitboard mask_rook_attacks(int square);
static inline Bitboard mask_bishop_attacks(int square);
static inline Bitboard generate_rook_attacks(int square, Bitboard occupancy);
static inline Bitboard generate_bishop_attacks(int square, Bitboard occupancy);

// Utility functions 
static inline Bitboard pop_lsb(Bitboard* bb) {
    Bitboard lsb = *bb & -(*bb);
    *bb &= *bb - 1;
    return lsb;
}

static inline int get_lsb_index(Bitboard bb) {
    return __builtin_ctzll(bb);
}

static inline int get_msb_index(Bitboard bb) {
    return 63 - __builtin_clzll(bb);
}

// Bit manipulation utilities 
static inline int count_bits(Bitboard bb) {
    return __builtin_popcountll(bb);
}

static inline Bitboard shift_north(Bitboard bb) {
    return bb << 8;
}

static inline Bitboard shift_south(Bitboard bb) {
    return bb >> 8;
}

static inline Bitboard shift_east(Bitboard bb) {
    return (bb << 1) & ~FILE_A;
}

static inline Bitboard shift_west(Bitboard bb) {
    return (bb >> 1) & ~FILE_H;
}

// Memory alignment helpers 
static inline void* aligned_malloc(size_t size, size_t alignment) {
    void* p;
    if (posix_memalign(&p, alignment, size) != 0) {
        return NULL;
    }
    return p;
}

#define aligned_free free

#endif // BITBOARD_OPS_H