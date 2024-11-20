#ifndef PATTERNS_H
#define PATTERNS_H

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include "../bitboard/bitboard_ops.h"

#ifdef USE_ARM_NEON
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

// Constants (unchanged)
#define NO_PIECE -1
#define WHITE 0
#define BLACK 1

#define LIGHT_SQUARES 0x55AA55AA55AA55AAULL
#define DARK_SQUARES  0xAA55AA55AA55AA55ULL

// File and rank masks 
#define FILE_A 0x0101010101010101ULL
#define FILE_B 0x0202020202020202ULL
#define FILE_C 0x0404040404040404ULL
#define FILE_D 0x0808080808080808ULL
#define FILE_E 0x1010101010101010ULL
#define FILE_F 0x2020202020202020ULL
#define FILE_G 0x4040404040404040ULL
#define FILE_H 0x8080808080808080ULL

#define RANK_1 0x00000000000000FFULL
#define RANK_2 0x000000000000FF00ULL
#define RANK_3 0x0000000000FF0000ULL
#define RANK_4 0x00000000FF000000ULL
#define RANK_5 0x000000FF00000000ULL
#define RANK_6 0x0000FF0000000000ULL
#define RANK_7 0x00FF000000000000ULL
#define RANK_8 0xFF00000000000000ULL

// Memory alignment
#define ALIGN_32 __attribute__((aligned(32)))

// Position representation
typedef struct Position {
    uint64_t pieces[2][6] ALIGN_32;
    uint64_t occupied[2] ALIGN_32;
    uint64_t occupied_total;
    int side_to_move;           // 0 for white, 1 for black
    uint8_t castling_rights;    // Bitmask for castling rights
    int en_passant_square;      // Square index for en passant (-1 if none)
    int halfmove_clock;         // Number of halfmoves since last capture or pawn move
    int fullmove_number;        // Number of full moves
} Position;



// Architecture-specific pattern structures
#ifdef USE_ARM_NEON
typedef struct ALIGN_32 {
    uint64x2_t key_squares[2];    // 4x uint64 squares for pattern matching
    uint64x2_t piece_masks[2];    // 4x uint64 required piece positions
    int32x4_t scores;            // 4x int32 scores
} SIMDPattern;

#else
typedef struct ALIGN_32 {
    uint64_t key_squares[4] ALIGN_32;
    uint64_t piece_masks[4] ALIGN_32;
    int32_t scores[4] ALIGN_32;
} SIMDPattern;


#endif

// Endgame structures (unchanged)
typedef struct {
    uint32_t position_hash;
    uint8_t best_move;
    int8_t eval;
    uint8_t dtm;
    uint8_t flags;
} EndgameEntry;

typedef struct {
    uint32_t material_key;
    uint32_t position_count;
    const uint8_t* compressed_data;
    uint32_t chunk_size;
    const int8_t* dtm_table;
} EndgameDefinition;


// Architecture-specific function declarations
#ifdef USE_ARM_NEON
uint64x2_t load_position_neon(const Position* pos);
int32x4_t horizontal_sum_neon(int32x4_t v);
uint64x2_t calculate_doubled_pawns_neon(uint64_t white_pawns, uint64_t black_pawns);
uint64x2_t calculate_isolated_pawns_neon(uint64_t white_pawns, uint64_t black_pawns);
int32x4_t calculate_control_scores_neon(uint64x2_t control);
int32x4_t calculate_structure_scores_neon(uint64x2_t structure);

#else
__m256i load_position_simd(const Position* pos);
__m128i horizontal_sum_128(__m128i v);
__m256i calculate_doubled_pawns_simd(__m256i white_pawns, __m256i black_pawns);
__m256i calculate_isolated_pawns_simd(__m256i white_pawns, __m256i black_pawns);
__m128i calculate_control_scores_simd(__m256i control);
__m128i calculate_structure_scores_simd(__m256i structure);
#endif

// Architecture-independent functions (unchanged)
int evaluate_london_position(const Position* pos);
int evaluate_caro_kann_position(const Position* pos);
void load_endgame_solver(int material_key);
EndgameEntry* probe_endgame_table(const Position* pos);

// Utility functions (unchanged)
void init_pattern_tables(void);
void cleanup_endgame_solver(void);
uint16_t calculate_material_key(const Position* pos);
bool is_endgame_material(uint16_t material_key);
void init_development_masks(void);
void init_coordination_masks(void);
void init_structure_masks(void);
int calculate_dtm(int index, uint16_t material_key);

// Memory management (unchanged)
void* aligned_alloc(size_t alignment, size_t size);
void aligned_free(void* ptr);

// Architecture-specific helper functions
#ifdef USE_ARM_NEON
static int binary_search_neon(const uint32_t* array, uint32_t target, size_t size);
static uint32_t compress_position_neon(const Position* pos);
static uint64x2_t evaluate_development_neon(const Position* pos);
static uint64x2_t evaluate_pawn_structure_neon(const Position* pos);
static uint64x2_t evaluate_coordination_neon(const Position* pos);

// NEON pattern matching helpers
static inline uint64x2_t match_pattern_neon(const uint64x2_t position, const SIMDPattern* pattern);
static inline int32x4_t calculate_scores_neon(const uint64x2_t matches, const int32x4_t scores);

#else
static int binary_search_simd(const uint32_t* array, uint32_t target, size_t size);
static uint32_t compress_position(const Position* pos);
static __m256i evaluate_development_simd(const Position* pos);
static __m256i evaluate_coordination_simd(const Position* pos);

// SIMD pattern matching helpers
static inline __m256i match_pattern_simd(const __m256i position, const SIMDPattern* pattern);
static inline __m128i calculate_scores_simd(const __m256i matches, const __m128i scores);
#endif

// Common helper functions (unchanged)
static inline void decompress_tablebase_chunk(
    const uint8_t* compressed_data,
    uint32_t* positions,
    uint8_t* moves,
    int8_t* scores,
    uint32_t position_count
);

static inline const EndgameDefinition* find_endgame_definition(uint16_t material_key);
static inline uint16_t compress_piece_placement(const Position* pos);

// External pattern tables
#ifdef USE_ARM_NEON
extern const SIMDPattern LONDON_PATTERNS_NEON[] ALIGN_32;
extern const SIMDPattern CARO_KANN_PATTERNS_NEON[] ALIGN_32;
extern const SIMDPattern ENDGAME_PATTERNS_NEON[] ALIGN_32;
#else
extern const SIMDPattern LONDON_PATTERNS[] ALIGN_32;
extern const SIMDPattern CARO_KANN_PATTERNS[] ALIGN_32;
extern const SIMDPattern ENDGAME_PATTERNS[] ALIGN_32;
#endif

extern const EndgameDefinition ENDGAME_DEFS[];

// Architecture-specific macros for vector operations
#ifdef USE_ARM_NEON
#define VEC_ADD_64 vaddq_u64
#define VEC_SUB_64 vsubq_u64
#define VEC_AND_64 vandq_u64
#define VEC_OR_64  vorrq_u64
#define VEC_XOR_64 veorq_u64
#else
#define VEC_ADD_64 _mm256_add_epi64
#define VEC_SUB_64 _mm256_sub_epi64
#define VEC_AND_64 _mm256_and_si256
#define VEC_OR_64  _mm256_or_si256
#define VEC_XOR_64 _mm256_xor_si256
#endif

#endif // PATTERNS_H