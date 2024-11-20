#ifndef EVAL_OPS_H
#define EVAL_OPS_H

#include <stdint.h>

#include "../bitboard/bitboard_ops.h"

// Game stages 
#define STAGE_OPENING 0
#define STAGE_MIDDLEGAME 1
#define STAGE_ENDGAME 2
#define STAGE_TABLEBASE 3

// Position conditions (add to eval_ops.h before structs)
#define POS_OUTPOST 0
#define POS_OPEN_DIAGONALS 1
#define POS_OPEN_FILE 2

extern const Bitboard FILE_MASKS[8];
extern const Bitboard KING_ZONE_MASKS[64];
extern const int MOBILITY_WEIGHTS[6];
extern const int PAWN_STRUCTURE_WEIGHTS[8][8];
extern const int KING_SAFETY_WEIGHTS[8];

// Pattern recognition structures 
typedef struct {
    Bitboard mask;
    Bitboard required;
    Bitboard forbidden;
    int16_t score_mg;
    int16_t score_eg;
} Pattern;

// Positional rules structure 
typedef struct {
    uint8_t condition;
    uint8_t piece_type;
    int8_t bonus_mg;
    int8_t bonus_eg;
} PositionalRule;

// Stage weights structure 
typedef struct {
    int16_t opening;
    int16_t middlegame;
    int16_t endgame;
} StageWeights;

// Evaluation terms structure 
typedef struct {
    int32_t material;
    int32_t psqt;
    int32_t mobility;
    int32_t pawn_structure;
    int32_t king_safety;
    int32_t piece_coordination;
    int32_t threats;
    int32_t space;
    int32_t initiative;
} EvalTerms;

// Architecture-specific SIMD structures
#ifdef USE_ARM_NEON
typedef struct {
    uint32x4_t pieces;
    uint32x4_t attacks;
    uint32x4_t control;
} EvalVector;

#else
typedef struct {
    __m256i pieces;
    __m256i attacks;
    __m256i control;
} EvalVector;
#endif

// Main evaluation context 
typedef struct {
    // Position representation
    Bitboard pieces[2][6];
    Bitboard occupied[2];
    Bitboard all_occupied;
    
    // Attack maps
    Bitboard attacks[2][6];
    Bitboard all_attacks[2];
    
    // Control maps
    Bitboard space_control[2];
    Bitboard center_control[2];
    
    // Pawn structure
    Bitboard pawn_shields[2];
    Bitboard passed_pawns[2];
    Bitboard pawn_chains[2];
    
    // Piece mobility
    uint8_t mobility_area[2];
    uint8_t piece_counts[2][6];
    
    // Game state
    int stage;
    int turn;
    uint8_t castling_rights;
    
    // Evaluation accumulators
    EvalTerms terms;
} EvalContext;

// Architecture-specific evaluation functions
#ifdef USE_ARM_NEON
void evaluate_material_neon(EvalContext* ctx);
void evaluate_mobility_neon(EvalContext* ctx);
void evaluate_king_safety_neon(EvalContext* ctx);
void evaluate_pawn_structure_neon(EvalContext* ctx);
void evaluate_piece_coordination_neon(EvalContext* ctx);

// NEON vector operation macros
#define EVAL_VEC_ADD vaddq_u32
#define EVAL_VEC_SUB vsubq_u32
#define EVAL_VEC_MUL vmulq_u32
#define EVAL_VEC_AND vandq_u32
#define EVAL_VEC_OR vorrq_u32

#else
void evaluate_material_simd(EvalContext* ctx);
void evaluate_mobility_simd(EvalContext* ctx);
void evaluate_king_safety_simd(EvalContext* ctx);
void evaluate_pawn_structure_simd(EvalContext* ctx);
void evaluate_piece_coordination_simd(EvalContext* ctx);

// AVX2 vector operation macros
#define EVAL_VEC_ADD _mm256_add_epi32
#define EVAL_VEC_SUB _mm256_sub_epi32
#define EVAL_VEC_MUL _mm256_mullo_epi32
#define EVAL_VEC_AND _mm256_and_si256
#define EVAL_VEC_OR _mm256_or_si256
#endif

// Architecture-independent functions 
int match_patterns(const EvalContext* ctx, const Pattern* patterns, int count);
int apply_positional_rules(const EvalContext* ctx, const PositionalRule* rules, int count);
int evaluate_position(EvalContext* ctx);

// Opening principles 
int evaluate_development(const EvalContext* ctx);
int evaluate_center_control(const EvalContext* ctx);
int evaluate_king_safety_early(const EvalContext* ctx);
int evaluate_piece_coordination(const EvalContext* ctx);

// Endgame knowledge 
int evaluate_endgame_patterns(const EvalContext* ctx);
int evaluate_winning_potential(const EvalContext* ctx);
int evaluate_fortress_detection(const EvalContext* ctx);

// Utility functions 
int calculate_game_stage(const EvalContext* ctx);
void update_attack_maps(EvalContext* ctx);
void update_pawn_structure(EvalContext* ctx);

// Vector operation helpers
#ifdef USE_ARM_NEON
static inline int32_t horizontal_sum_neon(uint32x4_t v) {
    uint32x2_t sum = vadd_u32(vget_low_u32(v), vget_high_u32(v));
    return vget_lane_u32(vpadd_u32(sum, sum), 0);
}
#endif

// Cleanup macros to avoid naming conflicts
#undef EVAL_VEC_ADD
#undef EVAL_VEC_SUB
#undef EVAL_VEC_MUL
#undef EVAL_VEC_AND
#undef EVAL_VEC_OR

#endif // EVAL_OPS_H