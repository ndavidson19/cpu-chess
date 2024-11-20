#include "eval_ops.h"
#include <string.h>
#include <stdbool.h>

// Initialize constants
const Bitboard FILE_MASKS[8] = {
    FILE_A, FILE_B, FILE_C, FILE_D,
    FILE_E, FILE_F, FILE_G, FILE_H
};

const Bitboard KING_ZONE_MASKS[64] = {
    0x0000000000000302ULL, // a1
    0x0000000000000705ULL, // b1
    0x0000000000000E0AULL, // c1
    0x0000000000001C14ULL, // d1
    0x0000000000003828ULL, // e1
    0x0000000000007050ULL, // f1
    0x000000000000E0A0ULL, // g1
    0x000000000000C040ULL, // h1
    // ... (complete the initialization)
};

const int MOBILITY_WEIGHTS[6] = {
    0,    // Pawn
    4,    // Knight
    5,    // Bishop
    4,    // Rook
    10,   // Queen
    0     // King
};

const int PAWN_STRUCTURE_WEIGHTS[8][8] = {
    {10, -10, 5, -5, 8, -8, 0, 0},  // Doubled
    {-20, 15, -10, 5, -8, 4, 0, 0}, // Isolated
    {15, -12, 8, -4, 6, -3, 0, 0},  // Protected
    {12, -8, 6, -3, 4, -2, 0, 0},   // Connected
    {25, -20, 15, -10, 8, -4, 0, 0}, // Passed
    {-15, 12, -8, 4, -6, 3, 0, 0},  // Backward
    {5, -4, 3, -2, 2, -1, 0, 0},    // Count
    {0, 0, 0, 0, 0, 0, 0, 0}        // Reserved
};

const int KING_SAFETY_WEIGHTS[8] = {
    0,    // No attackers
    -10,  // One attacker
    -25,  // Two attackers
    -45,  // Three attackers
    -70,  // Four attackers
    -100, // Five attackers
    -135, // Six attackers
    -175  // Seven or more attackers
};

// Opening principles encoded as patterns
static const Pattern OPENING_PATTERNS[] = {
    // Center control patterns
    {
        .mask = 0x0000001818000000ULL,     // e4,d4,e5,d5
        .required = 0x0000001818000000ULL,  // Pawns/pieces on these squares
        .forbidden = 0,
        .score_mg = 50,
        .score_eg = 20
    },
    // Development patterns
    {
        .mask = 0x00000000000000C7ULL,     // Knights and bishops developed
        .required = 0x0000000000000042ULL,  // Knights out
        .forbidden = 0,
        .score_mg = 30,
        .score_eg = 10
    },
    // King safety patterns
    {
        .mask = 0x000000000000000EULL,     // Kingside castling structure
        .required = 0x000000000000000EULL,  // King and pawns in place
        .forbidden = 0,
        .score_mg = 60,
        .score_eg = 20
    }
    // Add more patterns...
};

// Positional rules for consistent play
static const PositionalRule POSITIONAL_RULES[] = {
    // Knights
    {
        .condition = POS_OUTPOST,
        .piece_type = KNIGHT,
        .bonus_mg = 20,
        .bonus_eg = 15
    },
    // Bishops
    {
        .condition = POS_OPEN_DIAGONALS,
        .piece_type = BISHOP,
        .bonus_mg = 15,
        .bonus_eg = 15
    },
    // Rooks
    {
        .condition = POS_OPEN_FILE,
        .piece_type = ROOK,
        .bonus_mg = 25,
        .bonus_eg = 20
    }
    // Add more rules...
};


// Helper functions for pawn structure evaluation
static inline bool is_passed_pawn(Bitboard pawn, Bitboard enemy_pawns) {
    int square = get_lsb_index(pawn);
    int file = square % 8;
    Bitboard file_mask = FILE_MASKS[file];
    Bitboard adjacent_files = 0;
    if (file > 0) adjacent_files |= FILE_MASKS[file - 1];
    if (file < 7) adjacent_files |= FILE_MASKS[file + 1];
    return !(enemy_pawns & (file_mask | adjacent_files));
}

static inline bool is_isolated_pawn(Bitboard pawn, Bitboard friendly_pawns) {
    int square = get_lsb_index(pawn);
    int file = square % 8;
    Bitboard adjacent_files = 0;
    if (file > 0) adjacent_files |= FILE_MASKS[file - 1];
    if (file < 7) adjacent_files |= FILE_MASKS[file + 1];
    return !(friendly_pawns & adjacent_files);
}

static inline bool is_backward_pawn(Bitboard pawn, Bitboard friendly_pawns) {
    int square = get_lsb_index(pawn);
    int rank = square / 8;
    int file = square % 8;
    Bitboard support_mask = 0;
    if (file > 0) support_mask |= FILE_MASKS[file - 1];
    if (file < 7) support_mask |= FILE_MASKS[file + 1];
    support_mask &= ((1ULL << (rank * 8)) - 1);
    return !(friendly_pawns & support_mask);
}

static inline bool is_protected_pawn(Bitboard pawn, Bitboard friendly_pawns) {
    int square = get_lsb_index(pawn);
    return friendly_pawns & get_pawn_attacks(square, 1);
}

static inline bool is_connected_pawn(Bitboard pawn, Bitboard friendly_pawns) {
    int square = get_lsb_index(pawn);
    int file = square % 8;
    Bitboard adjacent_files = 0;
    if (file > 0) adjacent_files |= FILE_MASKS[file - 1];
    if (file < 7) adjacent_files |= FILE_MASKS[file + 1];
    return friendly_pawns & adjacent_files & (1ULL << (square / 8));
}

static inline bool is_doubled_pawn(Bitboard pawn) {
    int square = get_lsb_index(pawn);
    int file = square % 8;
    return __builtin_popcountll(pawn & FILE_MASKS[file]) > 1;
}

void evaluate_king_safety_simd(EvalContext* ctx) {
    __m256i total_safety = _mm256_setzero_si256();
    
    for (int color = 0; color < 2; color++) {
        Bitboard king = ctx->pieces[color][KING];
        int king_square = get_lsb_index(king);
        
        // Get king zone (king + adjacent squares)
        Bitboard king_zone = KING_ZONE_MASKS[king_square];
        
        // Count attackers and attack weight
        __m256i attack_counts = _mm256_setzero_si256();
        for (int piece = 0; piece < 6; piece++) {
            Bitboard attackers = ctx->attacks[!color][piece] & king_zone;
            attack_counts = _mm256_insert_epi32(attack_counts, 
                __builtin_popcountll(attackers), piece);
        }
        
        // Evaluate pawn shield
        Bitboard pawn_shield = ctx->pawn_shields[color];
        __m256i shield_score = _mm256_set1_epi32(
            __builtin_popcountll(pawn_shield & king_zone) * 10);
        
        // Calculate attack penalty
        __m256i weights = _mm256_load_si256((__m256i*)KING_SAFETY_WEIGHTS);
        __m256i attack_score = _mm256_mullo_epi32(attack_counts, weights);
        
        // Combine shield bonus and attack penalty
        __m256i color_score = _mm256_sub_epi32(shield_score, attack_score);
        
        // Add or subtract based on color
        if (color == 0) {
            total_safety = _mm256_add_epi32(total_safety, color_score);
        } else {
            total_safety = _mm256_sub_epi32(total_safety, color_score);
        }
    }
    
    ctx->terms.king_safety = _mm256_extract_epi32(horizontal_sum(total_safety), 0);
}

// SIMD-optimized evaluation of piece mobility
void evaluate_mobility_simd(EvalContext* ctx) {
    __m256i total_mobility = _mm256_setzero_si256();
    
    for (int color = 0; color < 2; color++) {
        for (int piece = 0; piece < 6; piece++) {
            Bitboard pieces = ctx->pieces[color][piece];
            __m256i mobility_scores = _mm256_setzero_si256();
            
            while (pieces) {
                int square = get_lsb_index(pieces);
                Bitboard attacks = ctx->attacks[color][piece];
                
                // Count available moves excluding friendly pieces
                Bitboard moves = attacks & ~ctx->occupied[color];
                int move_count = __builtin_popcountll(moves);
                
                // Pack move counts into SIMD register
                __m256i counts = _mm256_set1_epi32(move_count);
                __m256i weights = _mm256_set1_epi32(MOBILITY_WEIGHTS[piece]);
                mobility_scores = _mm256_add_epi32(
                    mobility_scores,
                    _mm256_mullo_epi32(counts, weights)
                );
                
                pieces &= pieces - 1;
            }
            
            // Accumulate scores with color sign
            if (color == 0) {
                total_mobility = _mm256_add_epi32(total_mobility, mobility_scores);
            } else {
                total_mobility = _mm256_sub_epi32(total_mobility, mobility_scores);
            }
        }
    }
    
    ctx->terms.mobility = _mm256_extract_epi32(horizontal_sum(total_mobility), 0);
}

// Sophisticated pawn structure evaluation
void evaluate_pawn_structure_simd(EvalContext* ctx) {
    __m256i pawn_scores = _mm256_setzero_si256();
    
    for (int color = 0; color < 2; color++) {
        Bitboard pawns = ctx->pieces[color][PAWN];
        Bitboard enemy_pawns = ctx->pieces[!color][PAWN];
        
        // Evaluate each pawn file
        for (int file = 0; file < 8; file++) {
            Bitboard file_mask = FILE_MASKS[file];
            Bitboard file_pawns = pawns & file_mask;
            
            // Pack various pawn features into SIMD register
            __m256i features = _mm256_set_epi32(
                __builtin_popcountll(file_pawns),           // Count in file
                is_passed_pawn(file_pawns, enemy_pawns),    // Passed pawn
                is_isolated_pawn(file_pawns, pawns),        // Isolated pawn
                is_backward_pawn(file_pawns, pawns),        // Backward pawn
                is_protected_pawn(file_pawns, pawns),       // Protected pawn
                is_connected_pawn(file_pawns, pawns),       // Connected pawn
                is_doubled_pawn(file_pawns),                // Doubled pawn
                0
            );
            
            // Load appropriate weights and multiply
            __m256i weights = _mm256_load_si256((__m256i*)&PAWN_STRUCTURE_WEIGHTS[file]);
            __m256i file_scores = _mm256_mullo_epi32(features, weights);
            pawn_scores = _mm256_add_epi32(pawn_scores, file_scores);
        }
        
        // Adjust scores based on color
        if (color == 1) {
            pawn_scores = _mm256_sub_epi32(_mm256_setzero_si256(), pawn_scores);
        }
    }
    
    ctx->terms.pawn_structure = _mm256_extract_epi32(horizontal_sum(pawn_scores), 0);
}

// Main evaluation function
int evaluate_position(EvalContext* ctx) {
    // Update attack maps and other context
    update_attack_maps(ctx);
    update_pawn_structure(ctx);
    
    // Clear evaluation terms
    memset(&ctx->terms, 0, sizeof(EvalTerms));
    
    // Calculate game stage
    ctx->stage = calculate_game_stage(ctx);
    
    // Evaluate each component using SIMD
    evaluate_material_simd(ctx);
    evaluate_mobility_simd(ctx);
    evaluate_king_safety_simd(ctx);
    evaluate_pawn_structure_simd(ctx);
    evaluate_piece_coordination_simd(ctx);
    
    // Apply pattern matching and positional rules
    int pattern_score = match_patterns(ctx, OPENING_PATTERNS, 
                                     sizeof(OPENING_PATTERNS)/sizeof(Pattern));
    int positional_score = apply_positional_rules(ctx, POSITIONAL_RULES,
                                                sizeof(POSITIONAL_RULES)/sizeof(PositionalRule));
    
    // Stage-dependent evaluation
    int score = 0;
    if (ctx->stage == STAGE_OPENING) {
        score = evaluate_development(ctx) +
                evaluate_center_control(ctx) +
                evaluate_king_safety_early(ctx);
    } else if (ctx->stage == STAGE_ENDGAME) {
        score = evaluate_endgame_patterns(ctx) +
                evaluate_winning_potential(ctx) +
                evaluate_fortress_detection(ctx);
    }
    
    // Combine all evaluation terms
    return (ctx->terms.material +
            ctx->terms.psqt +
            ctx->terms.mobility +
            ctx->terms.pawn_structure +
            ctx->terms.king_safety +
            ctx->terms.piece_coordination +
            pattern_score +
            positional_score +
            score) * (ctx->turn ? 1 : -1);
}
