#include "eval_ops.h"
#include "patterns.h"
#include <string.h>
#include <stdbool.h>
#include "../utils/utils.h"


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

int evaluate_development(const EvalContext* ctx) {
    return _mm256_extract_epi32(evaluate_development_simd(ctx->pos), 0);
}

int evaluate_center_control(const EvalContext* ctx) {
    return _mm256_extract_epi32(evaluate_center_control_simd(ctx->pos), 0);
}

int evaluate_king_safety_early(const EvalContext* ctx) {
    return _mm256_extract_epi32(evaluate_king_safety_early_simd(ctx->pos), 0);
}

int evaluate_endgame_patterns(const EvalContext* ctx) {
    return _mm256_extract_epi32(evaluate_endgame_patterns_simd(ctx->pos), 0);
}

int evaluate_winning_potential(const EvalContext* ctx) {
    return _mm256_extract_epi32(evaluate_winning_potential_simd(ctx->pos), 0);
}

int evaluate_fortress_detection(const EvalContext* ctx) {
    return _mm256_extract_epi32(evaluate_fortress_detection_simd(ctx->pos), 0);
}


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
                Bitboard piece_bb = pieces & -pieces; // Get least significant bit
                int piece_square = __builtin_ctzll(pieces);
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
                
                pieces &= pieces - 1;  // Clear least significant bit
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

// Function to calculate game stage based on material and position
int calculate_game_stage(const EvalContext* ctx) {
    // Count total material excluding pawns
    int material_count = 0;
    for (int color = 0; color < 2; color++) {
        for (int piece = KNIGHT; piece <= QUEEN; piece++) {
            material_count += __builtin_popcountll(ctx->pieces[color][piece]);
        }
    }
    
    // Define stage thresholds
    const int OPENING_THRESHOLD = 14;  // Both sides have most pieces
    const int MIDDLEGAME_THRESHOLD = 10;  // Some pieces exchanged
    const int ENDGAME_THRESHOLD = 6;  // Few pieces remain
    
    if (material_count >= OPENING_THRESHOLD) {
        return STAGE_OPENING;
    } else if (material_count >= MIDDLEGAME_THRESHOLD) {
        return STAGE_MIDDLEGAME;
    } else if (material_count >= ENDGAME_THRESHOLD) {
        return STAGE_ENDGAME;
    }
    return STAGE_ENDGAME;
}

// Function to update attack maps for all pieces
void update_attack_maps(EvalContext* ctx) {
    // Clear existing attack maps
    memset(ctx->attacks, 0, sizeof(ctx->attacks));
    
    // Update for each color and piece type
    for (int color = 0; color < 2; color++) {
        Bitboard occupied = ctx->occupied[WHITE] | ctx->occupied[BLACK];
        
        // Pawns
        Bitboard pawns = ctx->pieces[color][PAWN];
        while (pawns) {
            int sq = get_lsb_index(pawns);
            ctx->attacks[color][PAWN] |= get_pawn_attacks(sq, color);
            pawns &= pawns - 1;
        }
        
        // Knights
        Bitboard knights = ctx->pieces[color][KNIGHT];
        while (knights) {
            int sq = get_lsb_index(knights);
            ctx->attacks[color][KNIGHT] |= get_knight_attacks(sq);
            knights &= knights - 1;
        }
        
        // Bishops
        Bitboard bishops = ctx->pieces[color][BISHOP];
        while (bishops) {
            int sq = get_lsb_index(bishops);
            ctx->attacks[color][BISHOP] |= get_bishop_attacks(sq, occupied);
            bishops &= bishops - 1;
        }
        
        // Rooks
        Bitboard rooks = ctx->pieces[color][ROOK];
        while (rooks) {
            int sq = get_lsb_index(rooks);
            ctx->attacks[color][ROOK] |= get_rook_attacks(sq, occupied);
            rooks &= rooks - 1;
        }
        
        // Queens
        Bitboard queens = ctx->pieces[color][QUEEN];
        while (queens) {
            int sq = get_lsb_index(queens);
            ctx->attacks[color][QUEEN] |= get_queen_attacks(sq, occupied);
            queens &= queens - 1;
        }
        
        // Kings
        Bitboard kings = ctx->pieces[color][KING];
        while (kings) {
            int sq = get_lsb_index(kings);
            ctx->attacks[color][KING] |= get_king_attacks(sq);
            kings &= kings - 1;
        }
    }
}

// Function to update pawn structure information
void update_pawn_structure(EvalContext* ctx) {
    // Clear existing pawn structure
    memset(ctx->pawn_shields, 0, sizeof(ctx->pawn_shields));
    
    // Update for each color
    for (int color = 0; color < 2; color++) {
        Bitboard pawns = ctx->pieces[color][PAWN];
        Bitboard kings = ctx->pieces[color][KING];
        
        // Find king square
        int king_sq = get_lsb_index(kings);
        int king_file = king_sq % 8;
        
        // Calculate pawn shield
        Bitboard shield_mask = 0;
        if (color == WHITE) {
            // Shield is on 6th and 7th ranks in front of king
            shield_mask = FILE_MASKS[king_file];
            if (king_file > 0) shield_mask |= FILE_MASKS[king_file - 1];
            if (king_file < 7) shield_mask |= FILE_MASKS[king_file + 1];
            shield_mask &= (RANK_6 | RANK_7);
        } else {
            // Shield is on 2nd and 3rd ranks in front of king
            shield_mask = FILE_MASKS[king_file];
            if (king_file > 0) shield_mask |= FILE_MASKS[king_file - 1];
            if (king_file < 7) shield_mask |= FILE_MASKS[king_file + 1];
            shield_mask &= (RANK_2 | RANK_3);
        }
        
        // Store pawn shield
        ctx->pawn_shields[color] = pawns & shield_mask;
    }
}

// Fix match_patterns function - incorrect position_bits type handling
int match_patterns(const EvalContext* ctx, const Pattern* patterns, int count) {
    int total_score = 0;
    Bitboard total_position = 0;
    
    // Combine position bits for both colors
    for (int piece = 0; piece < 6; piece++) {
        total_position |= ctx->pieces[WHITE][piece];
        total_position |= ctx->pieces[BLACK][piece];
    }

    // Process patterns in groups of 4 for SIMD efficiency
    int i;
    for (i = 0; i + 3 < count; i += 4) {
        __m256i masks = _mm256_set_epi64x(
            patterns[i+3].mask,
            patterns[i+2].mask,
            patterns[i+1].mask,
            patterns[i].mask
        );
        
        __m256i required = _mm256_set_epi64x(
            patterns[i+3].required,
            patterns[i+2].required,
            patterns[i+1].required,
            patterns[i].required
        );
        
        __m256i forbidden = _mm256_set_epi64x(
            patterns[i+3].forbidden,
            patterns[i+2].forbidden,
            patterns[i+1].forbidden,
            patterns[i].forbidden
        );

        __m256i pos_vector = _mm256_set1_epi64x(total_position);
        
        // Check pattern matches using SIMD
        __m256i masked_pos = _mm256_and_si256(pos_vector, masks);
        __m256i req_match = _mm256_cmpeq_epi64(
            _mm256_and_si256(masked_pos, required),
            required
        );
        __m256i forb_match = _mm256_cmpeq_epi64(
            _mm256_and_si256(masked_pos, forbidden),
            _mm256_setzero_si256()
        );
        
        // Combine required and forbidden matches
        __m256i pattern_matches = _mm256_and_si256(req_match, forb_match);
        
        // Extract match results and accumulate scores
        uint64_t matches = _mm256_extract_epi64(pattern_matches, 0) |
                          (_mm256_extract_epi64(pattern_matches, 1) << 1) |
                          (_mm256_extract_epi64(pattern_matches, 2) << 2) |
                          (_mm256_extract_epi64(pattern_matches, 3) << 3);
        
        for (int j = 0; j < 4 && (i + j) < count; j++) {
            if (matches & (1ULL << j)) {
                if (ctx->stage == STAGE_OPENING || ctx->stage == STAGE_MIDDLEGAME) {
                    total_score += patterns[i+j].score_mg;
                } else {
                    total_score += patterns[i+j].score_eg;
                }
            }
        }
    }
    
    // Handle remaining patterns
    for (; i < count; i++) {
        Bitboard masked_pos = total_position & patterns[i].mask;
        if ((masked_pos & patterns[i].required) == patterns[i].required &&
            (masked_pos & patterns[i].forbidden) == 0) {
            
            if (ctx->stage == STAGE_OPENING || ctx->stage == STAGE_MIDDLEGAME) {
                total_score += patterns[i].score_mg;
            } else {
                total_score += patterns[i].score_eg;
            }
        }
    }
    
    return total_score;
}

// Fix check_outpost function
static inline bool check_outpost(const EvalContext* ctx, int square, int color) {
    int file = square % 8;
    
    // Check if square is protected by friendly pawn
    Bitboard pawn_attacks = color == WHITE ? 
        ((1ULL << square) >> 9) | ((1ULL << square) >> 7) :
        ((1ULL << square) << 9) | ((1ULL << square) << 7);
    
    // Check if no enemy pawns can attack the square
    Bitboard enemy_pawn_control = 0;
    if (file > 0) enemy_pawn_control |= FILE_MASKS[file - 1];
    if (file < 7) enemy_pawn_control |= FILE_MASKS[file + 1];
    enemy_pawn_control &= ctx->pieces[!color][PAWN];
    
    return (ctx->pieces[color][PAWN] & pawn_attacks) && !enemy_pawn_control;
}

static inline bool check_open_diagonals(const EvalContext* ctx, int square) {
    Bitboard diagonals = get_bishop_attacks(square, ctx->occupied[WHITE] | ctx->occupied[BLACK]);
    return __builtin_popcountll(diagonals) >= 7; // At least 7 diagonal squares accessible
}

static inline bool check_open_file(const EvalContext* ctx, int square) {
    int file = square % 8;
    Bitboard file_mask = FILE_MASKS[file];
    return !(file_mask & (ctx->pieces[WHITE][PAWN] | ctx->pieces[BLACK][PAWN]));
}

int apply_positional_rules(const EvalContext* ctx, const PositionalRule* rules, int count) {
    int total_score = 0;
    
    // Process each color
    for (int color = 0; color < 2; color++) {
        // Process rules in groups of 4 for SIMD efficiency
        for (int i = 0; i < count; i += 4) {
            // Define arrays to hold bonuses before SIMD packing
            int mg_bonuses[4] = {0};
            int eg_bonuses[4] = {0};
            
            // Process up to 4 rules
            int rules_to_process = min(4, count - i);
            
            for (int j = 0; j < rules_to_process; j++) {
                const PositionalRule* rule = &rules[i + j];
                Bitboard pieces = ctx->pieces[color][rule->piece_type];
                
                while (pieces) {
                    int square = get_lsb_index(pieces);
                    bool condition_met = false;
                    
                    // Check positional condition
                    switch (rule->condition) {
                        case POS_OUTPOST:
                            condition_met = check_outpost(ctx, square, color);
                            break;
                            
                        case POS_OPEN_DIAGONALS:
                            condition_met = check_open_diagonals(ctx, square);
                            break;
                            
                        case POS_OPEN_FILE:
                            condition_met = check_open_file(ctx, square);
                            break;
                    }
                    
                    if (condition_met) {
                        mg_bonuses[j] += rule->bonus_mg;
                        eg_bonuses[j] += rule->bonus_eg;
                    }
                    
                    pieces &= pieces - 1;  // Clear least significant bit
                }
            }
            
            // Pack bonuses into SIMD registers
            __m128i bonuses_mg = _mm_setr_epi32(
                mg_bonuses[0],
                mg_bonuses[1],
                mg_bonuses[2],
                mg_bonuses[3]
            );
            
            __m128i bonuses_eg = _mm_setr_epi32(
                eg_bonuses[0],
                eg_bonuses[1],
                eg_bonuses[2],
                eg_bonuses[3]
            );
            
            // Calculate stage-dependent score
            __m128i stage_factor = _mm_set1_epi32(ctx->stage == STAGE_ENDGAME ? 256 : 0);
            __m128i interpolated = _mm_add_epi32(
                _mm_mullo_epi32(
                    _mm_sub_epi32(_mm_set1_epi32(256), stage_factor),
                    bonuses_mg
                ),
                _mm_mullo_epi32(stage_factor, bonuses_eg)
            );
            interpolated = _mm_srli_epi32(interpolated, 8);  // Divide by 256
            
            // Extract scores using fixed indices
            int rule_scores = 0;
            switch (rules_to_process) {
                case 4:
                    rule_scores += _mm_extract_epi32(interpolated, 3);
                    /* fallthrough */
                case 3:
                    rule_scores += _mm_extract_epi32(interpolated, 2);
                    /* fallthrough */
                case 2:
                    rule_scores += _mm_extract_epi32(interpolated, 1);
                    /* fallthrough */
                case 1:
                    rule_scores += _mm_extract_epi32(interpolated, 0);
            }
            
            total_score += rule_scores;
        }
    }
    
    // Apply color factor
    return ctx->turn == WHITE ? total_score : -total_score;
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
