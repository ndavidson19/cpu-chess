#include "bitboard_ops.h"
#include <stdlib.h>
#include <string.h>

// Pre-computed magic numbers for rooks (from Stockfish)
static const Bitboard ROOK_MAGIC_NUMBERS[64] = {
    0x0080001020400080ull, 0x0040001000200040ull, 0x0080081000200080ull, 0x0080040800100080ull,
    0x0080020400080080ull, 0x0080010200040080ull, 0x0080008001000200ull, 0x0080002040800100ull,
    0x0000800020400080ull, 0x0000400020005000ull, 0x0000801000200080ull, 0x0000800800100080ull,
    0x0000800400080080ull, 0x0000800200040080ull, 0x0000800100020080ull, 0x0000800040800100ull,
    0x0000208000400080ull, 0x0000404000201000ull, 0x0000808010002000ull, 0x0000808008001000ull,
    0x0000808004000800ull, 0x0000808002000400ull, 0x0000010100020004ull, 0x0000020000408104ull,
    0x0000208080004000ull, 0x0000200040005000ull, 0x0000100080200080ull, 0x0000080080100080ull,
    0x0000040080080080ull, 0x0000020080040080ull, 0x0000010080800200ull, 0x0000800080004100ull,
    0x0000204000800080ull, 0x0000200040401000ull, 0x0000100080802000ull, 0x0000080080801000ull,
    0x0000040080800800ull, 0x0000020080800400ull, 0x0000020001010004ull, 0x0000800040800100ull,
    0x0000204000808000ull, 0x0000200040008080ull, 0x0000100020008080ull, 0x0000080010008080ull,
    0x0000040008008080ull, 0x0000020004008080ull, 0x0000010002008080ull, 0x0000004081020004ull,
    0x0000204000800080ull, 0x0000200040008080ull, 0x0000100020008080ull, 0x0000080010008080ull,
    0x0000040008008080ull, 0x0000020004008080ull, 0x0000800100020080ull, 0x0000800041000080ull,
    0x00FFFCDDFCED714Aull, 0x007FFCDDFCED714Aull, 0x003FFFCDFFD88096ull, 0x0000040810002101ull,
    0x0001000204080011ull, 0x0001000204000801ull, 0x0001000082000401ull, 0x0001FFFAABFAD1A2ull
};

// Pre-computed magic numbers for bishops (from Stockfish)
static const Bitboard BISHOP_MAGIC_NUMBERS[64] = {
    0x0002020202020200ull, 0x0002020202020000ull, 0x0004010202000000ull, 0x0004040080000000ull,
    0x0001104000000000ull, 0x0000821040000000ull, 0x0000410410400000ull, 0x0000104104104000ull,
    0x0000040404040400ull, 0x0000020202020200ull, 0x0000040102020000ull, 0x0000040400800000ull,
    0x0000011040000000ull, 0x0000008210400000ull, 0x0000004104104000ull, 0x0000002082082000ull,
    0x0004000808080800ull, 0x0002000404040400ull, 0x0001000202020200ull, 0x0000800802004000ull,
    0x0000800400A00000ull, 0x0000200100884000ull, 0x0000400082082000ull, 0x0000200041041000ull,
    0x0002080010101000ull, 0x0001040008080800ull, 0x0000208004010400ull, 0x0000404004010200ull,
    0x0000840000802000ull, 0x0000404002011000ull, 0x0000808001041000ull, 0x0000404000820800ull,
    0x0001041000202000ull, 0x0000820800101000ull, 0x0000104400080800ull, 0x0000020080080080ull,
    0x0000404040040100ull, 0x0000808100020100ull, 0x0001010100020800ull, 0x0000808080010400ull,
    0x0000820820004000ull, 0x0000410410002000ull, 0x0000082088001000ull, 0x0000002011000800ull,
    0x0000080100400400ull, 0x0001010101000200ull, 0x0002020202000400ull, 0x0001010101000200ull,
    0x0000410410400000ull, 0x0000208208200000ull, 0x0000002084100000ull, 0x0000000020880000ull,
    0x0000001002020000ull, 0x0000040408020000ull, 0x0004040404040000ull, 0x0002020202020000ull,
    0x0000104104104000ull, 0x0000002082082000ull, 0x0000000020841000ull, 0x0000000000208800ull,
    0x0000000010020200ull, 0x0000000404080200ull, 0x0000040404040400ull, 0x0002020202020200ull
};

// Pre-computed shift amounts
static const int ROOK_SHIFTS[64] = {
    52, 53, 53, 53, 53, 53, 53, 52,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    52, 53, 53, 53, 53, 53, 53, 52
};

static const int BISHOP_SHIFTS[64] = {
    58, 59, 59, 59, 59, 59, 59, 58,
    59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59,
    58, 59, 59, 59, 59, 59, 59, 58
};

// Global lookup tables for attacks
MagicEntry ROOK_MAGICS[64];  
MagicEntry BISHOP_MAGICS[64];
Bitboard KNIGHT_ATTACKS_TABLE[64];
Bitboard KING_ATTACKS_TABLE[64];
Bitboard PAWN_ATTACKS_TABLE[2][64]; // [color][square]

// Mask function implementations
static inline Bitboard mask_rook_attacks(int square) {
    Bitboard attacks = 0ULL;
    int rank = square / 8;
    int file = square % 8;
    
    // North
    for (int r = rank + 1; r < 7; r++)
        attacks |= (1ULL << (r * 8 + file));
    // South
    for (int r = rank - 1; r > 0; r--)
        attacks |= (1ULL << (r * 8 + file));
    // East
    for (int f = file + 1; f < 7; f++)
        attacks |= (1ULL << (rank * 8 + f));
    // West
    for (int f = file - 1; f > 0; f--)
        attacks |= (1ULL << (rank * 8 + f));
    
    return attacks;
}

static inline Bitboard mask_bishop_attacks(int square) {
    Bitboard attacks = 0ULL;
    int rank = square / 8;
    int file = square % 8;
    
    // North-East
    for (int r = rank + 1, f = file + 1; r < 7 && f < 7; r++, f++)
        attacks |= (1ULL << (r * 8 + f));
    // North-West
    for (int r = rank + 1, f = file - 1; r < 7 && f > 0; r++, f--)
        attacks |= (1ULL << (r * 8 + f));
    // South-East
    for (int r = rank - 1, f = file + 1; r > 0 && f < 7; r--, f++)
        attacks |= (1ULL << (r * 8 + f));
    // South-West
    for (int r = rank - 1, f = file - 1; r > 0 && f > 0; r--, f--)
        attacks |= (1ULL << (r * 8 + f));
    
    return attacks;
}


// Helper function to generate knight attacks
static Bitboard generate_knight_attacks(int square) {
    Bitboard bb = 1ULL << square;
    Bitboard attacks = 0ULL;
    
    attacks |= (bb << 17) & ~FILE_A;
    attacks |= (bb << 15) & ~FILE_H;
    attacks |= (bb << 10) & ~(FILE_A | FILE_B);
    attacks |= (bb << 6) & ~(FILE_G | FILE_H);
    attacks |= (bb >> 17) & ~FILE_H;
    attacks |= (bb >> 15) & ~FILE_A;
    attacks |= (bb >> 10) & ~(FILE_G | FILE_H);
    attacks |= (bb >> 6) & ~(FILE_A | FILE_B);
    
    return attacks;
}

// Helper function to generate king attacks
static Bitboard generate_king_attacks(int square) {
    Bitboard bb = 1ULL << square;
    Bitboard attacks = 0ULL;
    
    attacks |= (bb << 8);  // north
    attacks |= (bb >> 8);  // south
    attacks |= (bb << 1) & ~FILE_A;  // east
    attacks |= (bb >> 1) & ~FILE_H;  // west
    attacks |= (bb << 9) & ~FILE_A;  // north-east
    attacks |= (bb << 7) & ~FILE_H;  // north-west
    attacks |= (bb >> 7) & ~FILE_A;  // south-east
    attacks |= (bb >> 9) & ~FILE_H;  // south-west
    
    return attacks;
}

// Generate sliding attacks for rooks
static Bitboard generate_rook_attacks(int square, Bitboard occupancy) {
    Bitboard attacks = 0ULL;
    int rank = square / 8;
    int file = square % 8;
    
    // North
    for (int r = rank + 1; r < 8; r++) {
        attacks |= (1ULL << (r * 8 + file));
        if (occupancy & (1ULL << (r * 8 + file))) break;
    }
    
    // South
    for (int r = rank - 1; r >= 0; r--) {
        attacks |= (1ULL << (r * 8 + file));
        if (occupancy & (1ULL << (r * 8 + file))) break;
    }
    
    // East
    for (int f = file + 1; f < 8; f++) {
        attacks |= (1ULL << (rank * 8 + f));
        if (occupancy & (1ULL << (rank * 8 + f))) break;
    }
    
    // West
    for (int f = file - 1; f >= 0; f--) {
        attacks |= (1ULL << (rank * 8 + f));
        if (occupancy & (1ULL << (rank * 8 + f))) break;
    }
    
    return attacks;
}

// Generate sliding attacks for bishops
static Bitboard generate_bishop_attacks(int square, Bitboard occupancy) {
    Bitboard attacks = 0ULL;
    int rank = square / 8;
    int file = square % 8;
    
    // North-East
    for (int r = rank + 1, f = file + 1; r < 8 && f < 8; r++, f++) {
        attacks |= (1ULL << (r * 8 + f));
        if (occupancy & (1ULL << (r * 8 + f))) break;
    }
    
    // North-West
    for (int r = rank + 1, f = file - 1; r < 8 && f >= 0; r++, f--) {
        attacks |= (1ULL << (r * 8 + f));
        if (occupancy & (1ULL << (r * 8 + f))) break;
    }
    
    // South-East
    for (int r = rank - 1, f = file + 1; r >= 0 && f < 8; r--, f++) {
        attacks |= (1ULL << (r * 8 + f));
        if (occupancy & (1ULL << (r * 8 + f))) break;
    }
    
    // South-West
    for (int r = rank - 1, f = file - 1; r >= 0 && f >= 0; r--, f--) {
        attacks |= (1ULL << (r * 8 + f));
        if (occupancy & (1ULL << (r * 8 + f))) break;
    }
    
    return attacks;
}

// Initialize all attack tables
void init_attack_tables(void) {
    // Initialize knight attacks
    for (int square = 0; square < 64; square++) {
        KNIGHT_ATTACKS_TABLE[square] = generate_knight_attacks(square);
    }
    
    // Initialize king attacks
    for (int square = 0; square < 64; square++) {
        KING_ATTACKS_TABLE[square] = generate_king_attacks(square);
    }
    
    // Initialize pawn attacks
    for (int square = 0; square < 64; square++) {
        // White pawn attacks
        Bitboard bb = 1ULL << square;
        PAWN_ATTACKS_TABLE[WHITE][square] = ((bb & ~FILE_A) << 7) | ((bb & ~FILE_H) << 9);
        
        // Black pawn attacks
        PAWN_ATTACKS_TABLE[BLACK][square] = ((bb & ~FILE_H) >> 7) | ((bb & ~FILE_A) >> 9);
    }
}

// Initialize magic bitboards
void init_magic_bitboards(void) {
    init_attack_tables();
    
    // Initialize rook magic bitboards
    for (int square = 0; square < 64; square++) {
        ROOK_MAGICS[square].mask = mask_rook_attacks(square);
        ROOK_MAGICS[square].magic = ROOK_MAGIC_NUMBERS[square];
        ROOK_MAGICS[square].shift = ROOK_SHIFTS[square];
        ROOK_MAGICS[square].attacks = (Bitboard*)aligned_alloc(32, sizeof(Bitboard) * (1 << (64 - ROOK_SHIFTS[square])));
        
        // Generate all possible occupancy variations
        Bitboard mask = ROOK_MAGICS[square].mask;
        int bits = __builtin_popcountll(mask);
        int variations = 1 << bits;
        
        for (int i = 0; i < variations; i++) {
            Bitboard occupancy = 0ULL;
            Bitboard temp = mask;
            int bit_count = 0;
            
            // Generate occupancy pattern
            while (temp) {
                int bit = __builtin_ctzll(temp);
                if (i & (1 << bit_count)) {
                    occupancy |= (1ULL << bit);
                }
                temp &= temp - 1;
                bit_count++;
            }
            
            // Calculate magic index
            uint64_t magic_index = ((occupancy * ROOK_MAGICS[square].magic) >> ROOK_MAGICS[square].shift);
            ROOK_MAGICS[square].attacks[magic_index] = generate_rook_attacks(square, occupancy);
        }
    }
    
    // Initialize bishop magic bitboards
    for (int square = 0; square < 64; square++) {
        BISHOP_MAGICS[square].mask = mask_bishop_attacks(square);
        BISHOP_MAGICS[square].magic = BISHOP_MAGIC_NUMBERS[square];
        BISHOP_MAGICS[square].shift = BISHOP_SHIFTS[square];
        BISHOP_MAGICS[square].attacks = (Bitboard*)aligned_alloc(32, sizeof(Bitboard) * (1 << (64 - BISHOP_SHIFTS[square])));
        
        // Generate all possible occupancy variations
        Bitboard mask = BISHOP_MAGICS[square].mask;
        int bits = __builtin_popcountll(mask);
        int variations = 1 << bits;
        
        for (int i = 0; i < variations; i++) {
            Bitboard occupancy = 0ULL;
            Bitboard temp = mask;
            int bit_count = 0;
            
            // Generate occupancy pattern
            while (temp) {
                int bit = __builtin_ctzll(temp);
                if (i & (1 << bit_count)) {
                    occupancy |= (1ULL << bit);
                }
                temp &= temp - 1;
                bit_count++;
            }
            
            // Calculate magic index
            uint64_t magic_index = ((occupancy * BISHOP_MAGICS[square].magic) >> BISHOP_MAGICS[square].shift);
            BISHOP_MAGICS[square].attacks[magic_index] = generate_bishop_attacks(square, occupancy);
        }
    }
}

// Get attacks for different piece types
Bitboard get_pawn_attacks(int square, int color) {
    return PAWN_ATTACKS_TABLE[color][square];
}

Bitboard get_knight_attacks(int square) {
    return KNIGHT_ATTACKS_TABLE[square];
}

Bitboard get_king_attacks(int square) {
    return KING_ATTACKS_TABLE[square];
}

Bitboard get_rook_attacks(int square, Bitboard occupancy) {
    MagicEntry* entry = &ROOK_MAGICS[square];
    uint64_t magic_index = (((occupancy & entry->mask) * entry->magic) >> entry->shift);
    return entry->attacks[magic_index];
}

Bitboard get_bishop_attacks(int square, Bitboard occupancy) {
    MagicEntry* entry = &BISHOP_MAGICS[square];
    uint64_t magic_index = (((occupancy & entry->mask) * entry->magic) >> entry->shift);
    return entry->attacks[magic_index];
}

Bitboard get_queen_attacks(int square, Bitboard occupancy) {
    return get_rook_attacks(square, occupancy) | get_bishop_attacks(square, occupancy);
}

// SIMD-optimized evaluation functions
int evaluate_position_simd(const int* positions, const int* weights, const int* psqt, int length) {
    __m256i sum = _mm256_setzero_si256();
    
    for (int i = 0; i < length; i += 8) {
        // Load data
        __m256i pos = _mm256_load_si256((__m256i*)&positions[i]);
        __m256i w = _mm256_load_si256((__m256i*)&weights[i]);
        __m256i sq = _mm256_load_si256((__m256i*)&psqt[i]);
        
        // Material score
        __m256i material = _mm256_mullo_epi32(pos, w);
        
        // Piece-square table score
        __m256i positional = _mm256_mullo_epi32(pos, sq);
        
        // Accumulate scores
        sum = _mm256_add_epi32(sum, material);
        sum = _mm256_add_epi32(sum, positional);
    }
    
    // Horizontal sum
    return _mm256_extract_epi32(horizontal_sum(sum), 0);
}

int evaluate_pawns_simd(Bitboard wp, Bitboard bp, const int* weights) {
    __m256i score = _mm256_setzero_si256();
    
    // Pre-compute pawn attack masks
    Bitboard w_attacks = 0, b_attacks = 0;
    Bitboard temp_wp = wp, temp_bp = bp;
    
    while (temp_wp) {
        int sq = __builtin_ctzll(temp_wp);
        w_attacks |= PAWN_ATTACKS_TABLE[WHITE][sq];
        temp_wp &= temp_wp - 1;
    }
    
    while (temp_bp) {
        int sq = __builtin_ctzll(temp_bp);
        b_attacks |= PAWN_ATTACKS_TABLE[BLACK][sq];
        temp_bp &= temp_bp - 1;
    }
    
    // Process 8 files at a time
    for (int file = 0; file < 8; file++) {
        Bitboard file_mask = 0x0101010101010101ULL << file;
        
        // Count pawns on each file
        int white_count = __builtin_popcountll(wp & file_mask);
        int black_count = __builtin_popcountll(bp & file_mask);
        
        // Analyze pawn structure
        int white_doubled = white_count > 1 ? white_count - 1 : 0;
        int black_doubled = black_count > 1 ? black_count - 1 : 0;
        
        // Check for isolated pawns
        Bitboard adjacent_files = 0;
        if (file > 0) adjacent_files |= 0x0101010101010101ULL << (file - 1);
        if (file < 7) adjacent_files |= 0x0101010101010101ULL << (file + 1);
        
        int white_isolated = (wp & adjacent_files) == 0 && white_count > 0;
        int black_isolated = (bp & adjacent_files) == 0 && black_count > 0;
        
        // Check for passed pawns
        int white_passed = 0, black_passed = 0;
        Bitboard file_wp = wp & file_mask;
        Bitboard file_bp = bp & file_mask;
        
        while (file_wp) {
            int sq = __builtin_ctzll(file_wp);
            if (!(b_attacks & PAWN_ATTACKS_TABLE[WHITE][sq])) white_passed++;
            file_wp &= file_wp - 1;
        }
        
        while (file_bp) {
            int sq = __builtin_ctzll(file_bp);
            if (!(w_attacks & PAWN_ATTACKS_TABLE[BLACK][sq])) black_passed++;
            file_bp &= file_bp - 1;
        }
        
        // Pack values into SIMD register
        __m256i counts = _mm256_set_epi32(
            white_count, black_count,
            white_doubled, black_doubled,
            white_isolated, black_isolated,
            white_passed, black_passed
        );
        
        // Load weights and multiply
        __m256i w = _mm256_load_si256((__m256i*)&weights[file * 8]);
        __m256i file_score = _mm256_mullo_epi32(counts, w);
        score = _mm256_add_epi32(score, file_score);
    }
    
    return _mm256_extract_epi32(horizontal_sum(score), 0);
}

// Cleanup function
void cleanup_magic_bitboards(void) {
    for (int square = 0; square < 64; square++) {
        free(ROOK_MAGICS[square].attacks);
        free(BISHOP_MAGICS[square].attacks);
    }
}