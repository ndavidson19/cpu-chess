#include "patterns.h"
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>
#include "utils.h"

// Define the pattern data using arrays of constants
const SIMDPattern LONDON_PATTERNS[] ALIGN_32 = {
    {
        .key_squares = {
            0x0000001414000000ULL,  // Central  presence
            0x0000200000000000ULL,  // Nf3 control
            0x0000000040400000ULL,  // Dark square bishop control
            0x0000141400000000ULL   // d4, e3 control
        },
        .piece_masks = {
            0x0000000000001400ULL,  // d4+e3 structure
            0x0000000000200000ULL,  // Nf3
            0x0000000000000040ULL,  // Bf4
            0x0000000000001000ULL   // d4 
        },
        .scores = {15, 20, 25, 30}  // Development scores
    },
    // Add more London patterns as needed...
};

const SIMDPattern CARO_KANN_PATTERNS[] ALIGN_32 = {
    {
        .key_squares = {
            0x0000141414000000ULL,  // Extended control
            0x0000001414000000ULL,  // Advanced center
            0x0000000014140000ULL,  // c6-d5 structure
            0x0000000000140000ULL   // c6 base
        },
        .piece_masks = {
            0x0000001414141414ULL,  // Ideal structure
            0x0000000014141414ULL,  // Full structure
            0x0000000000141414ULL,  // Expanded structure
            0x0000000000001414ULL   // Basic structure
        },
        .scores = {20, 25, 30, 35}  // Structure scores
    },
    // Add more Caro-Kann patterns as needed...
};

const SIMDPattern ENDGAME_PATTERNS[] ALIGN_32 = {
    {
        .key_squares = {
            0x0000000000FF0000ULL,  // Enemy king zone
            0x00000000FF000000ULL,  //  advance
            0x000000FF00000000ULL,  // King zone
            0x0000FF0000000000ULL   // Rook cutting off king
        },
        .piece_masks = {
            0x0000000000000080ULL,  // Enemy king
            0x0000000000080000ULL,  //  position
            0x0000000000800000ULL,  // King position
            0x0000008000000000ULL   // Rook position
        },
        .scores = {25, 50, 75, 100}  // Position scores
    },
    // Add more endgame patterns as needed...
};

__m256i match_pattern_simd(const __m256i position, const SIMDPattern* pattern) {
    // Load key_squares and piece_masks into SIMD variables
    __m256i key_squares_simd = _mm256_loadu_si256((const __m256i*)pattern->key_squares);
    __m256i piece_masks_simd = _mm256_loadu_si256((const __m256i*)pattern->piece_masks);
    
    // Compare key squares (all must match)
    __m256i key_match = _mm256_and_si256(position, key_squares_simd);
    __m256i key_cmp = _mm256_cmpeq_epi64(key_match, key_squares_simd);
    
    // Compare piece placement (required pieces must be present)
    __m256i piece_match = _mm256_and_si256(position, piece_masks_simd);
    __m256i piece_cmp = _mm256_cmpeq_epi64(piece_match, piece_masks_simd);
    
    // Combine results
    return _mm256_and_si256(key_cmp, piece_cmp);
}


// SIMD optimized score calculation
static __m128i calculate_scores_simd(const __m256i matches, const __m128i scores) {
    // Extract match results to 32-bit integers
    __m128i match_low = _mm256_extracti128_si256(matches, 0);
    __m128i match_high = _mm256_extracti128_si256(matches, 1);
    
    // Convert matches to mask
    __m128i mask = _mm_and_si128(
        _mm_packs_epi32(match_low, match_high),
        _mm_set1_epi32(0xFFFFFFFF)
    );
    
    // Apply mask to scores
    return _mm_and_si128(mask, scores);
}

// Memory-efficient endgame tablebases using compressed storage
static struct {
    uint32_t* positions;     // Compressed position hashes
    uint8_t* moves;         // Compressed best moves
    int8_t* scores;        // Position scores
    uint32_t size;         // Number of positions
    bool is_loaded;        // Whether tablebase is currently loaded
    uint16_t material_key;  // Material key for current tablebase
} endgame_tb = {0};

// SIMD-optimized London System evaluation
int evaluate_london_position(const Position* pos) {
    __m256i position_vector = load_position_simd(pos);
    __m128i total_scores = _mm_setzero_si128();
    
    // Process patterns in parallel
    for (size_t i = 0; i < sizeof(LONDON_PATTERNS)/sizeof(SIMDPattern); i++) {
        __m256i matches = match_pattern_simd(position_vector, &LONDON_PATTERNS[i]);
        __m128i scores_simd = _mm_load_si128((const __m128i*)LONDON_PATTERNS[i].scores);
        __m128i scores = calculate_scores_simd(matches, scores_simd);
        total_scores = _mm_add_epi32(total_scores, scores);
    }
    
    // Additional SIMD-optimized evaluations
    __m256i control_masks = _mm256_set_epi64x(
        0x0000001818000000ULL,  // Central squares
        0x00003C3C3C3C0000ULL,  // Extended center
        0x0000000040201008ULL,  // Light square control
        0x0000000008102040ULL   // Dark square control
    );
    
    __m256i control = _mm256_and_si256(position_vector, control_masks);
    __m128i control_scores = calculate_control_scores_simd(control);
    total_scores = _mm_add_epi32(total_scores, control_scores);
    
    // Horizontal sum
    return _mm_extract_epi32(horizontal_sum_128(total_scores), 0);
}

// SIMD-optimized Caro-Kann evaluation
int evaluate_caro_kann_position(const Position* pos) {
    __m256i position_vector = load_position_simd(pos);
    __m128i total_scores = _mm_setzero_si128();
    
    // Process patterns in parallel
    for (size_t i = 0; i < sizeof(CARO_KANN_PATTERNS)/sizeof(SIMDPattern); i++) {
        __m256i matches = match_pattern_simd(position_vector, &CARO_KANN_PATTERNS[i]);
        __m128i scores_simd = _mm_load_si128((const __m128i*)CARO_KANN_PATTERNS[i].scores);
        __m128i scores = calculate_scores_simd(matches, scores_simd);
        total_scores = _mm_add_epi32(total_scores, scores);
    }
    
    // Specific Caro-Kann structure evaluation using SIMD
    __m256i structure_masks = _mm256_set_epi64x(
        0x0000000014000000ULL,  // c6 
        0x0000000014140000ULL,  // c6-d5 chain
        0x0000001414140000ULL,  // Extended chain
        0x0000000008000000ULL   // e6 square (weakness check)
    );
    
    __m256i structure = _mm256_and_si256(position_vector, structure_masks);
    __m128i structure_scores = calculate_structure_scores_simd(structure);
    total_scores = _mm_add_epi32(total_scores, structure_scores);
    
    return _mm_extract_epi32(horizontal_sum_128(total_scores), 0);
}

// Define the compressed data arrays
static const uint8_t compressed_data_1[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
static const int8_t dtm_table_1[] = {0, 0, 0, 0, 0, 0, 0, 0};

static const uint8_t compressed_data_2[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07};
static const int8_t dtm_table_2[] = {0, 0, 0, 0, 0, 0, 0, 0};

// Initialize ENDGAME_DEFINITIONS
const EndgameDefinition ENDGAME_DEFINITIONS[] = {
    {
        .material_key = 0x00010000,
        .position_count = 1000,
        .compressed_data = compressed_data_1,
        .chunk_size = sizeof(compressed_data_1),
        .dtm_table = dtm_table_1,
    },
    {
        .material_key = 0x01010000,
        .position_count = 2000,
        .compressed_data = compressed_data_2,
        .chunk_size = sizeof(compressed_data_2),
        .dtm_table = dtm_table_2,
    }
};



const EndgameDefinition* find_endgame_definition(uint16_t material_key) {
    // Find the correct endgame definition
    for (size_t i = 0; i < sizeof(ENDGAME_DEFINITIONS)/sizeof(EndgameDefinition); i++) {
        if (ENDGAME_DEFINITIONS[i].material_key == material_key) {
            return &ENDGAME_DEFINITIONS[i];
        }
    }
    
    return NULL;
}

void decompress_tablebase_chunk(
    const uint8_t* compressed_data,
    uint32_t* positions,
    uint8_t* moves,
    int8_t* scores,
    uint32_t position_count
) {
    // Decompress data in parallel
    #pragma omp parallel for
    for (size_t i = 0; i < position_count; i++) {
        positions[i] = compressed_data[i * 4] |
                       (compressed_data[i * 4 + 1] << 8) |
                       (compressed_data[i * 4 + 2] << 16) |
                       (compressed_data[i * 4 + 3] << 24);
        moves[i] = compressed_data[position_count * 4 + i];
        scores[i] = compressed_data[position_count * 5 + i];
    }
}

// Memory-efficient endgame handling
void load_endgame_solver(int material_key) {
    if (endgame_tb.is_loaded && endgame_tb.material_key == material_key)
        return;
        
    // Clean up previous tablebase if loaded
    if (endgame_tb.is_loaded) {
        cleanup_endgame_solver();
    }
    
    // Load compressed tablebase data
    const EndgameDefinition* def = find_endgame_definition(material_key);
    if (!def) return;
    
    // Decompress only the needed portion
    size_t positions_size = def->position_count * sizeof(uint32_t);
    size_t moves_size = def->position_count * sizeof(uint8_t);
    size_t scores_size = def->position_count * sizeof(int8_t);
    
    endgame_tb.positions = aligned_alloc(32, positions_size);
    endgame_tb.moves = aligned_alloc(32, moves_size);
    endgame_tb.scores = aligned_alloc(32, scores_size);
    
    decompress_tablebase_chunk(def->compressed_data, 
                             endgame_tb.positions,
                             endgame_tb.moves,
                             endgame_tb.scores,
                             def->position_count);
                             
    endgame_tb.size = def->position_count;
    endgame_tb.is_loaded = true;
    endgame_tb.material_key = material_key;
}

__m256i count_material_simd(const Position* pos) {
    // Load piece counts into SIMD registers
    __m256i white_pieces = _mm256_load_si256((__m256i*)pos->pieces[WHITE]);
    __m256i black_pieces = _mm256_load_si256((__m256i*)pos->pieces[BLACK]);

    // Count material using SIMD
    __m256i white_counts = avx2_popcnt_epi64(white_pieces);
    __m256i black_counts = avx2_popcnt_epi64(black_pieces);

    // Horizontal sum of piece counts
    __m128i total_counts = _mm256_extracti128_si256(white_counts, 0);
    total_counts = _mm_add_epi64(total_counts, _mm256_extracti128_si256(white_counts, 1));
    total_counts = _mm_add_epi64(total_counts, _mm256_extracti128_si256(black_counts, 0));

    return _mm256_castsi128_si256(total_counts);

}

uint16_t calculate_material_key_simd(__m256i material_counts) {
    // Extract piece counts to 64-bit integers
    uint64_t white_counts = _mm256_extract_epi64(material_counts, 0);  // Extract lower 64 bits
    uint64_t black_counts = _mm256_extract_epi64(material_counts, 2);  // Extract upper 64 bits


    // Calculate material key using SIMD
    uint64_t material_key = (white_counts << 32) | black_counts;
    return (uint16_t)(material_key ^ (material_key >> 16));
}

int calculate_dtm(int index, uint16_t material_key) {
    // Find the correct endgame definition
    const EndgameDefinition* def = find_endgame_definition(material_key);
    if (!def) return 0;

    // Calculate distance to mate using DTM table
    return def->dtm_table[index];
}

// SIMD-optimized endgame probing
EndgameEntry* probe_endgame_table(const Position* pos) {
    static EndgameEntry entry;
    
    // Quick material check using SIMD
    __m256i material_counts = count_material_simd(pos);
    uint16_t material_key = calculate_material_key_simd(material_counts);
    
    if (!is_endgame_material(material_key))
        return NULL;
        
    // Ensure correct tablebase is loaded
    load_endgame_solver(material_key);
    if (!endgame_tb.is_loaded)
        return NULL;
        
    // Generate position hash
    uint32_t pos_hash = compress_position(pos);
    
    // SIMD-accelerated binary search in tablebase
    int index = binary_search_simd(endgame_tb.positions, pos_hash, endgame_tb.size);
    if (index < 0)
        return NULL;
        
    // Fill entry with found data
    entry.position_hash = pos_hash;
    entry.best_move = endgame_tb.moves[index];
    entry.eval = endgame_tb.scores[index];
    entry.dtm = calculate_dtm(index, material_key);
    
    return &entry;
}

// Helper function for SIMD binary search
static int binary_search_simd(const uint32_t* array, uint32_t target, size_t size) {
    int left = 0;
    int right = size - 1;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        
        // Load 8 consecutive values
        __m256i values = _mm256_load_si256((__m256i*)&array[mid & ~7]);
        __m256i targets = _mm256_set1_epi32(target);
        
        // Compare all values simultaneously
        __m256i cmp = _mm256_cmpeq_epi32(values, targets);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));

        // Continue binary search with SIMD compare results
        if (mask != 0) {
            // Found the value, calculate exact index
            return (mid & ~7) + __builtin_ctz(mask);
        }
        
        // Compare for less than using SIMD
        __m256i lt = _mm256_cmpgt_epi32(targets, values);
        int lt_mask = _mm256_movemask_ps(_mm256_castsi256_ps(lt));
        
        // Adjust binary search bounds based on SIMD comparison
        if (lt_mask == 0xFF) {
            left = (mid + 8) & ~7;  // All values less, move right
        } else {
            right = (mid - 1) & ~7;  // Some values greater, move left
        }
    }
    
    return -1;  // Not found
}

// SIMD-optimized position compression
static uint32_t compress_position(const Position* pos) {
    // Load position data into SIMD registers
    __m256i pieces = _mm256_load_si256((__m256i*)pos->pieces);
    __m256i occupied = _mm256_load_si256((__m256i*)pos->occupied);
    
    // Compress using parallel operations
    __m256i compressed = _mm256_xor_si256(pieces, occupied);
    compressed = _mm256_mul_epu32(compressed, _mm256_set1_epi64x(0x783AF2E147D96C5BULL));
    
    // Mix the results
    uint64_t hash = _mm256_extract_epi64(compressed, 0) ^
                   _mm256_extract_epi64(compressed, 1) ^
                   _mm256_extract_epi64(compressed, 2) ^
                   _mm256_extract_epi64(compressed, 3);
                   
    return (uint32_t)(hash ^ (hash >> 32));
}

// Optimized pattern matching with SIMD for development evaluation
static __m256i evaluate_development_simd(const Position* pos) {
    __m256i piece_development = _mm256_setzero_si256();
    
    // Load piece positions
    __m256i knights = _mm256_load_si256((__m256i*)&pos->pieces[0][KNIGHT]);
    __m256i bishops = _mm256_load_si256((__m256i*)&pos->pieces[0][BISHOP]);
    
    // Development masks for different pieces
    __m256i knight_masks = _mm256_set_epi64x(
        0x0000000000240000ULL,  // Ideal knight squares
        0x0000000042000000ULL,  // Good knight development
        0x0000000000003C00ULL,  // Starting squares
        0x0000000000000042ULL   // Initial squares
    );
    
    __m256i bishop_masks = _mm256_set_epi64x(
        0x0000000040201008ULL,  // Ideal bishop squares
        0x0000000010080402ULL,  // Good bishop development
        0x0000000000002400ULL,  // Starting squares
        0x0000000000000024ULL   // Initial squares
    );
    
    // Calculate development scores using SIMD
    __m256i knight_dev = _mm256_and_si256(knights, knight_masks);
    __m256i bishop_dev = _mm256_and_si256(bishops, bishop_masks);
    
    // Combine scores
    return _mm256_add_epi64(knight_dev, bishop_dev);
}

__m256i calculate_doubled_s_simd(__m256i white_s, __m256i black_s) {
    // Calculate doubled s using SIMD
    __m256i white_doubled = _mm256_and_si256(white_s, _mm256_slli_epi64(white_s, 8));
    __m256i black_doubled = _mm256_and_si256(black_s, _mm256_slli_epi64(black_s, 8));
    
    // Combine doubled  scores
    return _mm256_add_epi64(white_doubled, black_doubled);
}

__m256i calculate_isolated_s_simd(__m256i white_s, __m256i black_s) {
    // Calculate isolated s using SIMD
    __m256i white_isolated = _mm256_and_si256(white_s, _mm256_slli_epi64(white_s, 1));
    white_isolated = _mm256_or_si256(white_isolated, _mm256_and_si256(white_s, _mm256_srli_epi64(white_s, 1)));
    
    __m256i black_isolated = _mm256_and_si256(black_s, _mm256_slli_epi64(black_s, 1));
    black_isolated = _mm256_or_si256(black_isolated, _mm256_and_si256(black_s, _mm256_srli_epi64(black_s, 1)));
    
    // Combine isolated  scores
    return _mm256_add_epi64(white_isolated, black_isolated);
}

// SIMD-optimized  structure evaluation for both openings
static __m256i evaluate__structure_simd(const Position* pos) {
    // Load  positions
    __m256i white_s = _mm256_load_si256((__m256i*)&pos->pieces[WHITE][0]);
    __m256i black_s = _mm256_load_si256((__m256i*)&pos->pieces[BLACK][0]);
    
    // Define structure masks for both London and Caro-Kann
    __m256i london_masks = _mm256_set_epi64x(
        0x0000001414140000ULL,  // Ideal London center
        0x0000000014140000ULL,  // Basic London structure
        0x0000141414000000ULL,  // Extended London control
        0x0000001414000000ULL   // Minimal London setup
    );
    
    __m256i caro_masks = _mm256_set_epi64x(
        0x0000000014140000ULL,  // Ideal Caro-Kann structure
        0x0000000000140000ULL,  // Basic Caro-Kann 
        0x0000001414140000ULL,  // Extended Caro-Kann control
        0x0000000014000000ULL   // Minimal Caro-Kann setup
    );
    
    // Evaluate structures using SIMD
    __m256i london_structure = _mm256_and_si256(white_s, london_masks);
    __m256i caro_structure = _mm256_and_si256(black_s, caro_masks);
    
    // Calculate doubled and isolated s using SIMD
    __m256i doubled = calculate_doubled_s_simd(white_s, black_s);
    __m256i isolated = calculate_isolated_s_simd(white_s, black_s);
    
    // Combine all evaluations
    __m256i structure_scores = _mm256_add_epi64(
        _mm256_add_epi64(london_structure, caro_structure),
        _mm256_sub_epi64(doubled, isolated)
    );
    
    return structure_scores;
}

// SIMD-optimized piece coordination evaluation
static __m256i evaluate_coordination_simd(const Position* pos) {
    // Load piece positions into SIMD registers
    __m256i knights = _mm256_load_si256((__m256i*)&pos->pieces[0][KNIGHT]);
    __m256i bishops = _mm256_load_si256((__m256i*)&pos->pieces[0][BISHOP]);
    __m256i rooks = _mm256_load_si256((__m256i*)&pos->pieces[0][ROOK]);
    
    // Define coordination masks
    __m256i coordination_masks = _mm256_set_epi64x(
        0x0000000040201008ULL,  // Bishop coordination
        0x0000000002040810ULL,  // Knight coordination
        0x00000000FF000000ULL,  // Rook coordination
        0x0000001818180000ULL   // Central coordination
    );
    
    // Calculate piece coordination using SIMD
    __m256i bishop_coord = _mm256_and_si256(bishops, coordination_masks);
    __m256i knight_coord = _mm256_and_si256(knights, coordination_masks);
    __m256i rook_coord = _mm256_and_si256(rooks, coordination_masks);
    
    // Combine coordination scores
    return _mm256_add_epi64(
        _mm256_add_epi64(bishop_coord, knight_coord),
        rook_coord
    );
}

// Cleanup and memory management
void cleanup_endgame_solver(void) {
    if (endgame_tb.is_loaded) {
        aligned_free(endgame_tb.positions);
        aligned_free(endgame_tb.moves);
        aligned_free(endgame_tb.scores);
        endgame_tb.is_loaded = false;
    }
}

// Initialize pattern tables and SIMD operations
void init_pattern_tables(void) {
    // Ensure proper alignment for SIMD operations
    assert(((uintptr_t)LONDON_PATTERNS & 31) == 0);
    assert(((uintptr_t)CARO_KANN_PATTERNS & 31) == 0);
    assert(((uintptr_t)ENDGAME_PATTERNS & 31) == 0);
    
    // Initialize any dynamic pattern data
    init_development_masks();
    init_coordination_masks();
    init_structure_masks();
}
