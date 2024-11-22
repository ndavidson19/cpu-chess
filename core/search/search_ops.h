#ifndef SEARCH_OPS_H
#define SEARCH_OPS_H

#include <stdint.h>
#include "../evaluation/eval_ops.h"
#include "../common/types.h"
#include "../common/context.h"
#include "../move/move.h"
#include "../utils/utils.h"

#define MATE_IN_MAX 30000
#define TT_EXACT 0
#define TT_ALPHA 1  // Upper bound
#define TT_BETA 2   // Lower bound

// Constants for TT sizing and alignment
#define TT_ALIGNMENT 32
#define MIN_TT_SIZE (1024)         // 1K entries minimum
#define MAX_TT_SIZE (1024*1024*64) // 64M entries maximum


// Search result
typedef struct {
    uint16_t best_move;     // Best move found
    int score;              // Position evaluation
    PVLine pv;             // Principal variation
    SearchStats stats;      // Search statistics
} SearchResult;

// Core search functions
void init_search(SearchContext* ctx, uint32_t tt_size);
EvalContext* create_eval_context_from_search(SearchContext* search_ctx);
SearchResult search_position(Position* pos, SearchParams* params, SearchContext* ctx);
void cleanup_search(SearchContext* ctx);

// Move generation and ordering
int generate_moves(const Position* pos, SearchMove* moves);
void sort_moves(SearchMove* moves, int count, const SearchContext* ctx, int ply);
int score_move(uint16_t move, const SearchContext* ctx, int ply);

// Search helper functions
int quiescence_search(Position* pos, int alpha, int beta, SearchContext* ctx);
bool should_stop_search(const SearchContext* ctx);

// Utility functions
uint64_t get_hash_key(const Position* pos);
void update_history(SearchContext* ctx, uint16_t move, int depth);
void update_killers(SearchContext* ctx, uint16_t move, int ply);
TTEntry* probe_tt(const SearchContext* ctx, uint64_t key);
void store_tt(SearchContext* ctx, uint64_t key, int score, uint16_t move, 
                          int depth, uint8_t flag);

extern uint64_t piece_keys[2][6][64];
extern uint64_t castling_keys[16];
extern uint64_t en_passant_keys[64];
extern uint64_t side_key;

void init_hash_keys(void);

#endif // SEARCH_OPS_H