#ifndef SEARCH_OPS_H
#define SEARCH_OPS_H

#include <stdint.h>
#include "../bitboard/bitboard_ops.h"
#include "../evaluation/eval_ops.h"
#include "../evaluation/patterns.h"
#include "../move/move.h"

// Memory-efficient move representation
typedef struct {
    uint16_t move;      // Packed move (from:6|to:6|promotion:4)
    int16_t score;      // Move score for ordering
} SearchMove;

// Transposition table entry
typedef struct {
    uint64_t key;       // Position hash
    int16_t score;      // Evaluation score
    uint16_t move;      // Best move found
    uint8_t depth;      // Search depth
    uint8_t flag;       // Entry type (exact, upper, lower bound)
} TTEntry;

// Principal Variation (PV) storage
typedef struct {
    uint16_t moves[64];  // PV line
    int length;          // Number of moves in line
} PVLine;

// Search statistics for tuning
typedef struct {
    uint32_t nodes;          // Nodes searched
    uint32_t tt_hits;        // Transposition table hits
    uint32_t beta_cutoffs;   // Beta cutoffs achieved
    uint32_t futile_prunes;  // Positions pruned by futility
    uint32_t null_cuts;      // Null move cutoffs
} SearchStats;

typedef struct {
    int max_depth;           // Maximum search depth
    int64_t time_limit;      // Time limit in milliseconds
    bool use_null_move;      // Enable null move pruning
    bool use_aspiration;     // Use aspiration windows
    int history_limit;       // History pruning threshold
    int futility_margin;     // Futility pruning margin
    int lmr_threshold;       // Late move reduction threshold
} SearchParams;

// Search parameters
typedef struct {
    TTEntry* tt;                // Transposition table
    uint32_t tt_size;           // Number of TT entries
    int16_t history[2][64][64]; // History heuristic scores [color][from][to]
    uint16_t killers[2][64];    // Killer moves [2 per ply][max_ply]
    SearchStats stats;          // Search statistics
    int64_t start_time;         // Search start time
    bool stop_search;           // Flag to stop search
    PVLine pv;                  // Principal variation
    SearchParams params;        
} SearchContext;

// Search result
typedef struct {
    uint16_t best_move;     // Best move found
    int score;              // Position evaluation
    PVLine pv;             // Principal variation
    SearchStats stats;      // Search statistics
} SearchResult;

// Core search functions
void init_search(SearchContext* ctx, uint32_t tt_size);
SearchResult search_position(Position* pos, SearchParams* params, SearchContext* ctx);
void cleanup_search(SearchContext* ctx);

// Move generation and ordering
int generate_moves(const Position* pos, SearchMove* moves);
void sort_moves(SearchMove* moves, int count, const SearchContext* ctx, int ply);
int score_move(uint16_t move, const SearchContext* ctx, int ply);

// Search helper functions
int quiescence_search(Position* pos, int alpha, int beta, SearchContext* ctx);
bool is_draw(const Position* pos);
bool should_stop_search(const SearchContext* ctx);

// Utility functions
static inline uint64_t get_hash_key(const Position* pos);
static inline void update_history(SearchContext* ctx, uint16_t move, int depth);
static inline void update_killers(SearchContext* ctx, uint16_t move, int ply);
static inline TTEntry* probe_tt(const SearchContext* ctx, uint64_t key);
static inline void store_tt(SearchContext* ctx, uint64_t key, int score, uint16_t move, 
                          int depth, uint8_t flag);

#endif // SEARCH_OPS_H