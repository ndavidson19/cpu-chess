// common/context.h
#ifndef CONTEXT_H
#define CONTEXT_H

#include "types.h"
#include <stdbool.h>

// Search statistics
typedef struct {
    uint32_t nodes;          // Nodes searched
    uint32_t tt_hits;        // Transposition table hits
    uint32_t beta_cutoffs;   // Beta cutoffs achieved
    uint32_t futile_prunes;  // Positions pruned by futility
    uint32_t null_cuts;      // Null move cutoffs
} SearchStats;

// Principal Variation
typedef struct {
    uint16_t moves[64];      // PV line
    int length;              // Number of moves in line
} PVLine;

// Search parameters
typedef struct {
    int max_depth;           // Maximum search depth
    int64_t time_limit;      // Time limit in milliseconds
    bool use_null_move;      // Enable null move pruning
    bool use_aspiration;     // Use aspiration windows
    int history_limit;       // History pruning threshold
    int futility_margin;     // Futility pruning margin
    int lmr_threshold;       // Late move reduction threshold
} SearchParams;

// Transposition table entry
typedef struct __attribute__((aligned(8), packed)) {
    uint64_t key;      // 8 bytes
    int16_t score;     // 2 bytes
    uint16_t move;     // 2 bytes
    uint8_t depth;     // 1 byte
    uint8_t flag;      // 1 byte
    uint8_t age;       // 1 byte
    uint8_t padding;   // 1 byte for alignment
} TTEntry;             // Total: 16 bytes

// Search context
typedef struct {
    TTEntry* tt;
    uint32_t tt_size;
    uint8_t tt_age;
    int ply;
    int16_t history[2][64][64];
    uint16_t killers[2][64];
    SearchStats stats;
    Position* pos;
    bool stop_search;
    int64_t start_time;
    PVLine pv;
    SearchParams params;
} SearchContext;

#endif