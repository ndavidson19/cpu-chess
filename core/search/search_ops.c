#include "search_ops.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include "../evaluation/eval_ops.h"
#include "../utils/utils.h"
#include "../move/move.h"

// Debugging
#define DEBUG_LOG(msg, ...) \
    fprintf(stderr, "[DEBUG] %s:%d - " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    fflush(stderr)

// Constants for search
#define INFINITY_SCORE 32000
#define MATE_SCORE 31000
#define DRAW_SCORE 0
#define NULL_MOVE_REDUCTION 3
#define FUTILITY_MARGIN 100
#define DELTA_MARGIN 200
#define LMR_THRESHOLD 4
#define HISTORY_MAX 16384

// TT flags
#define TT_EXACT 0
#define TT_ALPHA 1
#define TT_BETA 2

uint64_t piece_keys[2][6][64];
uint64_t castling_keys[16];
uint64_t en_passant_keys[64];
uint64_t side_key;


void init_hash_keys() {
   // Initialize using a PRNG seed
   uint64_t seed = 0x123456789ABCDEF0ULL;
   
   for (int c = 0; c < 2; c++) {
       for (int p = 0; p < 6; p++) {
           for (int s = 0; s < 64; s++) {
               seed ^= seed >> 12;
               seed ^= seed << 25;
               seed ^= seed >> 27;
               piece_keys[c][p][s] = seed * 2685821657736338717ULL;
           }
       }
   }

   for (int i = 0; i < 16; i++) {
       seed ^= seed >> 12;
       seed ^= seed << 25;
       seed ^= seed >> 27;
       castling_keys[i] = seed * 2685821657736338717ULL;
   }

   for (int i = 0; i < 64; i++) {
       seed ^= seed >> 12;
       seed ^= seed << 25; 
       seed ^= seed >> 27;
       en_passant_keys[i] = seed * 2685821657736338717ULL;
   }

   seed ^= seed >> 12;
   seed ^= seed << 25;
   seed ^= seed >> 27;
   side_key = seed * 2685821657736338717ULL;
}

// Helper for aligned size calculation
static size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

void init_search(SearchContext* ctx, uint32_t tt_size) {
    DEBUG_LOG("Initializing search context with tt_size=%u", tt_size);
    
    if (!ctx) {
        DEBUG_LOG("NULL context pointer");
        return;
    }
    
    // Validate and adjust TT size
    if (tt_size < MIN_TT_SIZE) {
        tt_size = MIN_TT_SIZE;
        DEBUG_LOG("TT size adjusted to minimum: %u", tt_size);
    }
    if (tt_size > MAX_TT_SIZE) {
        tt_size = MAX_TT_SIZE;
        DEBUG_LOG("TT size adjusted to maximum: %u", tt_size);
    }
    
    // Zero out context first
    memset(ctx, 0, sizeof(SearchContext));
    
    // Initialize hash keys first
    DEBUG_LOG("Initializing hash keys");
    init_hash_keys();
    
    // Calculate aligned size
    size_t entry_size = sizeof(TTEntry);
    size_t total_size = (size_t)tt_size * entry_size;
    size_t aligned_size = (total_size + TT_ALIGNMENT - 1) & ~(size_t)(TT_ALIGNMENT - 1);
    
    DEBUG_LOG("TT allocation details: entry_size=%zu total_size=%zu aligned_size=%zu",
              entry_size, total_size, aligned_size);
    
    // Use aligned allocation
    void* tt_mem = NULL;
    int alloc_result = posix_memalign(&tt_mem, TT_ALIGNMENT, aligned_size);
    
    if (alloc_result != 0) {
        DEBUG_LOG("TT allocation failed: %s", strerror(errno));
        return;
    }
    
    if (!tt_mem) {
        DEBUG_LOG("TT allocation returned NULL");
        return;
    }
    
    DEBUG_LOG("TT allocated at %p", tt_mem);
    
    // Zero the allocated memory
    memset(tt_mem, 0, aligned_size);
    
    // Set up context
    ctx->tt = (TTEntry*)tt_mem;
    ctx->tt_size = tt_size;
    ctx->tt_age = 0;
    
    // Initialize other fields
    ctx->ply = 0;
    ctx->pos = NULL;
    ctx->stop_search = false;
    ctx->start_time = 0;
    
    memset(ctx->history, 0, sizeof(ctx->history));
    memset(ctx->killers, 0, sizeof(ctx->killers));
    memset(&ctx->stats, 0, sizeof(SearchStats));
    memset(&ctx->pv, 0, sizeof(PVLine));
    
    // Initialize search parameters with defaults
    ctx->params.max_depth = 6;
    ctx->params.time_limit = 0;
    ctx->params.use_null_move = true;
    ctx->params.use_aspiration = true;
    ctx->params.history_limit = 8192;
    ctx->params.futility_margin = 100;
    ctx->params.lmr_threshold = 3;
    
    DEBUG_LOG("Search context initialized successfully: tt=%p size=%u", 
              (void*)ctx->tt, ctx->tt_size);
}


void store_tt(SearchContext* ctx, uint64_t key, int score, uint16_t move, 
                          int depth, uint8_t flag) {
    // Get index in transposition table (typically using lower bits of hash)
    const size_t index = key & (ctx->tt_size - 1);  // Assumes tt_size is power of 2
    TTEntry* entry = &ctx->tt[index];
    
    // Always replace if:
    // 1. Empty slot (key == 0)
    // 2. Current position has greater depth
    // 3. Current entry is old (from previous search)
    // 4. Same position but current search is deeper
    if (entry->key == 0 || 
        depth >= entry->depth || 
        entry->age != ctx->tt_age ||
        entry->key == key) {
        
        // Handle mate scores before storing
        if (score > MATE_IN_MAX) {
            score += ctx->ply;
        } else if (score < -MATE_IN_MAX) {
            score -= ctx->ply;
        }
        
        // Update the entry
        entry->key = key;
        entry->move = move;
        entry->score = score;
        entry->depth = depth;
        entry->flag = flag;
        entry->age = ctx->tt_age;
    }
}

uint64_t get_hash_key(const Position* pos) {
    uint64_t hash = 0;
    for (int color = 0; color < 2; color++) {
        for (int piece = 0; piece < 6; piece++) {
            uint64_t pieces = pos->pieces[color][piece];
            while (pieces) {
                int square = __builtin_ctzll(pieces);
                hash ^= piece_keys[color][piece][square];
                pieces &= (pieces - 1);
            }
        }
    }
    
    hash ^= castling_keys[pos->castling_rights];
    if (pos->en_passant_square >= 0) {
        hash ^= en_passant_keys[pos->en_passant_square];
    }
    if (pos->side_to_move) {
        hash ^= side_key;
    }
    
    return hash;
}

TTEntry* probe_tt(const SearchContext* ctx, uint64_t key) {
    const size_t index = key & (ctx->tt_size - 1);
    return &ctx->tt[index];
}

// Helper function to get piece bitboards (implement based on your Position structure)
static Bitboard get_piece_bitboard(Position* pos, int color, int piece) {
    // Implementation depends on your Position structure
    return pos->pieces[color][piece];
}

// Helper function to get color occupancy
static Bitboard get_color_occupancy(Position* pos, int color) {
    Bitboard occupancy = 0ULL;
    for (int piece = 0; piece < 6; piece++) {
        occupancy |= pos->pieces[color][piece];
    }
    return occupancy;
}

// Create and initialize an EvalContext from a SearchContext
EvalContext* create_eval_context_from_search(SearchContext* search_ctx) {
    DEBUG_LOG("Enter create_eval_context_from_search: search_ctx=%p", (void*)search_ctx);
    
    // Validate search context and position
    if (!search_ctx) {
        DEBUG_LOG("Error: search_ctx is NULL");
        return NULL;
    }
    DEBUG_LOG("Search context validation passed");
    
    DEBUG_LOG("Checking position pointer: pos=%p", (void*)search_ctx->pos);
    if (!search_ctx->pos) {
        DEBUG_LOG("Error: search_ctx->pos is NULL");
        return NULL;
    }
    DEBUG_LOG("Position pointer validation passed");

    // Memory allocation
    DEBUG_LOG("Attempting to allocate EvalContext: size=%zu", sizeof(EvalContext));
    EvalContext* eval_ctx = (EvalContext*)calloc(1, sizeof(EvalContext));
    if (!eval_ctx) {
        DEBUG_LOG("Error: Failed to allocate EvalContext");
        return NULL;
    }
    DEBUG_LOG("EvalContext allocated successfully at %p", (void*)eval_ctx);

    // Set position pointer
    Position* pos = search_ctx->pos;
    eval_ctx->pos = pos;
    DEBUG_LOG("Position pointer set in eval_ctx: %p", (void*)eval_ctx->pos);

    // Initialize basic fields first
    DEBUG_LOG("Initializing basic fields");
    eval_ctx->stage = 0;
    eval_ctx->turn = pos->side_to_move;
    eval_ctx->castling_rights = pos->castling_rights;
    DEBUG_LOG("Basic fields initialized: turn=%d castling_rights=%d", 
              eval_ctx->turn, eval_ctx->castling_rights);

    // Initialize bitboard arrays to zero first
    DEBUG_LOG("Zeroing bitboard arrays");
    memset(eval_ctx->pieces, 0, sizeof(eval_ctx->pieces));
    memset(eval_ctx->attacks, 0, sizeof(eval_ctx->attacks));
    memset(eval_ctx->occupied, 0, sizeof(eval_ctx->occupied));
    memset(eval_ctx->all_attacks, 0, sizeof(eval_ctx->all_attacks));
    DEBUG_LOG("Bitboard arrays zeroed");

    // Now try to copy bitboards
    DEBUG_LOG("Starting bitboard initialization");
    for (int color = 0; color < 2; color++) {
        DEBUG_LOG("Processing color %d", color);
        for (int piece = 0; piece < 6; piece++) {
            DEBUG_LOG("Accessing piece bitboard: color=%d piece=%d", color, piece);
            // Add bounds checking for array access
            if (color >= 0 && color < 2 && piece >= 0 && piece < 6) {
                Bitboard bb = pos->pieces[color][piece];
                eval_ctx->pieces[color][piece] = bb;
                eval_ctx->occupied[color] |= bb;
                DEBUG_LOG("Copied bitboard[%d][%d] = %llx", color, piece, bb);
            } else {
                DEBUG_LOG("Invalid color/piece index: color=%d piece=%d", color, piece);
                free(eval_ctx);
                return NULL;
            }
        }
    }

    // Set occupied bitboard
    eval_ctx->all_occupied = eval_ctx->occupied[0] | eval_ctx->occupied[1];
    DEBUG_LOG("All occupied bitboard set: %llx", eval_ctx->all_occupied);

    // Initialize space control and center control to zero
    DEBUG_LOG("Initializing control bitboards");
    for (int color = 0; color < 2; color++) {
        eval_ctx->space_control[color] = 0ULL;
        eval_ctx->center_control[color] = 0ULL;
    }

    // Zero out pawn structure bitboards
    DEBUG_LOG("Initializing pawn structure bitboards");
    for (int color = 0; color < 2; color++) {
        eval_ctx->pawn_shields[color] = 0ULL;
        eval_ctx->passed_pawns[color] = 0ULL;
        eval_ctx->pawn_chains[color] = 0ULL;
    }

    DEBUG_LOG("EvalContext initialization complete: %p", (void*)eval_ctx);
    return eval_ctx;
}

// Main alpha-beta search with modern pruning techniques
static int alpha_beta(Position* pos, int alpha, int beta, int depth, int ply, 
                     SearchContext* ctx, PVLine* pv) {
    DEBUG_LOG("Enter alpha_beta: depth=%d ply=%d alpha=%d beta=%d", depth, ply, alpha, beta);

    if (!ctx || !pos || !pv) {
        DEBUG_LOG("Null pointer in alpha_beta: ctx=%p pos=%p pv=%p", (void*)ctx, (void*)pos, (void*)pv);
        return 0;
    }

    DEBUG_LOG("Checking position state: side_to_move=%d", pos->side_to_move);

    DEBUG_LOG("Creating eval context: pos=%p ctx=%p", (void*)pos, (void*)ctx);

    EvalContext* eval_ctx = create_eval_context_from_search(ctx);
    if (!eval_ctx) return 0; // Handle allocation failure
    DEBUG_LOG("Eval context created successfully: eval_ctx=%p", (void*)eval_ctx);

    // Check time and node limits
    if (should_stop_search(ctx)) return 0;
    
    // Initialize PV length
    pv->length = 0;
    
    // Check for draws
    if (is_draw(pos)) return DRAW_SCORE;
    
    // Mate distance pruning
    alpha = alpha < -MATE_SCORE + ply ? alpha : -MATE_SCORE + ply;
    beta = beta > MATE_SCORE - ply ? beta : MATE_SCORE - ply;
    if (alpha >= beta) return alpha;
    
    // Transposition table lookup
    TTEntry* tt_entry = probe_tt(ctx, get_hash_key(pos));
    if (tt_entry && tt_entry->depth >= depth) {
        ctx->stats.tt_hits++;
        if (tt_entry->flag == TT_EXACT) return tt_entry->score;
        if (tt_entry->flag == TT_ALPHA && tt_entry->score <= alpha) return alpha;
        if (tt_entry->flag == TT_BETA && tt_entry->score >= beta) return beta;
    }
    
    // Quiescence search at horizon
    if (depth <= 0) {
        return quiescence_search(pos, alpha, beta, ctx);
    }
    
    // Null move pruning
    if (ctx->params.use_null_move && depth >= 3 && !in_check(pos)) {
        make_null_move(pos);
        int score = -alpha_beta(pos, -beta, -beta + 1, depth - NULL_MOVE_REDUCTION - 1,
                              ply + 1, ctx, pv);
        undo_null_move(pos);
        
        if (score >= beta) {
            ctx->stats.null_cuts++;
            return beta;
        }
    }
    
    // Initialize
    int best_score = -INFINITY_SCORE;
    uint16_t best_move = 0;
    SearchMove moves[256];
    int move_count = generate_moves(pos, moves);
    PVLine new_pv;
    
    DEBUG_LOG("Starting position evaluation");

    // Get static evaluation for pruning decisions
    int static_eval = evaluate_position(eval_ctx);
    // Clean up
    free(eval_ctx);

    // Sort moves
    sort_moves(moves, move_count, ctx, ply);
    
    // Loop through moves
    for (int i = 0; i < move_count; i++) {
        // Late move reductions (LMR)
        int new_depth = depth - 1;
        if (depth >= LMR_THRESHOLD && i >= 3 && !in_check(pos)) {
            new_depth--;
        }
        
        // Futility pruning
        if (depth == 1 && !in_check(pos) && static_eval + ctx->params.futility_margin <= alpha) {
            ctx->stats.futile_prunes++;
            continue;
        }
        
        // History pruning
        if (depth <= 3 && i > 0 && 
            score_move(moves[i].move, ctx, ply) < ctx->params.history_limit) {
            continue;
        }
        
        // Make move
        make_move(pos, moves[i].move);
        ctx->stats.nodes++;
        
        // Recursive search
        int score;
        if (i == 0) {
            // Search first move with full window
            score = -alpha_beta(pos, -beta, -alpha, new_depth, ply + 1, ctx, &new_pv);
        } else {
            // Late move reductions and null window search
            score = -alpha_beta(pos, -alpha - 1, -alpha, new_depth, ply + 1, ctx, &new_pv);
            if (score > alpha && score < beta) {
                // Research with full window if the move might be good
                score = -alpha_beta(pos, -beta, -alpha, depth - 1, ply + 1, ctx, &new_pv);
            }
        }
        
        // Undo move
        undo_move(pos);
        
        // Update best score
        if (score > best_score) {
            best_score = score;
            best_move = moves[i].move;
            
            // Update PV
            pv->moves[0] = moves[i].move;
            memcpy(pv->moves + 1, new_pv.moves, new_pv.length * sizeof(uint16_t));
            pv->length = new_pv.length + 1;
            
            // Update alpha
            if (score > alpha) {
                alpha = score;
                if (alpha >= beta) {
                    // Beta cutoff
                    ctx->stats.beta_cutoffs++;
                    update_history(ctx, moves[i].move, depth);
                    update_killers(ctx, moves[i].move, ply);
                    break;
                }
            }
        }
    }
    
    // Store position in transposition table
    uint8_t tt_flag = best_score <= alpha ? TT_ALPHA :
                     best_score >= beta ? TT_BETA : TT_EXACT;
    store_tt(ctx, get_hash_key(pos), best_score, best_move, depth, tt_flag);
    
    return best_score;
}

// Quiescence search to handle tactical positions
int quiescence_search(Position* pos, int alpha, int beta, SearchContext* ctx) {
    EvalContext* eval_ctx = create_eval_context_from_search(ctx);
    if (!eval_ctx) return 0;

    ctx->stats.nodes++;
    
    // Get static evaluation
    int stand_pat = evaluate_position(eval_ctx);
    free(eval_ctx);

    // Return if we're way ahead
    if (stand_pat >= beta) return beta;
    
    // Update alpha if static eval is better
    if (stand_pat > alpha) alpha = stand_pat;
    
    // Generate and score captures
    SearchMove moves[256];
    int move_count = generate_captures(pos, moves);
    sort_moves(moves, move_count, ctx, 0);
    
    // Loop through captures
    for (int i = 0; i < move_count; i++) {
        // Delta pruning
        if (stand_pat + get_piece_value(get_captured_piece(moves[i].move)) + DELTA_MARGIN < alpha) {
            continue;
        }
        
        // Make capture
        make_move(pos, moves[i].move);
        
        // Recursive search
        int score = -quiescence_search(pos, -beta, -alpha, ctx);
        
        // Undo capture
        undo_move(pos);
        
        // Update alpha
        if (score > alpha) {
            alpha = score;
            if (alpha >= beta) return beta;
        }
    }
    
    return alpha;
}

// Main search function
SearchResult search_position(Position* pos, SearchParams* params, SearchContext* ctx) {
    SearchResult result = {0};

    // Initialize search if not already done
    if (!ctx->tt) {
        DEBUG_LOG("Initializing search context");
        init_search(ctx, 1024 * 1024);  // 1MB default table size
    }

    ctx->start_time = get_current_time();
    ctx->stop_search = false;
    ctx->params = *params;
        // Validate input parameters
    if (!pos || !params || !ctx) {
        DEBUG_LOG("Null pointer passed to search_position");
        return result;
    }
    
    DEBUG_LOG("Starting search with max_depth=%d", params->max_depth);
    DEBUG_LOG("Context initialized: start_time=%lld", ctx->start_time);

    // Iterative deepening
    int depth = 1;
    int alpha = -INFINITY_SCORE;
    int beta = INFINITY_SCORE;
    
    while (depth <= params->max_depth && !should_stop_search(ctx)) {
        // Aspiration windows
        if (depth >= 4 && params->use_aspiration) {
            alpha = result.score - 50;
            beta = result.score + 50;
            DEBUG_LOG("Using aspiration window: alpha=%d beta=%d", alpha, beta);

        }
        
        // Search with current window
        PVLine pv = {0};
        DEBUG_LOG("Starting alpha-beta search at depth %d", depth);
        int score = alpha_beta(pos, alpha, beta, depth, 0, ctx, &pv);
        
        // Handle aspiration window failures
        if (score <= alpha || score >= beta) {
            DEBUG_LOG("Aspiration window failed, retrying with full window");
            alpha = -INFINITY_SCORE;
            beta = INFINITY_SCORE;
            score = alpha_beta(pos, alpha, beta, depth, 0, ctx, &pv);
        }
        
        // Update result if search completed
        if (!ctx->stop_search) {
            result.best_move = pv.moves[0];
            result.score = score;
            result.pv = pv;
            result.stats = ctx->stats;
        }
        
        depth++;
    }
    
    return result;
}



// Utility functions
void update_history(SearchContext* ctx, uint16_t move, int depth) {
    int from = get_move_from(move);
    int to = get_move_to(move);
    int color = get_move_color(move);
    
    ctx->history[color][from][to] += depth * depth;
    if (ctx->history[color][from][to] > HISTORY_MAX) {
        // Age history scores periodically
        for (int c = 0; c < 2; c++)
            for (int f = 0; f < 64; f++)
                for (int t = 0; t < 64; t++)
                    ctx->history[c][f][t] /= 2;
    }
}

void update_killers(SearchContext* ctx, uint16_t move, int ply) {
    if (move != ctx->killers[0][ply]) {
        ctx->killers[1][ply] = ctx->killers[0][ply];
        ctx->killers[0][ply] = move;
    }
}

// Cleanup
void cleanup_search(SearchContext* ctx) {
    if (ctx->tt) {
        free(ctx->tt);
        ctx->tt = NULL;
    }
}