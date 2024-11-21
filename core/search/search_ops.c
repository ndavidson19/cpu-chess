#include "search_ops.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../evaluation/eval_ops.h"
#include "../utils/utils.h"
#include "../move/move.h"

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

// Initialize search context
void init_search(SearchContext* ctx, uint32_t tt_size) {
    // Initialize hash keys
    init_hash_keys();
    ctx->tt = (TTEntry*)aligned_alloc(32, tt_size * sizeof(TTEntry));
    ctx->tt_size = tt_size;
    memset(ctx->tt, 0, tt_size * sizeof(TTEntry));
    memset(ctx->history, 0, sizeof(ctx->history));
    memset(ctx->killers, 0, sizeof(ctx->killers));
    memset(&ctx->stats, 0, sizeof(SearchStats));
    ctx->stop_search = false;
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

// Main alpha-beta search with modern pruning techniques
static int alpha_beta(Position* pos, int alpha, int beta, int depth, int ply, 
                     SearchContext* ctx, PVLine* pv) {
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
    
    // Get static evaluation for pruning decisions
    int static_eval = evaluate_position(pos);
    
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
    ctx->stats.nodes++;
    
    // Get static evaluation
    int stand_pat = evaluate_position(pos);
    
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
    ctx->start_time = get_current_time();
    ctx->stop_search = false;
    ctx->params = *params;
    
    // Iterative deepening
    int depth = 1;
    int alpha = -INFINITY_SCORE;
    int beta = INFINITY_SCORE;
    
    while (depth <= params->max_depth && !should_stop_search(ctx)) {
        // Aspiration windows
        if (depth >= 4 && params->use_aspiration) {
            alpha = result.score - 50;
            beta = result.score + 50;
        }
        
        // Search with current window
        PVLine pv = {0};
        int score = alpha_beta(pos, alpha, beta, depth, 0, ctx, &pv);
        
        // Handle aspiration window failures
        if (score <= alpha || score >= beta) {
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