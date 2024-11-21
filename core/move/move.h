// move.h

#ifndef MOVE_H
#define MOVE_H

#include <stdint.h>
#include <stdbool.h>
#include "../common/types.h"
#include "../common/context.h"


// Constants for move encoding
#define MOVE_FROM_SHIFT     0
#define MOVE_TO_SHIFT       6
#define MOVE_PIECE_SHIFT    12
#define MOVE_CAPTURE_SHIFT  16
#define MOVE_PROMO_SHIFT    20
#define MOVE_FLAGS_SHIFT    24

#define MOVE_FROM_MASK      0x3F        // 6 bits
#define MOVE_TO_MASK        0xFC0       // 6 bits shifted
#define MOVE_PIECE_MASK     0xF000      // 4 bits shifted
#define MOVE_CAPTURE_MASK   0xF0000     // 4 bits shifted
#define MOVE_PROMO_MASK     0xF00000    // 4 bits shifted
#define MOVE_FLAGS_MASK     0xF000000   // 4 bits shifted


// Memory-efficient move representation
typedef struct {
    uint16_t move;      // Packed move (from:6|to:6|promotion:4)
    int16_t score;      // Move score for ordering
} SearchMove;


// Function to create a move
static inline uint32_t create_move(int from, int to, int piece, int captured, int promotion, int flags) {
    return (from & 0x3F) |
           ((to & 0x3F) << MOVE_TO_SHIFT) |
           ((piece & 0xF) << MOVE_PIECE_SHIFT) |
           ((captured & 0xF) << MOVE_CAPTURE_SHIFT) |
           ((promotion & 0xF) << MOVE_PROMO_SHIFT) |
           ((flags & 0xF) << MOVE_FLAGS_SHIFT);
}

// Functions to extract move information
static inline int get_move_from(uint32_t move) {
    return (move >> MOVE_FROM_SHIFT) & 0x3F;
}

static inline int get_move_to(uint32_t move) {
    return (move >> MOVE_TO_SHIFT) & 0x3F;
}

static inline int get_move_piece(uint32_t move) {
    return (move >> MOVE_PIECE_SHIFT) & 0xF;
}

static inline int get_move_captured(uint32_t move) {
    return (move >> MOVE_CAPTURE_SHIFT) & 0xF;
}

static inline int get_move_promotion(uint32_t move) {
    return (move >> MOVE_PROMO_SHIFT) & 0xF;
}

static inline int get_move_flags(uint32_t move) {
    return (move >> MOVE_FLAGS_SHIFT) & 0xF;
}

bool in_check(const Position* pos);
bool is_square_attacked(const Position* pos, int square, int color);
void make_move(Position* pos, uint32_t move);
void undo_move(Position* pos);
void make_null_move(Position* pos);
void undo_null_move(Position* pos);
void sort_moves(SearchMove* moves, int count, const SearchContext* ctx, int ply);
int score_move(uint16_t move, const SearchContext* ctx, int ply);
int generate_moves(const Position* pos, SearchMove* moves);
int generate_captures(const Position* pos, SearchMove* moves);
int get_captured_piece(uint16_t move);
int get_move_color(uint16_t move);
int get_piece_value(int piece);
static int get_piece_at(const Position* pos, int square, int color);

#endif // MOVE_H
