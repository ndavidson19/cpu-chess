// types.h
#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

// Special values
#define NO_PIECE -1

// Colors
#define WHITE   0
#define BLACK   1

// Castling rights
#define CASTLE_WK   0x1
#define CASTLE_WQ   0x2
#define CASTLE_BK   0x4
#define CASTLE_BQ   0x8

// Maximum ply for move history
#define MAX_PLY 1024

#define ALIGN_32 __attribute__((aligned(32)))
typedef uint64_t Bitboard;

// Define Piece Types
typedef enum {
    PAWN = 0,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    NUM_PIECE_TYPES // Helps in array sizes
} PieceType;


typedef struct Position {
    uint64_t pieces[2][6] ALIGN_32;    // Piece bitboards [color][piece type]
    uint64_t occupied[2] ALIGN_32;      // Color-wise occupancy
    uint64_t occupied_total;            // Total occupancy
    int side_to_move;                   // Current side to move (WHITE/BLACK)
    uint8_t castling_rights;            // Castling availability
    int en_passant_square;              // En passant target square
    int halfmove_clock;                 // Halfmove clock for 50-move rule
    int fullmove_number;                // Fullmove number
    
    // New fields for draw detection
    int repetition_count;               // Number of position repetitions
    int fifty_move_count;               // Counter for fifty move rule
    uint64_t position_hash;             // Zobrist hash of the position
    
    // Previous state for move undo
    struct {
        uint8_t castling_rights;
        int en_passant_square;
        int halfmove_clock;
        uint64_t position_hash;
        uint64_t captured_piece;
    } history[MAX_PLY];                 // Move history stack
    int history_ply;                    // Current ply in history
} Position;

#endif