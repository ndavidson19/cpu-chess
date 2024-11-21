#include "../common/types.h"
#include "move.h"
#include <string.h>
#include "../bitboard/bitboard_ops.h"
#include "../search/search_ops.h"
#include <string.h>

static const int PIECE_VALUES[6] = {100, 300, 300, 500, 900, 20000}; // P,N,B,R,Q,K

// Keep track of move history for undo
typedef struct {
    uint32_t move;
    uint8_t castling_rights;
    int en_passant_square;
    int halfmove_clock;
    uint64_t hash_key;  // Previous position hash
} MoveUndo;

static MoveUndo move_history[1024];
static int history_count = 0;

void make_move(Position* pos, uint32_t move) {
    // Store state for undo
    MoveUndo* undo = &move_history[history_count++];
    undo->move = move;
    undo->castling_rights = pos->castling_rights;
    undo->en_passant_square = pos->en_passant_square;
    undo->halfmove_clock = pos->halfmove_clock;
    
    int from = get_move_from(move);
    int to = get_move_to(move);
    int piece = get_move_piece(move);
    int captured = get_move_captured(move);
    int promotion = get_move_promotion(move);
    int flags = get_move_flags(move);

    // Update bitboards
    Bitboard from_bb = 1ULL << from;
    Bitboard to_bb = 1ULL << to;

    // Remove piece from source square
    pos->pieces[pos->side_to_move][piece] &= ~from_bb;
    pos->occupied[pos->side_to_move] &= ~from_bb;
    pos->occupied_total &= ~from_bb;

    // Handle captures
    if (captured != NO_PIECE) {
        pos->pieces[1 - pos->side_to_move][captured] &= ~to_bb;
        pos->occupied[1 - pos->side_to_move] &= ~to_bb;
        pos->halfmove_clock = 0;
    }

    // Place piece on destination square
    if (promotion != NO_PIECE) {
        pos->pieces[pos->side_to_move][promotion] |= to_bb;
        pos->halfmove_clock = 0;
    } else {
        pos->pieces[pos->side_to_move][piece] |= to_bb;
    }
    pos->occupied[pos->side_to_move] |= to_bb;
    pos->occupied_total |= to_bb;

    // Update en passant square
    if (piece == PAWN) {
        pos->halfmove_clock = 0;
        if (abs(to - from) == 16) {  // Double pawn push
            pos->en_passant_square = (from + to) / 2;
        } else {
            pos->en_passant_square = -1;
        }
    } else {
        pos->en_passant_square = -1;
        pos->halfmove_clock++;
    }

    // Update castling rights
    if (piece == KING) {
        pos->castling_rights &= ~(3 << (2 * pos->side_to_move));
    } else if (piece == ROOK) {
        if (from == 0 || from == 56)  // Queen-side rook
            pos->castling_rights &= ~(1 << (2 * pos->side_to_move));
        if (from == 7 || from == 63)  // King-side rook
            pos->castling_rights &= ~(2 << (2 * pos->side_to_move));
    }

    // Handle castling moves
    if (piece == KING && abs(to - from) == 2) {
        int rook_from = to > from ? from + 3 : from - 4;
        int rook_to = to > from ? from + 1 : from - 1;
        Bitboard rook_from_bb = 1ULL << rook_from;
        Bitboard rook_to_bb = 1ULL << rook_to;

        pos->pieces[pos->side_to_move][ROOK] &= ~rook_from_bb;
        pos->occupied[pos->side_to_move] &= ~rook_from_bb;
        pos->occupied_total &= ~rook_from_bb;

        pos->pieces[pos->side_to_move][ROOK] |= rook_to_bb;
        pos->occupied[pos->side_to_move] |= rook_to_bb;
        pos->occupied_total |= rook_to_bb;
    }

    // Increment full move number if black moved
    if (pos->side_to_move == BLACK) {
        pos->fullmove_number++;
    }

    // Switch side to move
    pos->side_to_move ^= 1;
}


// Implementation of undo_move
void undo_move(Position* pos) {
    if (history_count == 0) return;

    MoveUndo* undo = &move_history[--history_count];
    uint32_t move = undo->move;

    // Switch side back
    pos->side_to_move ^= 1;

    int from = get_move_from(move);
    int to = get_move_to(move);
    int piece = get_move_piece(move);
    int captured = get_move_captured(move);
    int promotion = get_move_promotion(move);

    // Restore previous state
    pos->castling_rights = undo->castling_rights;
    pos->en_passant_square = undo->en_passant_square;
    pos->halfmove_clock = undo->halfmove_clock;

    // Rest of undo implementation...
    // (Similar to make_move but in reverse)
}

// Implement other functions like make_null_move, undo_null_move, in_check
// In move.c

void make_null_move(Position* pos) {
    MoveUndo* undo = &move_history[history_count++];
    undo->en_passant_square = pos->en_passant_square;
    
    pos->en_passant_square = -1;
    pos->side_to_move ^= 1;
}

void undo_null_move(Position* pos) {
    if (history_count == 0) return;
    
    MoveUndo* undo = &move_history[--history_count];
    pos->en_passant_square = undo->en_passant_square;
    pos->side_to_move ^= 1;
}

bool is_square_attacked(const Position* pos, int square, int attacker_side) {
    Bitboard square_bb = 1ULL << square;

    // Pawn attacks
    if (get_pawn_attacks(square, 1 - attacker_side) & pos->pieces[attacker_side][PAWN])
        return true;

    // Knight attacks
    if (get_knight_attacks(square) & pos->pieces[attacker_side][KNIGHT])
        return true;

    // Bishop/Queen attacks
    Bitboard bishop_attacks = get_bishop_attacks(square, pos->occupied_total);
    if (bishop_attacks & (pos->pieces[attacker_side][BISHOP] | pos->pieces[attacker_side][QUEEN]))
        return true;

    // Rook/Queen attacks
    Bitboard rook_attacks = get_rook_attacks(square, pos->occupied_total);
    if (rook_attacks & (pos->pieces[attacker_side][ROOK] | pos->pieces[attacker_side][QUEEN]))
        return true;

    // King attacks
    if (get_king_attacks(square) & pos->pieces[attacker_side][KING])
        return true;

    return false;
}

bool in_check(const Position* pos) {
    // Find king square
    int king_sq = -1;
    Bitboard king_bb = pos->pieces[pos->side_to_move][KING];
    if (king_bb) {
        king_sq = __builtin_ctzll(king_bb);
    }
    
    if (king_sq == -1) return false;  // No king (shouldn't happen in real game)

    // Check for attacks to king square
    return is_square_attacked(pos, king_sq, 1 - pos->side_to_move);
}

int get_piece_value(int piece) {
    return PIECE_VALUES[piece];
}

int generate_moves(const Position* pos, SearchMove* moves) {
    int count = 0;
    const int side = pos->side_to_move;
    
    // Generate moves for each piece type
    for (int piece = 0; piece < 6; piece++) {
        Bitboard pieces = pos->pieces[side][piece];
        
        while (pieces) {
            const int from = __builtin_ctzll(pieces);
            Bitboard attacks;
            
            // Get appropriate attack bitboard based on piece type
            switch (piece) {
                case PAWN:
                    attacks = get_pawn_attacks(from, side);
                    break;
                case KNIGHT:
                    attacks = get_knight_attacks(from);
                    break;
                case BISHOP:
                    attacks = get_bishop_attacks(from, pos->occupied_total);
                    break;
                case ROOK:
                    attacks = get_rook_attacks(from, pos->occupied_total);
                    break;
                case QUEEN:
                    attacks = get_queen_attacks(from, pos->occupied_total);
                    break;
                case KING:
                    attacks = get_king_attacks(from);
                    break;
            }
            
            // Remove friendly pieces from attack targets
            attacks &= ~pos->occupied[side];
            
            // Generate moves for all attack targets
            while (attacks) {
                const int to = __builtin_ctzll(attacks);
                const int captured = (pos->occupied[!side] & (1ULL << to)) ? 1 : 0;
                moves[count].move = create_move(from, to, piece, captured, 0, 0);
                moves[count].score = 0;
                count++;
                attacks &= (attacks - 1);
            }
            
            pieces &= (pieces - 1);
        }
    }
    
    return count;
}

int generate_captures(const Position* pos, SearchMove* moves) {
    int count = 0;
    const int side = pos->side_to_move;
    
    // Only generate moves that capture enemy pieces
    for (int piece = 0; piece < 6; piece++) {
        Bitboard pieces = pos->pieces[side][piece];
        
        while (pieces) {
            const int from = __builtin_ctzll(pieces);
            Bitboard attacks;
            
            switch (piece) {
                case PAWN:
                    attacks = get_pawn_attacks(from, side);
                    break;
                case KNIGHT:
                    attacks = get_knight_attacks(from);
                    break;
                case BISHOP:
                    attacks = get_bishop_attacks(from, pos->occupied_total);
                    break;
                case ROOK:
                    attacks = get_rook_attacks(from, pos->occupied_total);
                    break;
                case QUEEN:
                    attacks = get_queen_attacks(from, pos->occupied_total);
                    break;
                case KING:
                    attacks = get_king_attacks(from);
                    break;
            }
            
            // Only keep attacks on enemy pieces
            attacks &= pos->occupied[!side];
            
            while (attacks) {
                const int to = __builtin_ctzll(attacks);
                const int captured = get_piece_at(pos, to, !side);
                moves[count].move = create_move(from, to, piece, captured, 0, 0);
                moves[count].score = 0;
                count++;
                attacks &= (attacks - 1);
            }
            
            pieces &= (pieces - 1);
        }
    }
    
    return count;
}

int get_captured_piece(uint16_t move) {
    return get_move_captured(move);
}

int get_move_color(uint16_t move) {
    return get_move_piece(move) >> 3;  // Color is encoded in high bit of piece
}

// Helper to get piece type at a square
static int get_piece_at(const Position* pos, int square, int color) {
    Bitboard bb = 1ULL << square;
    for (int piece = 0; piece < 6; piece++) {
        if (pos->pieces[color][piece] & bb) {
            return piece;
        }
    }
    return -1;
}

void sort_moves(SearchMove* moves, int count, const SearchContext* ctx, int ply) {
    // Score moves
    for (int i = 0; i < count; i++) {
        moves[i].score = score_move(moves[i].move, ctx, ply);
    }
    
    // Simple insertion sort
    for (int i = 1; i < count; i++) {
        SearchMove temp = moves[i];
        int j = i - 1;
        while (j >= 0 && moves[j].score < temp.score) {
            moves[j + 1] = moves[j];
            j--;
        }
        moves[j + 1] = temp;
    }
}

int score_move(uint16_t move, const SearchContext* ctx, int ply) {
    int score = 0;
    
    // Hash move gets highest priority
    TTEntry* tt_entry = probe_tt(ctx, get_hash_key(ctx->pos));
    if (tt_entry && tt_entry->move == move) {
        return 30000;
    }
    
    // Killer moves
    if (move == ctx->killers[0][ply]) return 29000;
    if (move == ctx->killers[1][ply]) return 28000;
    
    const int piece = get_move_piece(move);
    const int captured = get_move_captured(move);
    
    // MVV/LVA scoring for captures
    if (captured != -1) {
        score = get_piece_value(captured) * 10 - piece;
    }
    
    // History heuristic
    const int from = get_move_from(move);
    const int to = get_move_to(move);
    const int color = get_move_color(move);
    score += ctx->history[color][from][to] / 100;
    
    return score;
}