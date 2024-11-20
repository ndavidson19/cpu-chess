# High-Performance Chess Engine

A chess engine optimized for the FIDE & Google Efficient Chess AI Challenge, featuring SIMD-accelerated evaluation, C Bindings for Python, magic bitboard move generation, and **No External Dependencies**.

## Features

- SIMD-optimized position evaluation
- Magic bitboard move generation
- Compressed opening book and endgame tablebases
- Phase-aware evaluation
- Advanced search techniques (PVS, LMR, etc.)

## Installation

```bash
pip install cpu-chess
```

## Project Structure
```
cpu-chess/
├── setup.py
├── README.md
├── cpu_chess/
│   ├── __init__.py
│   ├── engine.py            # Main ChessBot class
│   ├── bitboard/
│   │   ├── __init__.py
│   │   ├── bitboard_ops.h   # C header for bitboard operations
│   │   └── bitboard_ops.c   # C implementation of bitboard operations
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py     # Python evaluation logic
│   │   ├── eval_ops.h       # C header for SIMD evaluation
│   │   └── eval_ops.c       # C implementation of SIMD evaluation
│   │   ├── patterns.h       # C header for pattern matching
│   │   └── patterns.c       # C implementation of SIMD pattern matching
│   ├── search/
│   │   ├── __init__.py
│   │   ├── searcher.py      # Python search logic
│   │   ├── search_ops.h     # C header for search operations
│   │   └── search_ops.c     # C implementation of move ordering/search
│   └── utils/
       ├── __init__.py
       └── utils.c  
       └── utils.h      
       └── constants.py      # Shared constants and tables
```

## Quick Start

```python
from chess_bot import ChessBot

# Initialize the bot
bot = ChessBot()

# Get a move for a position
move = bot.select_move(board_fen)
```

## Development Setup

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Verify SIMD support:
```python
from chess_bot.utils import check_simd_support
check_simd_support()  # Prints available SIMD instructions
```

3. Run tests:
```bash
pytest tests/
```

## Kaggle Competition Usage

1. Install the package locally
2. Use the compilation script to generate submission file:
```bash
python -m chess_bot.tools.compile
```
3. Submit the generated `main.py`

## Architecture

### Core Components

1. **SIMD Evaluation**
   - AVX2/AVX-512 piece evaluation
   - Parallel position scoring
   - Vectorized pawn structure analysis

2. **Magic Bitboards**
   - Precomputed attack tables
   - Efficient move generation
   - Sliding piece optimization

3. **Memory Management**
   - Compressed opening book
   - Minimal state representation
   - Efficient transposition table

### Performance Tips

1. **SIMD Optimization**
   - Ensure CPU supports AVX2/AVX-512
   - Align data for optimal SIMD performance
   - Use vectorized operations where possible

2. **Memory Usage**
   - Stay within 5 MiB limit
   - Monitor heap allocations
   - Use bitboards efficiently

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License