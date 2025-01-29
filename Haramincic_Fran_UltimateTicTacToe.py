"""
Code creating the Ultimate Tic-tac-toe game.
Done as part of the Artificial Inteligence course at the University of Zagreb, Faculty of Mechanical Engineering and Naval Architecture.
Made by: Fran Haraminčić
"""
import numpy as np
import tkinter as tk
from tkinter import ttk
from typing import Tuple, Optional, List
import time
import copy

# Class defining the logic of Ultimate tic-tac-toe
class UltimateTicTacToe:
    def __init__(self):
        """Initializes a new game with empty boards."""
        self.big_board = np.zeros((3, 3), dtype=int) # main 3x3 board
        self.small_boards = np.zeros((3, 3, 3, 3), dtype=int) # 9 smaller boards
        self.current_player = 1
        self.next_board = None
        
    def is_valid_move(self, big_pos: Tuple[int, int], small_pos: Tuple[int, int]) -> bool:
        """Checks if the move is valid according to game logic."""
        bx, by = big_pos
        sx, sy = small_pos
        
        if not (0 <= bx < 3 and 0 <= by < 3 and 0 <= sx < 3 and 0 <= sy < 3):
            return False
            
        if self.next_board is not None and (bx, by) != self.next_board:
            return False
            
        if self.big_board[bx, by] != 0:
            return False
            
        return self.small_boards[bx, by, sx, sy] == 0
        
    def make_move(self, big_pos: Tuple[int, int], small_pos: Tuple[int, int]) -> bool:
        """Makes the move, updates game state and determins next playable board."""
        if not self.is_valid_move(big_pos, small_pos):
            return False
            
        bx, by = big_pos
        sx, sy = small_pos
        
        self.small_boards[bx, by, sx, sy] = self.current_player
        
        if self.check_winner(self.small_boards[bx, by]):
            self.big_board[bx, by] = self.current_player
        elif self.is_board_full(self.small_boards[bx, by]):
            self.big_board[bx, by] = -1
            
        if self.big_board[sx, sy] == 0:
            self.next_board = (sx, sy)
        else:
            self.next_board = None
            
        self.current_player = 3 - self.current_player
        return True
        
    def check_winner(self, board) -> bool:
        """Checks if the game is won."""
        for i in range(3):
            if np.all(board[i, :] == board[i, 0]) and board[i, 0] != 0:
                return True
            if np.all(board[:, i] == board[0, i]) and board[0, i] != 0:
                return True
        
        if np.all(np.diag(board) == board[0, 0]) and board[0, 0] != 0:
            return True
        if np.all(np.diag(np.fliplr(board)) == board[0, 2]) and board[0, 2] != 0:
            return True
            
        return False
        
    def is_board_full(self, board) -> bool:
        """Checks if the whole board is full (in case of a draw)."""
        return np.all(board != 0)
        
    def get_valid_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Returns a list containing all available moves."""
        valid_moves = []
        if self.next_board is None:
            for bx in range(3):
                for by in range(3):
                    if self.big_board[bx, by] == 0:
                        for sx in range(3):
                            for sy in range(3):
                                if self.small_boards[bx, by, sx, sy] == 0:
                                    valid_moves.append(((bx, by), (sx, sy)))
        else:
            bx, by = self.next_board
            if self.big_board[bx, by] == 0:
                for sx in range(3):
                    for sy in range(3):
                        if self.small_boards[bx, by, sx, sy] == 0:
                            valid_moves.append(((bx, by), (sx, sy)))
        
        return valid_moves
        
    def get_game_state(self) -> int:
        """Returns the current state of the game."""
        if self.check_winner(self.big_board):
            return 3 - self.current_player # there is a winner
        if self.is_board_full(self.big_board):
            return 0 # game is a draw
        return -1 # game ongoing

# Class containing the logic for the AI player
class AIPlayer:
    DIFFICULTY_DEPTHS = {
        "Easy": 2,
        "Medium": 4,
        "Hard": 7
    }
    
    # Scoring weights for different difficulty levels
    DIFFICULTY_WEIGHTS = {
        "Easy": {
            "win": 1000,
            "near_win": 100,
            "potential": 10,
            "center": 5,
            "corner": 3
        },
        "Medium": {
            "win": 5000,
            "near_win": 200,
            "potential": 20,
            "center": 8,
            "corner": 5
        },
        "Hard": {
            "win": 10000,
            "near_win": 500,
            "potential": 50,
            "center": 15,
            "corner": 10
        }
    }
    
    MAX_MOVE_TIME = 30  # Maximum time in seconds for AI to make a move
    
    def __init__(self, difficulty="Medium"): # default difficulty is medium
        """Initializes the AI player based on the chosen difficulty."""
        self.difficulty = difficulty
        self.set_difficulty(difficulty)
        print(f"Game started on {difficulty} difficulty")
        
    def set_difficulty(self, difficulty):
        """If new difficulty is chosen sets it as active."""
        self.difficulty = difficulty
        self.max_depth = self.DIFFICULTY_DEPTHS.get(difficulty, 4)
        self.weights = self.DIFFICULTY_WEIGHTS.get(difficulty)
        print(f"AI difficulty changed to: {difficulty}")
        
    def evaluate_board(self, board) -> int:
        """Board evaluation function with difficulty-based weighting."""
        score = 0
        weights = self.weights # weights for the active difficulty
        
        # Check for wins
        if np.all(board == 1):
            return weights["win"]
        elif np.all(board == 2):
            return -weights["win"]
        
        # Evaluate rows, columns, and diagonals
        for i in range(3):
            # Rows
            score += self._evaluate_line(board[i, :])
            # Columns
            score += self._evaluate_line(board[:, i])
        
        # Diagonals
        score += self._evaluate_line(np.diag(board))
        score += self._evaluate_line(np.diag(np.fliplr(board)))
        
        # Strategic positions
        # Center control
        if board[1, 1] == 1:
            score += weights["center"]
        elif board[1, 1] == 2:
            score -= weights["center"]
        
        # Corner control
        corners = [board[0, 0], board[0, 2], board[2, 0], board[2, 2]]
        for corner in corners:
            if corner == 1:
                score += weights["corner"]
            elif corner == 2:
                score -= weights["corner"]
        
        return score
        
    def _evaluate_line(self, line) -> int:
        """Evaluate a single line (row, column, diagonal) with difficulty-based weighting."""
        score = 0
        weights = self.weights
        
        x_count = np.sum(line == 1) # Counts all the Xs
        o_count = np.sum(line == 2) # Counts all the Os
        empty_count = np.sum(line == 0) # Counts all the empty positions
        
        # Near wins
        if x_count == 2 and empty_count == 1:
            score += weights["near_win"]
        elif o_count == 2 and empty_count == 1:
            score -= weights["near_win"]
        
        # Potential lines
        if x_count == 1 and empty_count == 2:
            score += weights["potential"]
        elif o_count == 1 and empty_count == 2:
            score -= weights["potential"]
        
        # Blocking opponent's potential lines
        if self.difficulty == "Hard":
            if o_count == 2:
                score -= weights["near_win"] * 1.5  # Prioritize blocking wins
        
        return score
        
    def minimax(self, game: UltimateTicTacToe, depth: int, alpha: float, beta: float, 
                maximizing_player: bool, start_time: float) -> Tuple[int, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Implemented minimax algorithm with alpha-beta pruning for finding the best move."""
        # Time limit check
        if time.time() - start_time > self.MAX_MOVE_TIME - 0.1:
            return self.evaluate_board(game.big_board), None
        
        game_state = game.get_game_state()
        # If game is still going call evaluate_board
        if game_state != -1: 
            return self.evaluate_board(game.big_board), None
        
        # Mimimax at the top of the decision tree
        if depth == 0:
            return self.evaluate_board(game.big_board), None
        
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return 0, None
        
        # Dynamic move ordering based on preliminary evaluation
        moves_with_eval = []
        for move in valid_moves:
            game_copy = copy.deepcopy(game)
            game_copy.make_move(*move)
            eval = self.evaluate_board(game_copy.big_board)
            
            # Add position-based bonus for move ordering
            big_pos, small_pos = move
            if small_pos == (1, 1):  # Center
                eval += self.weights["center"]
            elif small_pos in [(0, 0), (0, 2), (2, 0), (2, 2)]:  # Corners
                eval += self.weights["corner"]
            
            moves_with_eval.append((eval, move))
        
        # Sort moves based on preliminary evaluation
        moves_with_eval.sort(reverse=maximizing_player)
        valid_moves = [move for _, move in moves_with_eval]
        
        # Alpha-beta pruning
        best_move = None
        if maximizing_player: # AI move
            max_eval = float('-inf') # Start eval
            for move in valid_moves:
                game_copy = copy.deepcopy(game) # Makes a copy of the current game
                game_copy.make_move(*move) # Simulates a move
                eval, _ = self.minimax(game_copy, depth - 1, alpha, beta, False, start_time) # Next depth
                
                # Update eval
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        # This part implements the same logic just for minimizing
        else: # Player move
            min_eval = float('inf') # Start eval
            for move in valid_moves:
                game_copy = copy.deepcopy(game)
                game_copy.make_move(*move)
                eval, _ = self.minimax(game_copy, depth - 1, alpha, beta, True, start_time)
                
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            
            return min_eval, best_move
    
    def get_move(self, game: UltimateTicTacToe) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Returns the best move that the AI got using minimax."""
        start_time = time.time()
        _, best_move = self.minimax(game, self.max_depth, float('-inf'), float('inf'), 
                                  game.current_player == 1, start_time)
        end_time = time.time()
        move_time = end_time - start_time
        print(f"AI took {move_time:.2f} seconds to make a move")
        
        if move_time >= self.MAX_MOVE_TIME:
            print("Warning: AI move time exceeded limit")
        
        return best_move

# Class for creating the GUI
class UltimateTicTacToeGUI:
    def __init__(self, root):
        """Initializes the GUI."""
        self.root = root
        self.root.title("Ultimate Tic-Tac-Toe")
        
        self.game = UltimateTicTacToe() # Object with game rules
        self.ai = AIPlayer() # Object pointing to the AI
        
        self.setup_gui()
        self.update_board()
        
    def setup_gui(self):
        """Sets up all the GUI elements."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Game info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        self.status_label = ttk.Label(info_frame, text="Player's turn", font=('Arial', 12))
        self.status_label.grid(row=0, column=0, padx=5)
        
        # Difficulty selector
        ttk.Label(info_frame, text="AI Difficulty:").grid(row=0, column=1, padx=5)
        self.difficulty_var = tk.StringVar(value="Medium")
        difficulty_combo = ttk.Combobox(info_frame, textvariable=self.difficulty_var, 
                                      values=["Easy", "Medium", "Hard"], state="readonly")
        difficulty_combo.grid(row=0, column=2, padx=5)
        difficulty_combo.bind('<<ComboboxSelected>>', self.change_difficulty)
        
        # New game button
        new_game_btn = ttk.Button(info_frame, text="New Game", command=self.reset_game)
        new_game_btn.grid(row=0, column=3, padx=5)
        
        # Create the game board
        self.buttons = {}
        for i in range(3):
            for j in range(3):
                # Create frame for each small board
                board_frame = ttk.Frame(main_frame, padding=2)
                board_frame.grid(row=i+1, column=j, padx=2, pady=2)
                board_frame.configure(style='Board.TFrame')
                
                # Set up the buttons
                for x in range(3):
                    for y in range(3):
                        btn = ttk.Button(board_frame, width=3, style='Cell.TButton')
                        btn.grid(row=x, column=y, padx=1, pady=1)
                        btn.configure(command=lambda bi=i, bj=j, si=x, sj=y: 
                                    self.make_move(bi, bj, si, sj))
                        self.buttons[(i, j, x, y)] = btn
        
        # Create styles
        style = ttk.Style()
        style.configure('Board.TFrame', relief='solid')
        style.configure('Cell.TButton', padding=5)
        style.configure('Active.Cell.TButton', background='lightblue')
        style.configure('Won.Cell.TButton', background='lightgreen')
        style.configure('Lost.Cell.TButton', background='lightcoral')
        
    def update_board(self):
        """Updates the board after every move."""
        # Update all buttons
        for (bi, bj, si, sj), btn in self.buttons.items():
            value = self.game.small_boards[bi, bj, si, sj]
            text = ' ' if value == 0 else 'X' if value == 1 else 'O'
            btn.configure(text=text)
            
            # Highlight next valid board(s)
            if self.game.next_board is None:
                is_active = self.game.big_board[bi, bj] == 0
            else:
                is_active = (bi, bj) == self.game.next_board
                
            if self.game.big_board[bi, bj] == 1:  # X won
                btn.configure(style='Won.Cell.TButton')
            elif self.game.big_board[bi, bj] == 2:  # O won
                btn.configure(style='Lost.Cell.TButton')
            elif is_active and self.game.get_game_state() == -1:
                btn.configure(style='Active.Cell.TButton')
            else:
                btn.configure(style='Cell.TButton')
        
        # Update status
        game_state = self.game.get_game_state()
        if game_state == -1:
            player = 'Player' if self.game.current_player == 1 else 'AI'
            self.status_label.configure(text=f"{player}'s turn")
        elif game_state == 0:
            self.status_label.configure(text="Game Over - Draw!")
        else:
            winner = 'Player' if game_state == 1 else 'AI'
            self.status_label.configure(text=f"Game Over - {winner} wins!")
            
    def make_move(self, bi, bj, si, sj):
        # b_ --> main board coordinates
        # s_ --> small board coordinates 
        if self.game.get_game_state() != -1: # is the game finished
            return
            
        if self.game.current_player == 1:  # Player's turn
            if self.game.make_move((bi, bj), (si, sj)): # Try make move
                self.update_board() # Update board view
                self.root.update() # Update window
                
                if self.game.get_game_state() == -1: # is the game still ongoing
                    self.root.after(100, self.ai_move) # AI's move
                    
    def ai_move(self):
        """Calls the AI to make a move."""
        move = self.ai.get_move(self.game)
        if move:
            self.game.make_move(*move)
            self.update_board()
            
    def change_difficulty(self, event=None):
        """Changes the difficulty."""
        self.ai.set_difficulty(self.difficulty_var.get())
        
    def reset_game(self):
        """Begins new game."""
        self.game = UltimateTicTacToe()
        self.update_board()
    


def main():
    """Starts the application."""
    root = tk.Tk()
    app = UltimateTicTacToeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()