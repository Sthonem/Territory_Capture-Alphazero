"""Small demo script that plays random Territory Capture games."""

from __future__ import annotations

import random

from .game import TerritoryCaptureGame
from .rules import PLAYER_O, PLAYER_X
from .utils import print_board


def play_random_game(verbose: bool = True) -> TerritoryCaptureGame:
    """Play one full game using uniformly random legal moves."""

    game = TerritoryCaptureGame()

    if verbose:
        print("Starting new random game")
        print_board(game.board)
        print()

    while not game.is_terminal():
        move = random.choice(game.get_legal_moves())
        current_player = game.current_player
        game.apply_move(move)

        if verbose:
            print(f"Player {current_player} plays {move}")
            print_board(game.board)
            print()

    return game


def print_final_result(game: TerritoryCaptureGame) -> None:
    """Print the final board, score, and winner."""

    result = game.get_result()
    print("Final board:")
    print_board(game.board)
    print()
    print(
        f"Territory score -> {PLAYER_X}: {result.scores[PLAYER_X]}, "
        f"{PLAYER_O}: {result.scores[PLAYER_O]}"
    )

    if result.winner is None:
        print("Result: Draw")
    else:
        print(f"Result: Player {result.winner} wins")


def main(number_of_games: int = 1, verbose: bool = True) -> None:
    """Run one or more random games."""

    for game_index in range(number_of_games):
        if number_of_games > 1:
            print(f"=== Game {game_index + 1} ===")
        game = play_random_game(verbose=verbose)
        print_final_result(game)
        if game_index < number_of_games - 1:
            print()


if __name__ == "__main__":
    main()
