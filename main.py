#!/usr/bin/env python3

import sys
import minesweeper as mine
import aiclp



def main():
    total_games = 10
    won_games = 0

    args = sys.argv[1:]
    if len(args) > 0:
        args[0] = int(args[0])
    if len(args) > 1:
        args[1] = int(args[1])
    if len(args) > 2:
        try:
            args[2] = int(args[2])
        except ValueError:
            args[2] = float(args[2])

    game = mine.MineSweeper(*args)
    ai = aiclp.AI()

    for _ in range(total_games):
        ai.new_game(game.board.shape[1], game.board.shape[0])

        while not game.finished:
            board = game.board
            print(game)
            move = ai.next_move(board)
            game.click(*move)

        print(game)
        if game.won:
            print("You won")
            won_games += 1
        if game.lost:
            print("You lost")
        game.restart()

    print("Won %d/%d games" % (won_games, total_games))




if __name__ == '__main__':
    main()
