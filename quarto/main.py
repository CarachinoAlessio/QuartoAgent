import logging
import argparse
import quarto
from GA_Player import evolve, GA_Player
from RandomPlayer import RandomPlayer
#from MinMax_Player import MinMax
from GA_MinMaxPlayer import  GA_MinMaxPlayer
from tqdm import tqdm

def main():
    '''
    print("old GA tuning second player-----------------------------")
    game = quarto.Quarto()
    find_genome = evolve()
    #print(find_genome)
    found_genome = {'alpha': 0.1, 'beta': 0.2}
    game.set_players((RandomPlayer(game), GA_Player(game, found_genome)))
    winner = game.run()
    logging.info(f"main: Winner: player {winner}")

    print("old GA tuning first player-----------------------------")
    game = quarto.Quarto()
    #find_genome = evolve()
    #print(find_genome)
    found_genome = {'alpha': 0.1, 'beta': 0.2}
    game.set_players((GA_Player(game, found_genome), RandomPlayer(game)))
    winner = game.run()
    logging.info(f"main: Winner: player {winner}")


    print("new GA tuning second player-----------------------------")
    game = quarto.Quarto()
    find_genome = new_evolve()
    print(find_genome)
    #found_genome = {'alpha': 0.0, 'beta': 0.1}
    game.set_players((RandomPlayer(game), GA_MinMaxPlayer(game, find_genome)))
    winner = game.run()
    logging.info(f"main: Winner: player {winner}")

    print("new GA tuning first player-----------------------------")
    game = quarto.Quarto()
    find_genome = new_evolve()
    print(find_genome)
    #found_genome = {'alpha': 0.0, 'beta': 0.1}
    game.set_players((GA_MinMaxPlayer(game, find_genome), RandomPlayer(game)))
    winner = game.run()
    logging.info(f"main: Winner: player {winner}")
    '''



    wr = 0
    dr = 0
    for _ in tqdm(range(1)):
        game = quarto.Quarto()
        # find_genome = evolve()
        found_genome = {'alpha': 0.1, 'beta': 0.2}
        game.set_players((RandomPlayer(game), GA_Player(game, found_genome)))
        winner = game.run()
        if winner == 0:
            wr += 1
        if winner == -1:
            dr += 1
    print(f"Random vs oldGA(1000 games): wins: {wr}, draws: {dr}")

    wr = 0
    dr = 0
    for _ in tqdm(range(1)):
        game = quarto.Quarto()
        # find_genome = evolve()
        found_genome = {'alpha': 0.1, 'beta': 0.2}
        game.set_players((GA_Player(game, found_genome), RandomPlayer(game)))
        winner = game.run()
        if winner == 0:
            wr += 1
        if winner == -1:
            dr += 1
    print(f"oldGA vs Random(1000 games): wins: {wr}, draws: {dr}")

    wr = 0
    dr = 0
    for _ in tqdm(range(50)):
        game = quarto.Quarto()
        # find_genome = evolve()
        found_genome = {'alpha': 0.1, 'beta': 0.3}
        game.set_players((RandomPlayer(game), GA_MinMaxPlayer(game, found_genome)))
        winner = game.run()
        if winner == 1: # If the second player (minmax) wins, add
            wr += 1
        if winner == -1:
            dr += 1
    print(f"Random vs newGA(1000 games): wins: {wr}, draws: {dr}")

    #wr = 0
    #dr = 0
    for _ in tqdm(range(50)):
        game = quarto.Quarto()
        # find_genome = evolve()
        found_genome = {'alpha': 0.1, 'beta': 0.3}
        game.set_players((GA_MinMaxPlayer(game, found_genome), RandomPlayer(game)))
        winner = game.run()
        if winner == 0:
            wr += 1
        if winner == -1:
            dr += 1
    print(f"newGA vs Random(1000 games): wins: {wr}, draws: {dr}")

    wr = 0
    dr = 0
    for _ in tqdm(range(5)):
        game = quarto.Quarto()
        # find_genome = evolve()
        found_genome0 = {'alpha': 0.0, 'beta': 0.1}
        found_genome1 = {'alpha': 0.1, 'beta': 0.2}
        game.set_players((GA_MinMaxPlayer(game, found_genome0), GA_Player(game, found_genome1)))
        winner = game.run()
        if winner == 0:
            wr += 1
        if winner == -1:
            dr += 1
    #print(f"new GA vs old GA(1000 games): wins: {wr}, draws: {dr}")

    #wr = 0
    #dr = 0
    for _ in tqdm(range(5)):
        game = quarto.Quarto()
        # find_genome = new_evolve()
        found_genome0 = {'alpha': 0.1, 'beta': 0.2}
        found_genome1 = {'alpha': 0.1, 'beta': 0.2}

        game.set_players((GA_Player(game, found_genome0), GA_MinMaxPlayer(game, found_genome1)))
        winner = game.run()
        if winner == 1: # If the second player (minmax) wins, add
            wr += 1
        if winner == -1:
            dr += 1
    print(f"old GA vs new GA(1000 games): wins: {wr}, draws: {dr}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=2, help='increase log verbosity')
    parser.add_argument('-d',
                        '--debug',
                        action='store_const',
                        dest='verbose',
                        const=2,
                        help='log debug messages (same as -vv)')
    args = parser.parse_args()

    if args.verbose == 0:
        logging.getLogger().setLevel(level=logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger().setLevel(level=logging.INFO)
    elif args.verbose == 2:
        logging.getLogger().setLevel(level=logging.DEBUG)
    main()
