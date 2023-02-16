# Computational Intelligence, Final Project: Quarto Agents

| **Contributors** | **sID** |
|---|---|
| Alessio Carachino | s296138 |
| Giuseppe Atanasio | s300733 |
| Francesco Sorrentino | s301665 |
| Francesco Di Gangi | s301793 |

## Note
**ALL MEMBERS EQUALLY CONTRIBUTED TO THE PROJECT. FIRST EACH MEMBER MAINLY FOCUSED ON ONE IDEA AND THEN WE MERGED THE FINDINGS**

## Directory Tree

```
.
├── README.md
├── poetry.lock
├── pyproject.toml
└── quarto
    ├── GA_MinMaxPlayer.py
    ├── GA_Player.py
    ├── MinMax_Player.py
    ├── RandomPlayer.py
    ├── __init__.py
    ├── __pycache__
    │   ├── GA_MinMaxPlayer.cpython-310.pyc
    │   ├── GA_Player.cpython-310.pyc
    │   ├── MinMax_Player.cpython-310.pyc
    │   ├── RandomPlayer.cpython-310.pyc
    │   └── minimax.cpython-310.pyc
    ├── image.jpg
    ├── image2.jpg
    ├── main.py
    ├── main_RL.py
    ├── minimax.py
    ├── quarto
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-310.pyc
    │   │   └── objects.cpython-310.pyc
    │   └── objects.py
    ├── readme.md
    └── reinforcement
        ├── Memory.py
        ├── Q_data_RL_GA.dat
        ├── __init__.py
        ├── images
        │   ├── RL&GA_vs_GA_2nd_test.svg
        │   ├── RL&GA_vs_GA_2nd_train.svg
        │   ├── RL&GA_vs_GA_eps0.95_alpha0.1_train.svg
        │   ├── RL&GA_vs_GA_eps0.995_alpha0.1_train.svg
        │   ├── RL&GA_vs_GA_eps0.995_alpha0.3_train.svg
        │   ├── RL&GA_vs_GA_eps0.9995_alpha0.1_train.svg
        │   └── RL&GA_vs_Random_2nd_test.svg
        ├── rl_agent.py
        ├── tables.py
        └── utils.py
```

## Setting up the environment
First of all you have to install `poetry` on your pc. Then, you have to move inside the "quarto_ci_2022_2023" folder and launch this command:
```
poetry install
```
After that, to test our code you have to launch the following:
```
poetry run python quarto/[main_RL or main].py
```
## Summary
- [GA Agent](#ga-agent)
- [MinMax](#minmax)
- [GA+MinMax Agent](#gaminmax-agent)
- [RL Agent](#rl-agent)
- [Conclusion](#conclusion)

## Timeline with contributions
**"ALL MEMBERS EQUALLY CONTRIBUTED TO THE PROJECT. FIRST EACH MEMBER MAINLY FOCUSED ON ONE IDEA AND THEN WE MERGED THE FINDINGS"**

1) At that time, we needed a better agent than random agent to conduct better experiments, so I came up with the idea of implementing an agent based on Genetic Algorithms. The GA agent worked better than expected.
2) In the meantime, my colleagues Di Gangi and Sorrentino, implemented a Player based on MinMax. We learned that we could not use only a MinMax with full-depth because of excessive computational costs
3) I came up with the idea that, in this case, to keep the computational costs acceptable, even in games with a bigger board, we needed to combine the GA Agent with MinMax(at limited depth)
4) At that point, Di Gangi and Sorrentino implemented a minmax with a variable depth with GA.
5) In the meantime, Attanasio developed an agent based on Reinforcement Learning, slightly combined with GA; we learned from his experience that RL was not what we were looking for.

## GA Agent
The GA Agent is the core of our final solution. Let's go deeper into its characteristics and then how the individuals take the choices of ```choose_piece``` and ```place_piece```.

### Fitness
It is basically the win rate, over 100 games, of the individual versus the Random Agent.

### Genome
It is made of two genes: ```alpha``` and ```beta```.

They act as probabilities, so they span from 0 to 1, and they have only one number after the decimal point. 
Cutting to only one number after the decimal point has been a reasoned choice: since the opponent made his moves randomly, the win rate varied a lot even for the same genome.

So, I thought that, in the end, the final genome would be the one that collects the best average win rate over the number of occurrences of the same genome. So, given the fact that a genome such as ```[0.129, 0.436]``` acts similarly to ```[0.1, 0.4]```,
if we find the same genomes more often, we will get more reliable results, for the final average over the number of occurrences.

It is explained later how the genome affects the strategy.

### Population management
Taking into account the following parameters:
```python
    NUM_GENERATIONS = 100
    POPULATION_SIZE = 20
    OFFSPRING_SIZE = 5
```

the initial population is randomly initialized. This means that each individual of the first 20 individuals has random ```alpha``` and ```beta``` genes.


### Genetic Operators
They are three and occur with different probabilities, according to how many generations elapsed (check the function ```compute_GO_probs```) for the details.
- Mutation: They randomly add or remove 0.1 to alpha or beta;
- Crossover: Regular crossover. Since the genome is made only by two genes, the new genome has alpha equal to g1's alpha and beta equal to g2's beta;
- Crossover2: Similar to crossover but slightly increases/decreases alpha and beta while they are being copied.

### Strategy
First of all, the strategy exploits ```cook_status``` that builds a so-made status:


- ```elements_per_type```: array of tuples → for each type of piece, the array contains the number of that type of piece and its binary value. (Since pieces are expressed in binary form, high_values are represented by 1000 (8), coloured_values by 0100 (4) and so on...).
  It is used for ```choose_piece```.
- ```rows_at_risk```: list of five lists representing the 5 different risks → according to the number of missing pieces in a row, it indicates the level of risk associated with that row;
  + a filled row is associated to risk 0
  + a row with 1 missing piece is associated to risk 4
  + a row with 2 missing pieces is associated to risk 3
  + a row with 3 missing pieces is associated to risk 2
  + a row with 4 missing pieces is associated to risk 1
- ```columns_at_risk```: same as ```rows_at_risk``` but for columns
- ```diagonals_at_risk```: array → this array contains 1 if the **main diagonal** is missing 1 piece to be filled; it contains 2 if the **secondary diagonal** is missing 1 piece to be filled


Now that we have all the ingredients, let's go deeper into the strategy:
- ```choose_piece```: the gene ```alpha```, which acts as a probability, determines if to choose a piece with high correlation, in terms of piece characteristics, or low correlation.
 In other words, it finds the most/less correlated characteristics and (binary) sums them to build a piece. If that piece is already on the board, a new piece will be generated until it is returned.
- ```place_piece```: First exploiting the genome, the algorithm checks if there is any emergency (in terms of risk) to address. In order:
  + If there is any diagonal where it is missing only one piece to be completed, the algorithm chooses that position to fill the diagonal;
  + If there is any row or column with risk 4 (only one piece is missing to be filled), the algorithm chooses that position to fill the row/column. If there is at least one row at risk 4 and one column at risk 4, the algorithm chooses randomly what to fill;
Now the gene ```beta``` joins the action: when there is no a situation of high risk, ```beta```, which acts as a probability, determines if the algorithm will choose if it is the case of putting the piece in a row/column with risk 3 (two pieces missing to be filled), or lower risk (empty row/column or three pieces to be filled).

### Code
The code can be found in the file GA_Player.py

### Results over 1000 games, ```genome={'alpha': 0.1, 'beta': 0.2}```
| **First player** | **Second player** | **Win Rate %** | **Draws Rate %** | **Loss Rate %** |
|------------------|-------------------|----------------|------------------|-----------------|
| GA Agent         | Random Agent      | 83.2           | 1.3              | 15.5            |
| Random Agent     | GA Agent          | 14.4           | 1.6              | 84              |

## MinMax
### How it works
While I was working on GA Agent, Francesco Di Gangi and Sorrentino worked on MinMax. 
In particular, they started from the MinMax implementation of lab3, with alpha-beta pruning, to reduce the computational costs, enabled.
Here is the pseudocode:
```python
function minimax(board, depth, alpha, beta, player)
    if game is over or depth == 0
        return the heuristic value of the board
    if player is MAX
        bestValue = -infinity
        for each legal move m
            make move m on board
            score = minimax(board, depth - 1, alpha, beta, MIN)
            bestValue = max(bestValue, score)
            alpha = max(alpha, bestValue)
            if beta <= alpha
                break
        return bestValue
    else if player is MIN
        bestValue = +infinity
        for each legal move m
            make move m on board
            score = minimax(board, depth - 1, alpha, beta, MAX)
            bestValue = min(bestValue, score)
            beta = min(beta, bestValue)
            if beta <= alpha
                break
        return bestValue
```

```python
score = minimax(board, depth - 1, alpha, beta, MAX)
            bestValue = min(bestValue, score)
            beta = min(beta, bestValue)
            if beta <= alpha
                break        
```


Results were not really encouraging in terms of time and did not leave space for scalability hopes. 


## GA+MinMax Agent

### Introduction
Summing up, the GA Agent was really fast and effective, MinMax is much more effective but really slow.
So I came up with the idea that we could strengthen the GA by recurring to MinMax in some cases only.

### MinMax optimizations
- ```choose_piece```: It is the same as GA Agent, but in addition, whenever there is only one missing piece to fill a 
diagonal/row/column, the algorithm uses MinMax (with depth fixed to 1). In this case, it is returned a piece for which it is guaranteed that the opponent won't be able to win with.
- ```place_piece```: Also in this case, MinMax joins the action when there is a diagonal/column/row with only one piece missing to be filled.
In particular, to fasten the computation, when there is a number of missing pieces greater or equal than 8, a MinMax with limited depth to 1 will return the solution to win the game, if it exists.
On the other hand, when the number of missing pieces in the board is less than 8, MinMax will try to find a winning move that allows to win with depth ```self.depth```.

### Code
The code can be found in GA_MinMaxPlayer.py
It is basically the same as GA Agent but of course with some MinMax additions.

First, ```cook_status``` is the same but with one more property called ```holes``` which represents the number of missing pieces in the board.

In the case of ```choose_piece```, here is the code (differences are highlighted):
```python
    def choose_piece(self) -> int:
        game = self.get_game()
        board = game.get_board_status()
        status = cook_status(game, board)
        elements_per_type = status["elements_per_type"]

        alpha = self.genome["alpha"]
        elements = 4

        not_winning_pieces = list()
        rows_high_risk = status["rows_at_risk"][4]
        columns_high_risk = status["columns_at_risk"][4]
        diagonals_high_risk = status["diagonals_at_risk"]
        
        # CHANGES START HERE
        minmax = MinMax_Player.MinMax(self.get_game())

        if len(diagonals_high_risk) != 0 or len(rows_high_risk) != 0 or len(columns_high_risk) != 0:
            not_winning_pieces = minmax.not_winning_pieces(board)
            if len(not_winning_pieces) == 1:
                return not_winning_pieces[0]

        # CHANGED END HERE
        if alpha < random.random():

            while True:
                sorted_combinations = list(
                    combinations(sorted(elements_per_type, key=lambda i: i[0])[:elements], r=4))
                random.shuffle(sorted_combinations)
                for combination in sorted_combinations:
                    piece_val = sum([val for e, val in combination])
                    if piece_val not in board:
                        if len(not_winning_pieces) == 0 or piece_val in not_winning_pieces:
                            return piece_val
                elements += 1
        else:
            while True:
                sorted_combinations = list(
                    combinations(sorted(elements_per_type, key=lambda i: i[0], reverse=True)[:elements], r=4))
                random.shuffle(sorted_combinations)
                for combination in sorted_combinations:
                    piece_val = sum([val for e, val in combination])
                    if piece_val not in board:
                        if len(not_winning_pieces) == 0 or piece_val in not_winning_pieces:
                            return piece_val
                elements += 1
```
while ```not_winning_pieces``` is a function that returns all the pieces which it is not possible to win with in the next opponent's turn. It is so defined:
```python
    def not_winning_pieces(self, board: numpy.ndarray) -> list:
        current_quarto = self.get_game()
        available_pieces = list(set(range(16)) - set(board.ravel()))
        not_winning_pieces = list()
        moves = []
        for i in range(4):
            for j in range(4):
                if board[i][j] == -1:
                    moves.append((j, i))
        for piece in available_pieces:
            this_piece_can_win = False
            for move in moves:
                quarto_copy: quarto.Quarto = copy.deepcopy(current_quarto)
                quarto_copy.select(piece)
                quarto_copy.place(move[0], move[1])
                score = self.heuristic2(quarto_copy)
                if score == float('inf'):
                    this_piece_can_win = True
                    break
            if not this_piece_can_win:
                not_winning_pieces.append(piece)

        return not_winning_pieces
```

On the other hand, ```place_piece```, earns the following highlighted changes:
```python
    def place_piece(self) -> tuple[int, int]:
        game = self.get_game()
        board = game.get_board_status()
        status = cook_status(game, board)
        beta = self.genome["beta"]

        rows_high_risk = status["rows_at_risk"][4]
        columns_high_risk = status["columns_at_risk"][4]
        diagonals_high_risk = status["diagonals_at_risk"]
        
        # CHANGES START HERE
        minmax = MinMax_Player.MinMax(self.get_game())

        if len(diagonals_high_risk) != 0 or len(rows_high_risk) != 0 or len(columns_high_risk) != 0:
            if status['holes'] < 8:
                can_beat_move = minmax.place_piece(self.depth)  # 10
                # print(can_beat_move)
                if can_beat_move is not None:
                    return can_beat_move
            else:
                can_beat_move = minmax.place_piece(1)  # max(16 - holes - 1, 1)
                # print(can_beat_move)
                if can_beat_move is not None:
                    return can_beat_move

        # CHANGES END HERE
        # and then it is the same as GA Agent
        # ...
```
In this case, ```minmax.place_piece(depth)```, is so defined
```python
def place_piece(self, depth) -> tuple[int, int]:
        '''place_piece using minmax'''
        game = self.get_game()
        move = play(game, depth)
        return move

def play(game, depth):
    scored_moves = []
    for move in get_all_possible_moves(game):
        game_t = deepcopy(game)
        game_t.place(move[0], move[1])
        scored_moves.append((move, minmax(game_t, depth, False)))
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    scored_moves = list(filter(lambda x: x[1] != -1, scored_moves))
    return scored_moves[0][0] if scored_moves[0][1] != float('-inf') or  scored_moves[0][1] != -1 else None
```

### Results ```GA: genome={'alpha': 0.1, 'beta': 0.2}```, ```GA+MinMax: genome={'alpha': 0.1, 'beta': 0.3}```
For each row, the results are retrived after 1000 games. First player and second player started as first equally.

| **First player**          | **Second player**         | **Win Rate %** | **Draws Rate %** |
|---------------------------|---------------------------|----------------|------------------|
| GA Agent                  | Random Agent              | 80-84          | 2.5              |
| GA+MinMax Agent (depth=1) | Random Agent              | 97.5-98        | 0-0.5            |
| GA+MinMax Agent (depth=2) | Random Agent              | 98-98.5        | 0.5-1            |
| GA+MinMax Agent (depth=1) | GA Agent                  | 75-80          | 6-11             |
| GA+MinMax Agent (depth=2) | GA+MinMax Agent (depth=1) | 44.4           | 11               |

## RL Agent
**THE FOLLOWING PARAGRAPH ON RL, HAS BEEN WRITTEN BY GIUSEPPE ATTANASIO**

Starting from our Lab3 code, we have created a mixed strategy based on:

- model-free Q-learning for `place_piece()`, in which the reward is +100 or -100 respectively according to winning or losing, and 0 in case of draw ;
- GA for `choose_piece()`, that is the same function defined in the previous strategies

### Model-free Q-learning

We have decided to use the model-free Q-learning for the function `place_piece()` since it best fits with trial-and-error, and we want our agent to learn game by game.

It is a **Markovian Decision Process** since it chooses the best policy based on its experience. In other words, we want our agent to maximize the expected value of total reward over any and all the following steps, starting from the current state.

The structure of Q-learning relies on Q-table, defined as follows:

`(state, action) = (actual state, ((x, y), piece))`

- **state**: it is the playground converted in tuple
- **action**:
    - **(x, y)**: coordinates of the action that the agent wants to do
    - **piece**: it is the piece placed on the previous coordinates taken

As we said before, we want our agent to maximize the expected value of total reward: this is done by retrieving the maximum value over a set of actions for a specific `state`, and this value is called **q-value**. In particular, we have implemented this logic inside the `q_move()` function.

For what concerns the reward, we handled it in the function `q_post()`, in which we update the q-value for the last (state, action) couple. Also in this function we handle the epsilon-greedy strategy explained further on. 

We leveraged this structure to build a Q-table based on dictionary of dictionaries, in which the  external is about `state`, the internal is about the `action`. We leveraged also the [[1]](https://www.notion.so/Atanasio-Giuseppe-Report-0191d86492cd42c899793b672ccf9adf) library to handle the Q-table.

For what concerns RL part, we have chosen the hyperparameters `alpha` and `gamma` with a greedy approach, you could check them in the table below. In order to handle in a different way the exploration-exploitation process, Giuseppe Atanasio introduced **Epsilon-Greedy Policy** [[2]](https://www.notion.so/Atanasio-Giuseppe-Report-0191d86492cd42c899793b672ccf9adf) in order to give a schedule to the probability of making a random move in training phase, starting from the following configuration:

| alpha | gamma | epsilon | min_epsilon | epsilon_decay |
|-------|-------|---------|-------------|---------------|
| 0.1   | 0.9   | 1       | 0.1         | 0.9995        |

`self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)`

![Unknown-2.png](Atanasio%20Giuseppe%20-%20Report%200191d86492cd42c899793b672ccf9adf/Unknown-2.png)

### Genetic Algorithm

For the GA part, we have used the following parameters, both for the mixed-strategy agent and the pure GA agent:

| alpha | beta |
|-------|------|
| 0.1   | 0.3  |

### Training and Testing Results

Given the previous structure, we want to train and test this strategy against pure GA, and use the previous training backbone to test against a Random Player. 
Table with iterations follows:

| Mode  | # Iterations | # Games in a sample (precision) |
|-------|--------------|---------------------------------|
| train | 100.000      | 1.000                           |
| test  | 30.000       | 300                             |

The first number has been chosen arbitrarily high to increase the coverage since there is a large sparsity due to the structure of the playground and of the problem as a whole, but it could be improved: in fact, we verified that Q-table didn’t cover all possible (state, action), and this is probably due to the lower number of iterations. Since we couldn’t have tested with a higher number of iterations for poor performances of our devices, we handled the missing previous couples by returning a random free cell in testing phase when (state, action) misses. For the same reasons, our choice was to play our agent as only second, and not both first and second.

Here are some statistics, followed by their graphs:

| Against      | Mode  | Avg Win rate % | Max Win rate %     | ETA (minutes) | it/s   |
|--------------|-------|----------------|--------------------|---------------|--------|
| GA           | train | 64.08          | 69.6 (iter. 58000) | 13:54         | 117.67 |
| GA           | test  | 73.69          | 78 (iter. 1800)    | 04:10         | 119.63 |
| RandomPlayer | test  | 56.36          | 64 (iter. 9300)    | 01:45         | 285.34 |

![Training against GA](Atanasio%20Giuseppe%20-%20Report%200191d86492cd42c899793b672ccf9adf/RLGA_vs_GA_eps0.9995_alpha0.1_train.svg)

Training against GA

![Testing against GA](Atanasio%20Giuseppe%20-%20Report%200191d86492cd42c899793b672ccf9adf/RLGA_vs_GA_2nd_test.svg)

Testing against GA

![Testing against Random Player](Atanasio%20Giuseppe%20-%20Report%200191d86492cd42c899793b672ccf9adf/RLGA_vs_Random_2nd_test.svg)

Testing against Random Player

The best possible way to train our model is to do it against an optimal strategy agent, and the difference in testing between pure GA and Random Player could be explained by this evidence.

### References:

[1]: [Bblais’ Github Library](https://github.com/bblais/Game)

[2]: [The Epsilon-Greedy Algorithm for Reinforcement Learning](https://medium.com/analytics-vidhya/the-epsilon-greedy-algorithm-for-reinforcement-learning-5fe6f96dc870)

### Code
The code can be found in main_RL.py and in reinforcement folder.

# Conclusion
Overall the most performing agent is our final strategy GA_MinMaxPlayer: even if it may seem slower than the others, it's actually really fast. At depth 1 it can reach 98% win rate and plays a game even in half a second.
