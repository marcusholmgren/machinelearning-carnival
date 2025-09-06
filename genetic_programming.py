# genetic_programming.py
"""
A framework for Genetic Programming (GP) in Python.

Genetic Programming is an evolutionary computation technique that evolves computer
programs, traditionally represented as tree structures. This script provides the
core components to build and evolve these programs to solve specific problems.

Core Concepts:
- Program Representation: Programs are represented as expression trees.
  - `FunctionNode`: Internal nodes that perform an operation (e.g., +, *).
  - `ParamNode`: Terminal nodes that represent an input variable (e.g., x, y).
  - `ConstantNode`: Terminal nodes that represent a constant value (e.g., 3, 5).
- Evolution Process: A population of these program trees is evolved over
  generations.
  - Fitness Function: Each program is evaluated and assigned a fitness score
    based on how well it solves the target problem.
  - Selection: Programs with better fitness scores are more likely to be
    selected to create the next generation.
  - Crossover: Two parent programs are combined by swapping sub-trees to create
    a new child program.
  - Mutation: A program is randomly modified, for instance by replacing a
    sub-tree with a new random one.
"""
from __future__ import annotations
from random import random, randint, choice
from copy import deepcopy
from math import log
from typing import List, Any, Callable
from dataclasses import dataclass

# --- Tree Representation ---

@dataclass
class FunctionWrapper:
    """A wrapper for functions used in the expression tree."""
    function: Callable[[List[Any]], Any]
    child_count: int
    name: str

@dataclass
class Node:
    """Base class for a node in the expression tree."""
    def evaluate(self, inputs: List[Any]) -> Any:
        raise NotImplementedError

    def display(self, indent: int = 0) -> None:
        raise NotImplementedError

@dataclass
class FunctionNode(Node):
    """A node representing a function call."""
    fw: FunctionWrapper
    children: List[Node]

    def evaluate(self, inputs: List[Any]) -> Any:
        results = [n.evaluate(inputs) for n in self.children]
        return self.fw.function(results)

    def display(self, indent: int = 0) -> None:
        print(f"{' ' * indent}{self.fw.name}")
        for child in self.children:
            child.display(indent + 1)

@dataclass
class ParamNode(Node):
    """A node representing an input parameter."""
    idx: int

    def evaluate(self, inputs: List[Any]) -> Any:
        return inputs[self.idx]

    def display(self, indent: int = 0) -> None:
        print(f"{' ' * indent}p{self.idx}")

@dataclass
class ConstantNode(Node):
    """A node representing a constant value."""
    value: int

    def evaluate(self, inputs: List[Any]) -> Any:
        return self.value

    def display(self, indent: int = 0) -> None:
        print(f"{' ' * indent}{self.value}")

# --- Core GP Operations ---

def make_random_tree(
    param_count: int,
    func_list: List[FunctionWrapper],
    max_depth: int = 4,
    func_prob: float = 0.5,
    param_prob: float = 0.6
) -> Node:
    """
    Creates a random expression tree.

    Args:
        param_count: The number of input parameters the tree can access.
        func_list: The list of FunctionWrappers to use as functions.
        max_depth: The maximum depth of the tree.
        func_prob: The probability of creating a function node (vs. a terminal).
        param_prob: If creating a terminal, the probability of it being a
                    parameter node (vs. a constant).

    Returns:
        A randomly generated Node, the root of the new tree.
    """
    if random() < func_prob and max_depth > 0:
        f = choice(func_list)
        children = [
            make_random_tree(param_count, func_list, max_depth - 1, func_prob, param_prob)
            for _ in range(f.child_count)
        ]
        return FunctionNode(f, children)
    elif random() < param_prob:
        return ParamNode(randint(0, param_count - 1))
    else:
        return ConstantNode(randint(0, 10))

def mutate(
    tree: Node,
    param_count: int,
    func_list: List[FunctionWrapper],
    prob_change: float = 0.1
) -> Node:
    """
    Randomly mutates a tree.

    With a probability of `prob_change`, the entire tree is replaced with a new
    random tree. Otherwise, it recursively attempts to mutate the children.

    Args:
        tree: The tree to mutate.
        param_count: The number of available input parameters.
        func_list: The list of available functions.
        prob_change: The probability of replacing a node with a new random one.

    Returns:
        A mutated tree.
    """
    if random() < prob_change:
        return make_random_tree(param_count, func_list)
    else:
        result = deepcopy(tree)
        if isinstance(tree, FunctionNode):
            result.children = [
                mutate(c, param_count, func_list, prob_change) for c in tree.children
            ]
        return result

def crossover(
    tree1: Node,
    tree2: Node,
    prob_swap: float = 0.7,
    top: bool = True
) -> Node:
    """
    Performs crossover between two trees, creating a new child tree.

    With a probability of `prob_swap`, a subtree from `tree1` is replaced with a
    randomly chosen subtree from `tree2`.

    Args:
        tree1: The first parent tree.
        tree2: The second parent tree.
        prob_swap: The probability of swapping a subtree.
        top: A flag to prevent swapping the entire tree at the root.

    Returns:
        A new tree created from the crossover operation.
    """
    if random() < prob_swap and not top:
        return deepcopy(tree2)
    else:
        result = deepcopy(tree1)
        if isinstance(tree1, FunctionNode) and isinstance(tree2, FunctionNode) and tree2.children:
            result.children = [
                crossover(c, choice(tree2.children), prob_swap, False)
                for c in tree1.children
            ]
        return result


class GeneticProgramming:
    """A class to manage the genetic programming evolutionary process."""

    def __init__(
        self,
        param_count: int,
        func_list: List[FunctionWrapper],
        population_size: int = 500,
        mutation_rate: float = 0.2,
        breeding_rate: float = 0.4,
        selection_pressure: float = 0.7,
        new_child_prob: float = 0.05,
    ):
        """Initializes the GP environment."""
        self.param_count = param_count
        self.func_list = func_list
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.breeding_rate = breeding_rate
        self.selection_pressure = selection_pressure
        self.new_child_prob = new_child_prob

    def _select_index(self) -> int:
        """
        Returns a random index, tending towards lower numbers.
        This is a form of rank selection where better-ranked (lower index)
        individuals are more likely to be chosen.
        """
        return int(log(random()) / log(self.selection_pressure))

    def evolve(self, rank_function: Callable, max_gen: int = 100) -> Node:
        """
        Runs the evolutionary process.

        Args:
            rank_function: A function that takes a population of trees and
                           returns a sorted list of (score, tree) tuples.
            max_gen: The maximum number of generations to run.

        Returns:
            The best tree found after all generations.
        """
        population = [
            make_random_tree(self.param_count, self.func_list)
            for _ in range(self.population_size)
        ]

        for i in range(max_gen):
            scores = rank_function(population)
            print(f"Generation {i}, Best Score: {scores[0][0]}")
            if scores[0][0] == 0:
                break

            # Elitism: The two best programs always survive to the next generation.
            new_pop = [scores[0][1], scores[1][1]]

            while len(new_pop) < self.population_size:
                if random() > self.new_child_prob:
                    # Create a new child via crossover and mutation
                    parent1 = scores[self._select_index()][1]
                    parent2 = scores[self._select_index()][1]
                    child = crossover(parent1, parent2, prob_swap=self.breeding_rate)
                    child = mutate(child, self.param_count, self.func_list, prob_change=self.mutation_rate)
                    new_pop.append(child)
                else:
                    # Add a completely new random tree to inject diversity
                    new_pop.append(make_random_tree(self.param_count, self.func_list))

            population = new_pop

        print("\n--- Best Program Found ---")
        scores[0][1].display()
        return scores[0][1]


if __name__ == '__main__':
    # --- Problem 1: Symbolic Regression ---
    def run_symbolic_regression():
        print("\n--- Running Symbolic Regression Example ---")

        # Define the set of functions available for this problem
        add_w = FunctionWrapper(lambda l: l[0] + l[1], 2, 'add')
        sub_w = FunctionWrapper(lambda l: l[0] - l[1], 2, 'subtract')
        mul_w = FunctionWrapper(lambda l: l[0] * l[1], 2, 'multiply')
        func_list = [add_w, sub_w, mul_w]

        # The "hidden" function we want the GP to discover
        def hidden_function(x: int, y: int) -> int:
            return x**2 + 2 * y + 3 * x + 5

        # Build a dataset from the hidden function
        def build_hidden_set() -> List[List[int]]:
            return [[randint(0, 40), randint(0, 40), hidden_function(x,y)] for x,y in [(randint(0,40), randint(0,40)) for i in range(200)]]

        dataset = build_hidden_set()

        # The fitness function: lower score is better
        def score_function(tree: Node, s: List[List[int]]) -> float:
            diff = 0
            for data in s:
                v = tree.evaluate([data[0], data[1]])
                diff += abs(v - data[2])
            return diff

        # A rank function that uses the score function
        def get_rank_function(data: List[List[int]]):
            def rank_function(population: List[Node]):
                scores = [(score_function(t, data), t) for t in population]
                scores.sort(key=lambda x: x[0])
                return scores
            return rank_function

        ranker = get_rank_function(dataset)

        # Initialize and run the GP
        gp = GeneticProgramming(param_count=2, func_list=func_list)
        gp.evolve(ranker, max_gen=100)

    # --- Problem 2: Evolving a Game AI ---
    def run_game_ai_evolution():
        print("\n\n--- Running Game AI Evolution Example ---")

        # Define the functions available for the AI's "brain"
        add_w = FunctionWrapper(lambda l: l[0] + l[1], 2, 'add')
        sub_w = FunctionWrapper(lambda l: l[0] - l[1], 2, 'subtract')
        def if_func(l):
            return l[1] if l[0] > 0 else l[2]
        if_w = FunctionWrapper(if_func, 3, 'if')
        def is_greater(l):
            return 1 if l[0] > l[1] else 0
        gt_w = FunctionWrapper(is_greater, 2, 'isgreater')
        func_list = [add_w, sub_w, if_w, gt_w]

        # The game environment
        def grid_game(p1: Node, p2: Node) -> int:
            """Plays one game. Returns 0 if p1 wins, 1 if p2 wins, -1 for a tie."""
            max_pos = (3, 3)
            location = [[randint(0, max_pos[0]), randint(0, max_pos[1])]]
            location.append([(location[0][0] + 2) % 4, (location[0][1] + 2) % 4])
            last_move = [-1, -1]
            players = [p1, p2]

            for _ in range(50): # 50 move limit
                for i in range(2):
                    # Inputs: my_x, my_y, other_x, other_y, my_last_move
                    inputs = location[i] + location[1-i] + [last_move[i]]
                    move = players[i].evaluate(inputs) % 4

                    if last_move[i] == move:
                        return 1 - i # Lose for repeating move
                    last_move[i] = move

                    if move == 0:
                        location[i][0] = max(0, location[i][0] - 1)
                    elif move == 1:
                        location[i][0] = min(max_pos[0], location[i][0] + 1)
                    elif move == 2:
                        location[i][1] = max(0, location[i][1] - 1)
                    elif move == 3:
                        location[i][1] = min(max_pos[1], location[i][1] + 1)

                    if location[i] == location[1-i]:
                        return i # Win by capture
            return -1 # Tie

        # The rank function for the tournament
        def tournament_ranker(population: List[Node]):
            losses = [0] * len(population)
            for i in range(len(population)):
                for j in range(len(population)):
                    if i == j:
                        continue
                    winner = grid_game(population[i], population[j])
                    if winner == 0:
                        losses[j] += 2 # p2 lost
                    elif winner == 1:
                        losses[i] += 2 # p1 lost
                    elif winner == -1: # Tie
                        losses[i] += 1
                        losses[j] += 1

            scores = sorted(zip(losses, population), key=lambda x: x[0])
            return scores

        # Initialize and run the GP
        gp_ai = GeneticProgramming(
            param_count=5, # my_x, my_y, other_x, other_y, last_move
            func_list=func_list,
            population_size=100,
            mutation_rate=0.3,
            breeding_rate=0.5,
        )
        gp_ai.evolve(tournament_ranker, max_gen=50)

    # --- Main Execution ---
    run_symbolic_regression()
    run_game_ai_evolution()
