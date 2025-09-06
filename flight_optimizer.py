"""
A module for solving the group travel optimization problem using various
metaheuristic algorithms.

The goal is to find the best flight schedule (outbound and return) for a group
of people traveling from different origins to a single destination. "Best" is
defined by a cost function that includes total ticket price, total waiting time
at the airport, and penalties for overnight stays.

This script implements four optimization algorithms to solve this problem:
1. Random Search: A simple baseline that tests random solutions.
2. Hill Climbing: A local search algorithm that iteratively finds better
   neighboring solutions.
3. Simulated Annealing: A probabilistic algorithm that can escape local optima
   by sometimes accepting worse solutions.
4. Genetic Algorithm: A population-based algorithm that "evolves" a solution
   through selection, crossover, and mutation.

To run this script, you need a `schedule.txt` file in the same directory.
Each line should be a comma-separated value (CSV) with the format:
ORIGIN,DESTINATION,DEPART_TIME,ARRIVE_TIME,PRICE

Example `schedule.txt`:
BOS,LGA,10:22,12:32,248
BOS,LGA,15:00,17:15,222
DAL,LGA,11:45,15:10,345
...
"""

import time
import random
import math
from typing import List, Tuple, Dict, Callable, NamedTuple

# --- Data Structures and Problem Definition ---

class Flight(NamedTuple):
    """Represents a single flight with its schedule and price."""
    depart_time: str
    arrive_time: str
    price: int

Person = Tuple[str, str]  # (Name, Origin_Airport_Code)

# A list of people and their respective origin airports.
PEOPLE: List[Person] = [
    ('Seymour', 'BOS'), ('Franny', 'DAL'), ('Zooey', 'CAK'),
    ('Walt', 'MIA'), ('Buddy', 'ORD'), ('Les', 'OMA')
]

# The common destination for all travelers.
DESTINATION = 'LGA'

def load_flight_data(filename: str = 'schedule.txt') -> Dict[Tuple[str, str], List[Flight]]:
    """
    Loads flight data from a specified file.

    Args:
        filename: The name of the file containing flight schedules.

    Returns:
        A dictionary where keys are (origin, destination) tuples and values
        are lists of available `Flight` objects for that route.
    """
    flights: Dict[Tuple[str, str], List[Flight]] = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                origin, dest, depart, arrive, price = line.strip().split(',')
                route = (origin, dest)
                flights.setdefault(route, [])
                flights[route].append(Flight(depart, arrive, int(price)))
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        print("Please create it with the format: ORIGIN,DEST,DEPART,ARRIVE,PRICE")
        exit()
    return flights

def get_minutes(t: str) -> int:
    """Converts a 'HH:MM' time string to minutes since midnight."""
    x = time.strptime(t, '%H:%M')
    return x.tm_hour * 60 + x.tm_min

def print_schedule(solution: List[int], flights: Dict[Tuple[str, str], List[Flight]]) -> None:
    """
    Prints a flight schedule in a human-readable format.

    Args:
        solution: A list of flight indices representing the schedule.
        flights: The dictionary of all available flights.
    """
    print("-" * 65)
    print(f"{'Name':>10}{'Origin':>10} {'Outbound':>12} {'Cost':>5} {'Return':>14} {'Cost':>5}")
    print("-" * 65)

    # Each person has an outbound and a return flight, so we iterate in pairs.
    for i in range(len(solution) // 2):
        name, origin = PEOPLE[i]
        out_idx = solution[i * 2]
        ret_idx = solution[i * 2 + 1]

        outbound_flight = flights[(origin, DESTINATION)][out_idx]
        return_flight = flights[(DESTINATION, origin)][ret_idx]

        print(
            f"{name:>10}{origin:>10} "
            f"{outbound_flight.depart_time:>5}-{outbound_flight.arrive_time:<5} ${outbound_flight.price:>3} "
            f"{return_flight.depart_time:>6}-{return_flight.arrive_time:<5} ${return_flight.price:>3}"
        )

def schedule_cost(solution: List[int], flights: Dict[Tuple[str, str], List[Flight]]) -> int:
    """
    Calculates the total cost of a given flight schedule solution.

    The cost includes total flight price, total waiting time, and a car
    rental penalty if an overnight stay is required.

    Args:
        solution: A list of flight indices [o1, r1, o2, r2, ...], where o1 is
                  the outbound flight index for person 1, r1 is the return, etc.
        flights: The dictionary of all available flights.

    Returns:
        The total cost of the schedule.
    """
    total_price = 0
    latest_arrival = 0
    earliest_departure = 24 * 60

    for i in range(len(solution) // 2):
        origin = PEOPLE[i][1]
        outbound = flights[(origin, DESTINATION)][solution[i * 2]]
        return_flight = flights[(DESTINATION, origin)][solution[i * 2 + 1]]

        total_price += outbound.price
        total_price += return_flight.price

        # Track the latest arrival and earliest departure for the group
        latest_arrival = max(latest_arrival, get_minutes(outbound.arrive_time))
        earliest_departure = min(earliest_departure, get_minutes(return_flight.depart_time))

    # Calculate total waiting time
    total_wait = 0
    for i in range(len(solution) // 2):
        origin = PEOPLE[i][1]
        outbound = flights[(origin, DESTINATION)][solution[i * 2]]
        return_flight = flights[(DESTINATION, origin)][solution[i * 2 + 1]]

        # Wait time upon arrival
        total_wait += latest_arrival - get_minutes(outbound.arrive_time)
        # Wait time for departure
        total_wait += get_minutes(return_flight.depart_time) - earliest_departure

    # Add a penalty for car rental if the trip requires an overnight stay
    if latest_arrival > earliest_departure:
        total_price += 50

    return total_price + total_wait

# --- Optimization Algorithms ---

SolutionType = List[int]
DomainType = List[Tuple[int, int]]
CostFunctionType = Callable[[SolutionType], int]

def random_search(domain: DomainType, cost_function: CostFunctionType, iterations: int = 1000) -> SolutionType:
    """
    Finds a solution using random search.

    Args:
        domain: A list of (min, max) tuples for each variable in the solution.
        cost_function: The function to evaluate the cost of a solution.
        iterations: The number of random solutions to test.

    Returns:
        The best solution found.
    """
    best_cost = float('inf')
    best_solution = []
    for _ in range(iterations):
        solution = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
        cost = cost_function(solution)
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
    return best_solution

def hill_climb(domain: DomainType, cost_function: CostFunctionType) -> SolutionType:
    """
    Finds a solution using the hill climbing algorithm.

    Starts with a random solution and iteratively moves to the best
    neighboring solution until no improvement can be made.

    Args:
        domain: A list of (min, max) tuples for each variable.
        cost_function: The function to evaluate the cost of a solution.

    Returns:
        A locally optimal solution.
    """
    solution = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]

    while True:
        neighbors = []
        for i in range(len(domain)):
            # Create neighbors by moving one step in each direction
            if solution[i] > domain[i][0]:
                neighbors.append(solution[:i] + [solution[i] - 1] + solution[i+1:])
            if solution[i] < domain[i][1]:
                neighbors.append(solution[:i] + [solution[i] + 1] + solution[i+1:])

        current_cost = cost_function(solution)
        best_cost = current_cost

        for neighbor in neighbors:
            cost = cost_function(neighbor)
            if cost < best_cost:
                best_cost = cost
                solution = neighbor

        # If no neighbor is better, we have reached a local optimum
        if best_cost == current_cost:
            break

    return solution

def simulated_annealing(domain: DomainType, cost_function: CostFunctionType,
                        temp: float = 10000.0, cool_rate: float = 0.95, step: int = 1) -> SolutionType:
    """
    Finds a solution using simulated annealing.

    This algorithm can escape local optima by occasionally accepting worse
    solutions, with the probability of acceptance decreasing over time as the
    "temperature" cools.

    Args:
        domain: A list of (min, max) tuples for each variable.
        cost_function: The function to evaluate the cost of a solution.
        temp: The initial temperature. Higher values allow more exploration.
        cool_rate: The rate at which the temperature decreases in each iteration.
        step: The maximum size of a random change to the solution.

    Returns:
        The best solution found.
    """
    solution = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]

    while temp > 0.1:
        # Choose a random index and a random change
        i = random.randint(0, len(domain) - 1)
        change = random.randint(-step, step)

        # Create a new solution with the change, respecting domain bounds
        new_solution = solution[:]
        new_solution[i] = max(domain[i][0], min(domain[i][1], new_solution[i] + change))

        current_cost = cost_function(solution)
        new_cost = cost_function(new_solution)

        # Acceptance probability (Metropolis-Hastings criterion)
        try:
            prob = math.exp((current_cost - new_cost) / temp)
        except OverflowError:
            prob = float('inf')

        # Decide whether to accept the new solution
        if new_cost < current_cost or random.random() < prob:
            solution = new_solution

        # Cool the temperature
        temp *= cool_rate

    return solution

def genetic_algorithm(domain: DomainType, cost_function: CostFunctionType,
                      pop_size: int = 50, elite_frac: float = 0.2,
                      mutation_prob: float = 0.2, max_iter: int = 100) -> SolutionType:
    """
    Finds a solution using a genetic algorithm.

    Evolves a population of solutions over generations using selection,
    crossover, and mutation to find a high-quality solution.

    Args:
        domain: A list of (min, max) tuples for each variable.
        cost_function: The function to evaluate the cost of a solution.
        pop_size: The number of solutions in the population.
        elite_frac: The fraction of the best solutions to carry over to the
                    next generation.
        mutation_prob: The probability of a new solution being a mutation
                       rather than a crossover.
        max_iter: The number of generations to run.

    Returns:
        The best solution found after all generations.
    """
    def mutate(solution: SolutionType) -> SolutionType:
        """Slightly changes one gene in a solution."""
        i = random.randint(0, len(domain) - 1)
        mutated = solution[:]
        if random.random() < 0.5 and solution[i] > domain[i][0]:
            mutated[i] -= 1
        elif solution[i] < domain[i][1]:
            mutated[i] += 1
        return mutated

    def crossover(sol1: SolutionType, sol2: SolutionType) -> SolutionType:
        """Combines two parent solutions to create a child."""
        i = random.randint(1, len(domain) - 2)
        return sol1[:i] + sol2[i:]

    # Build the initial population
    population = [[random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
                  for _ in range(pop_size)]

    num_elites = int(elite_frac * pop_size)

    print("Running Genetic Algorithm...")
    for i in range(max_iter):
        # Rank the population by cost
        scores = sorted([(cost_function(s), s) for s in population], key=lambda x: x[0])
        ranked_solutions = [s for cost, s in scores]

        # Start the next generation with the elites
        population = ranked_solutions[:num_elites]

        # Add new members through mutation and crossover
        while len(population) < pop_size:
            if random.random() < mutation_prob:
                # Mutate a random elite member
                c = random.randint(0, num_elites - 1)
                population.append(mutate(ranked_solutions[c]))
            else:
                # Crossover two random elite members
                c1 = random.randint(0, num_elites - 1)
                c2 = random.randint(0, num_elites - 1)
                population.append(crossover(ranked_solutions[c1], ranked_solutions[c2]))

        if (i + 1) % 20 == 0:
            print(f"Generation {i+1}, Best Cost: {scores[0][0]}")

    return scores[0][1]

if __name__ == '__main__':
    # 1. Load the flight data
    flight_data = load_flight_data('./data/schedule.txt')

    # 2. Define the domain for the solution vector
    # Each person needs an outbound and a return flight index.
    # The domain for each is (0, number_of_available_flights - 1).
    problem_domain: DomainType = []
    for name, origin in PEOPLE:
        # Domain for outbound flight
        num_outbound = len(flight_data.get((origin, DESTINATION), []))
        problem_domain.append((0, num_outbound - 1))
        # Domain for return flight
        num_return = len(flight_data.get((DESTINATION, origin), []))
        problem_domain.append((0, num_return - 1))

    # 3. Create a cost function that has the flight data "baked in"
    # This simplifies passing it to the optimization functions.
    cost_func = lambda sol: schedule_cost(sol, flight_data)

    # 4. Run one of the optimization algorithms
    print("\n--- Solving with Simulated Annealing ---")
    final_solution = simulated_annealing(problem_domain, cost_func)

    # You can uncomment and try other algorithms as well
    # print("\n--- Solving with Genetic Algorithm ---")
    # final_solution = genetic_algorithm(problem_domain, cost_func)

    # 5. Print the final results
    final_cost = cost_func(final_solution)
    print(f"\nOptimal schedule found with a total cost of: ${final_cost}")
    print_schedule(final_solution, flight_data)
