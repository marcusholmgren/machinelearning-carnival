"""
A script to solve the Dorm Room Assignment optimization problem.

This script aims to assign a group of students to dorm rooms in a way that
minimizes their overall dissatisfaction. Each student has a first and second
choice for their dorm, and each dorm has a limited number of spaces (two in
this case).

The Problem is framed as follows:
- A "solution" is a list of choices, one for each student.
- The "cost function" (`dorm_cost`) evaluates a solution by assigning penalty
  points: 0 for a first choice, 1 for a second choice, and 3 for any other dorm.
- The goal is to find the solution with the minimum total cost.

The solution is represented by a vector where each element corresponds to a
student's choice of an available room slot from a dynamically shrinking list.
This script includes a simple random search optimizer to find a good solution.
"""

import random
from typing import List, Tuple, NamedTuple

# --- Problem Definition ---

# Define a more structured way to hold student preference data
StudentPref = NamedTuple("StudentPref", [('name', str), ('choices', Tuple[str, str])])

# The dorms, each of which has two available spaces
DORMS: List[str] = ['Zeus', 'Athena', 'Hercules', 'Bacchus', 'Pluto']

# People, along with their first and second choices
PREFS: List[StudentPref] = [
    StudentPref('Toby', ('Bacchus', 'Hercules')),
    StudentPref('Steve', ('Zeus', 'Pluto')),
    StudentPref('Andrea', ('Athena', 'Zeus')),
    StudentPref('Sarah', ('Zeus', 'Pluto')),
    StudentPref('Dave', ('Athena', 'Bacchus')),
    StudentPref('Jeff', ('Hercules', 'Pluto')),
    StudentPref('Fred', ('Pluto', 'Athena')),
    StudentPref('Suzie', ('Bacchus', 'Hercules')),
    StudentPref('Laura', ('Bacchus', 'Hercules')),
    StudentPref('Neil', ('Hercules', 'Athena'))
]

# Define costs for assignments to improve readability
FIRST_CHOICE_COST = 0
SECOND_CHOICE_COST = 1
UNLISTED_CHOICE_COST = 3


def print_solution(solution_vector: List[int]) -> None:
    """
    Decodes a solution vector and prints the resulting dorm assignments.

    The vector represents choices from a shrinking list of available slots.
    For example, vec[0] is the index of the slot chosen by the first student
    from all 10 slots. vec[1] is the index chosen by the second student from
    the remaining 9 slots, and so on.

    Args:
        solution_vector: A list of integers representing the encoded solution.
    """
    # Create a list of available slots, with two for each dorm (e.g., [0, 0, 1, 1, ...])
    # representing [Zeus, Zeus, Athena, Athena, ...]
    slots = [i for i in range(len(DORMS)) for _ in range(2)]

    print("\n--- Dorm Assignments ---")
    # Loop over each student's assignment choice in the solution vector
    for i, student_choice_idx in enumerate(solution_vector):
        # The choice is an index into the *current* list of available slots
        chosen_slot = slots[student_choice_idx]
        dorm_name = DORMS[chosen_slot]
        student_name = PREFS[i].name

        print(f"{student_name:<10} -> {dorm_name}")

        # Remove this slot from the list of available slots for the next student
        del slots[student_choice_idx]


def dorm_cost(solution_vector: List[int]) -> int:
    """
    Calculates the total cost of a given solution vector.

    The cost function quantifies how good a particular assignment is. A lower
    cost is better. It works by decoding the solution vector using the same
    "shrinking list" logic as `print_solution`.

    Args:
        solution_vector: A list of integers representing the encoded solution.

    Returns:
        The total cost (integer) of the solution.
    """
    cost = 0
    # Create the initial list of available slots dynamically
    slots = [i for i in range(len(DORMS)) for _ in range(2)]

    # Loop over each student's assignment
    for i, student_choice_idx in enumerate(solution_vector):
        dorm_idx = slots[student_choice_idx]
        dorm_name = DORMS[dorm_idx]
        student_prefs = PREFS[i].choices

        # Assign cost based on preference
        if student_prefs[0] == dorm_name:
            cost += FIRST_CHOICE_COST
        elif student_prefs[1] == dorm_name:
            cost += SECOND_CHOICE_COST
        else:
            cost += UNLISTED_CHOICE_COST  # The student got neither of their choices

        # Remove the chosen slot
        del slots[student_choice_idx]

    return cost


def random_search_solver(
    num_students: int,
    iterations: int = 10000
) -> Tuple[List[int], int]:
    """
    Finds a good solution using a simple random search algorithm.

    This function generates many random solutions, evaluates their cost, and
    keeps track of the best one found.

    Args:
        num_students: The number of students to assign.
        iterations: The number of random solutions to generate and test.

    Returns:
        A tuple containing the best solution vector found and its cost.
    """
    best_cost = float('inf')
    best_solution = []

    print(f"\nSearching for the best solution over {iterations} iterations...")

    for i in range(iterations):
        # Create a random solution vector.
        # The choice for the first student is from 10 slots (0-9),
        # the second from 9 (0-8), and so on.
        random_solution = [random.randint(0, num_students - j - 1) for j in range(num_students)]

        current_cost = dorm_cost(random_solution)

        if current_cost < best_cost:
            best_cost = current_cost
            best_solution = random_solution

    print("Search complete.")
    return best_solution, best_cost


if __name__ == '__main__':
    # The number of students determines the length of the solution vector.
    # It must match the length of the PREFS list.
    number_of_students = len(PREFS)

    # --- Run the optimizer to find the best assignment ---
    # Because this is a random search, results may vary slightly on each run.
    best_solution_found, lowest_cost = random_search_solver(number_of_students)

    # --- Print the results ---
    print(f"\nBest solution found has a total cost of: {lowest_cost}")
    print_solution(best_solution_found)
