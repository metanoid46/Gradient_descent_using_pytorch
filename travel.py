import sys
sys.setrecursionlimit(100000)

# Model
class TravelProblem(object):
    def __init__(self, N):
        # N = number of blocks
        self.N = N
        self.best_solution = None  # Store best (cost, path)

    def startState(self):
        return 1

    def isEnd(self, state):
        return state == self.N

    def succAndCost(self, state):
        # Return a list of (action, newState, cost) tuples
        result = []
        if state + 1 <= self.N:
            result.append(('walk', state + 1, 1))
        if state * 2 <= self.N:
            result.append(('bus', state * 2, 2))
        return result

    # Backtracking algorithm to find the optimal path
    def backtracking(self, state, path=[], cost=0):

        if self.isEnd(state):  # Goal reached

            if self.best_solution is None or cost < self.best_solution[0]:
                self.best_solution = (cost, path[:])  # Store the best solution
            return

        for action, newState, stepCost in self.succAndCost(state):  # Fix variable name
            path.append((action, newState))  # Add to path
            self.backtracking(newState, path, cost + stepCost)  # Recursive call
            path.pop()  # Remove last step to backtrack

    def findBestPath(self):
        print("Starting backtracking search...")
        self.backtracking(self.startState())
        return self.best_solution

# Print the solution
def printSolution(solution):
    if solution is None:
        print("No solution found.")
        return

    totalCost, history = solution
    print('Total cost:', totalCost)
    for step in history:
        print(step)

# Run the algorithm
problem = TravelProblem(10)  # Example: Find best path to block 10
solution = problem.findBestPath()
printSolution(solution)
