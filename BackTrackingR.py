def backtrack_recursive(maze, start, end, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    # Add current node to path
    path.append(start)
    visited.add(start)

    # Base case: If start reaches end, return path
    if start == end:
        return path

    # Explore neighbors recursively
    for neighbor in maze.get(start, []):
        if neighbor not in visited:
            result = backtrack_recursive(maze, neighbor, end, visited, path)  # Recursive call
            
            if result:  # If a valid path is found, return it
                return result

    # If no path is found, backtrack
    path.pop()
    return None

# Define the maze
maze = {
    0: [1, 3],
    1: [2],
    2: [],
    3: [4, 5],
    4: [6, 7],
    5: [],
    6: [],
    7: [8]
}

# Run recursive backtracking search
path = backtrack_recursive(maze, 0, 8)
print("Path found:", path if path else "No path found")