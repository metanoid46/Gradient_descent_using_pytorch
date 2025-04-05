def backtrack_iterative(maze, start, end):
    stack = [(start, [start])]  # Stack stores (current node, path)
    print("Initial stack:", stack)  # Debug: Initial stack state
    visited = set()

    while stack:
        current, path = stack.pop()  # Get the last node in the stack
        print("Current node:", current)
        print("Current path:", path)
        print("Visited nodes:", visited)
        if current == end:
            return path  # If we reach the destination, return the path

        if current not in visited:
            visited.add(current)
            print("Visited:", visited)  # Debug: Mark current node as visited

            for neighbor in maze.get(current, []):  # Process neighbors
                if neighbor not in visited:
                    print("Exploring neighbor:", neighbor)
                    stack.append((neighbor, path + [neighbor]))  # Add new path to stack
                    print("Stack updated:", stack)

    return None  # No path found

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

# Run iterative backtracking search
path = backtrack_iterative(maze, 0, 8)
print("Path found:", path if path else "No path found")