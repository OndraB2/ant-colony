# ant-colony

This project implements the Ant Colony Optimization (ACO) algorithm, a bio-inspired heuristic for solving complex optimization problems. ACO is based on the foraging behavior of real ant colonies, where ants collectively find the shortest paths between their nest and a food source using pheromone trails. This project was developed as a school assignment in 2023.

## How It Works

1. **Initialization**  
   - A set of artificial ants is placed randomly in the solution space.  
   - A pheromone matrix is initialized to guide the search process.  

2. **Solution Construction**  
   - Each ant builds a solution step-by-step based on pheromone levels and problem-specific heuristics.  
   - The probability of selecting a path depends on:  
     - The amount of pheromone deposited.  
     - A heuristic function (e.g., distance for the Traveling Salesman Problem).  

3. **Pheromone Update**  
   - After all ants complete a solution, pheromone levels are updated.  
   - **Evaporation:** A portion of pheromone evaporates to prevent stagnation.  
   - **Reinforcement:** Paths taken by the best solutions receive additional pheromone, making them more attractive for future ants.  

4. **Iteration and Convergence**  
   - The process repeats over multiple iterations until a stopping condition is met (e.g., a maximum number of iterations or convergence to an optimal solution).  
