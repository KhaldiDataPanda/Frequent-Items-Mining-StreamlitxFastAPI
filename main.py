from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import numpy as np
from typing import Dict, List, Any, Union
import uvicorn
import math

app = FastAPI(title="Frequent Items Mining API", description="API for finding frequent items using various metaheuristic algorithms")

class DataGenerationRequest(BaseModel):
    data_size: int = 2000
    min_list_length: int = 1
    max_list_length: int = 6
    mean_list_length: float = 3.0
    std_list_length: float = 1.0
    min_item_id: int = 1
    max_item_id: int = 6
    distribution_type: str = "gaussian"  # "gaussian", "uniform", "poisson", "exponential", "binomial"
    lambda_param: float = 3.0  # For Poisson and Exponential distributions
    prob_param: float = 0.5    # For Binomial distribution
    trials_param: int = 10     # For Binomial distribution (n trials)

class GAParameters(BaseModel):
    population_size: int = 100
    num_generations: int = 100
    num_parents: int = 50
    mutation_rate: float = 0.1

class PSOParameters(BaseModel):
    num_particles: int = 50
    num_iterations: int = 200
    w: float = 0.9  # Inertia weight
    c1: float = 2.0  # Cognitive parameter
    c2: float = 2.0  # Social parameter

class SAParameters(BaseModel):
    initial_temp: float = 1000.0
    cooling_rate: float = 0.95
    min_temp: float = 1.0
    max_iterations: int = 1000

class ACOParameters(BaseModel):
    num_ants: int = 50
    num_iterations: int = 100
    alpha: float = 1.0  # Pheromone importance
    beta: float = 2.0   # Heuristic importance
    evaporation_rate: float = 0.5
    q: float = 100.0    # Pheromone deposit factor

class MiningRequest(BaseModel):
    algorithm: str  # "GA", "PSO", "SA", "ACO"
    data_generation: DataGenerationRequest
    ga_parameters: Union[GAParameters, None] = None
    pso_parameters: Union[PSOParameters, None] = None
    sa_parameters: Union[SAParameters, None] = None
    aco_parameters: Union[ACOParameters, None] = None

class MiningResponse(BaseModel):
    algorithm_used: str
    most_occurred_item: str
    item_counts: Dict[str, int]
    data_size: int
    parameters: Dict[str, Any]
    execution_time: float

def generate_data(params: DataGenerationRequest) -> Dict[int, List[str]]:
    """Generate synthetic transaction data based on parameters and selected distribution."""
    data_dict = {}
    
    def generate_list_length():
        """Generate list length based on the selected distribution."""
        if params.distribution_type == "gaussian":
            # Gaussian (Normal) distribution - default behavior
            length = round(random.gauss(params.mean_list_length, params.std_list_length))
            
        elif params.distribution_type == "uniform":
            # Uniform distribution between min and max
            length = random.randint(params.min_list_length, params.max_list_length)
            
        elif params.distribution_type == "poisson":
            # Poisson distribution - good for count data
            length = np.random.poisson(params.lambda_param)
            
        elif params.distribution_type == "exponential":
            # Exponential distribution - for modeling time between events
            length = round(np.random.exponential(1/params.lambda_param))
            
        elif params.distribution_type == "binomial":
            # Binomial distribution - for number of successes in n trials
            length = np.random.binomial(params.trials_param, params.prob_param)
            
        else:
            # Fallback to gaussian if unknown distribution
            length = round(random.gauss(params.mean_list_length, params.std_list_length))
        
        # Ensure list length is within bounds regardless of distribution
        length = max(params.min_list_length, min(params.max_list_length, length))
        return length
    
    while len(data_dict) < params.data_size:
        # Generate list length using selected distribution
        list_length = generate_list_length()
        
        # Generate items for this transaction
        items = [f"item{np.random.randint(low=params.min_item_id, high=params.max_item_id+1)}" 
                for _ in range(list_length)]
        
        data_dict[len(data_dict)] = items
    
    return data_dict

def fitness_function(candidate):
    """Calculate the occurrence count of each item in the candidate solution."""
    occurrence_count = {}
    for item_list in candidate:
        for item in item_list:
            if item in occurrence_count:
                occurrence_count[item] += 1
            else:
                occurrence_count[item] = 1
    return occurrence_count

def generate_initial_population(dictionary, population_size):
    """Generate an initial population of candidate solutions."""
    population = []
    for _ in range(population_size):
        candidate = list(dictionary.values())
        random.shuffle(candidate)
        population.append(tuple(candidate))
    return population

def select_parents(population, fitness_scores, num_parents):
    """Select parents for reproduction using tournament selection."""
    parents = []
    for _ in range(num_parents):
        tournament_size = min(3, len(population))
        tournament = random.sample(range(len(population)), tournament_size)
        selected_parent = tournament[0]
        for i in tournament[1:]:
            if sum(fitness_scores[i].values()) > sum(fitness_scores[selected_parent].values()):
                selected_parent = i
        parents.append(population[selected_parent])
    return parents

def crossover(parents, offspring_size):
    """Apply crossover to create new offspring."""
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = random.sample(parents, 2)
        crossover_point = random.randint(1, len(parent1) - 1)
        offspring.append(parent1[:crossover_point] + parent2[crossover_point:])
    return offspring

def mutate(offspring, mutation_rate):
    """Apply mutation to the offspring."""
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, len(offspring[i]) - 1)
            offspring[i] = list(offspring[i])
            offspring[i][mutation_point] = random.choice(offspring[i])
            offspring[i] = tuple(offspring[i])
    return offspring

# Genetic Algorithm Implementation
def genetic_algorithm(dictionary, params: GAParameters):
    """Run genetic algorithm to find most frequent item."""
    population = generate_initial_population(dictionary, params.population_size)
    
    for _ in range(params.num_generations):
        fitness_scores = [fitness_function(candidate) for candidate in population]
        parents = select_parents(population, fitness_scores, params.num_parents)
        offspring = crossover(parents, params.population_size - params.num_parents)
        offspring = mutate(offspring, params.mutation_rate)
        population = parents + offspring
    
    best_candidate = max(population, key=lambda candidate: sum(fitness_function(candidate).values()))
    item_counts = fitness_function(best_candidate)
    most_occurred_item = max(item_counts, key=item_counts.get)
    
    return most_occurred_item, item_counts

# Particle Swarm Optimization Implementation
class Particle:
    def __init__(self, data_items):
        self.position = list(data_items)
        random.shuffle(self.position)
        self.velocity = [random.uniform(-1, 1) for _ in range(len(self.position))]
        self.best_position = self.position.copy()
        self.best_fitness = self._calculate_fitness()
        
    def _calculate_fitness(self):
        item_counts = fitness_function([self.position])
        return sum(item_counts.values()) if item_counts else 0
    
    def update_velocity(self, global_best_position, w, c1, c2):
        for i in range(len(self.velocity)):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (hash(str(self.best_position[i])) - hash(str(self.position[i])))
            social = c2 * r2 * (hash(str(global_best_position[i])) - hash(str(self.position[i])))
            self.velocity[i] = w * self.velocity[i] + cognitive + social
    
    def update_position(self, data_items):
        indices = list(range(len(self.position)))
        # Sort indices by velocity (descending)
        indices.sort(key=lambda i: self.velocity[i], reverse=True)
        # Reorder position based on velocity
        self.position = [data_items[i] for i in indices[:len(data_items)]]
        
        current_fitness = self._calculate_fitness()
        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            self.best_position = self.position.copy()

def particle_swarm_optimization(dictionary, params: PSOParameters):
    """Run PSO algorithm to find most frequent item."""
    data_items = list(dictionary.values())
    particles = [Particle(data_items) for _ in range(params.num_particles)]
    
    global_best_particle = max(particles, key=lambda p: p.best_fitness)
    
    for _ in range(params.num_iterations):
        for particle in particles:
            particle.update_velocity(global_best_particle.best_position, params.w, params.c1, params.c2)
            particle.update_position(data_items)
            
            if particle.best_fitness > global_best_particle.best_fitness:
                global_best_particle = particle
    
    item_counts = fitness_function([global_best_particle.best_position])
    most_occurred_item = max(item_counts, key=item_counts.get) if item_counts else "item1"
    
    return most_occurred_item, item_counts

# Simulated Annealing Implementation
def simulated_annealing(dictionary, params: SAParameters):
    """Run simulated annealing to find most frequent item."""
    current_solution = list(dictionary.values())
    random.shuffle(current_solution)
    current_fitness = sum(fitness_function(current_solution).values())
    
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    temperature = params.initial_temp
    
    for iteration in range(params.max_iterations):
        if temperature < params.min_temp:
            break
            
        # Generate neighbor solution by swapping two random positions
        new_solution = current_solution.copy()
        i, j = random.sample(range(len(new_solution)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        
        new_fitness = sum(fitness_function(new_solution).values())
        
        # Accept or reject the new solution
        if new_fitness > current_fitness or random.random() < math.exp((new_fitness - current_fitness) / temperature):
            current_solution = new_solution
            current_fitness = new_fitness
            
            if current_fitness > best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
        
        temperature *= params.cooling_rate
    
    item_counts = fitness_function(best_solution)
    most_occurred_item = max(item_counts, key=item_counts.get) if item_counts else "item1"
    
    return most_occurred_item, item_counts

# Ant Colony Optimization Implementation
def ant_colony_optimization(dictionary, params: ACOParameters):
    """Run ACO algorithm to find most frequent item."""
    data_items = list(dictionary.values())
    num_items = len(data_items)
    
    # Initialize pheromone matrix
    pheromones = [[1.0 for _ in range(num_items)] for _ in range(num_items)]
    
    best_solution = None
    best_fitness = 0
    
    for iteration in range(params.num_iterations):
        solutions = []
        
        for ant in range(params.num_ants):
            solution = []
            available_items = list(range(num_items))
            
            while available_items:
                if len(solution) == 0:
                    next_item = random.choice(available_items)
                else:
                    # Calculate probabilities
                    probabilities = []
                    current_item = solution[-1]
                    
                    for item in available_items:
                        pheromone = pheromones[current_item][item] ** params.alpha
                        heuristic = (1.0 / (item + 1)) ** params.beta  # Simple heuristic
                        probabilities.append(pheromone * heuristic)
                    
                    # Normalize probabilities
                    total_prob = sum(probabilities)
                    if total_prob > 0:
                        probabilities = [p / total_prob for p in probabilities]
                        next_item = np.random.choice(available_items, p=probabilities)
                    else:
                        next_item = random.choice(available_items)
                
                solution.append(next_item)
                available_items.remove(next_item)
            
            # Convert indices back to actual items
            ant_solution = [data_items[i] for i in solution]
            solutions.append((solution, ant_solution))
        
        # Evaluate solutions and update pheromones
        for solution_indices, solution_items in solutions:
            fitness = sum(fitness_function([solution_items]).values())
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution_items
            
            # Update pheromones
            pheromone_deposit = params.q / (1 + fitness)  # Inverse fitness for minimization-like update
            for i in range(len(solution_indices) - 1):
                current_idx = solution_indices[i]
                next_idx = solution_indices[i + 1]
                pheromones[current_idx][next_idx] += pheromone_deposit
        
        # Evaporate pheromones
        for i in range(num_items):
            for j in range(num_items):
                pheromones[i][j] *= (1 - params.evaporation_rate)
                pheromones[i][j] = max(pheromones[i][j], 0.01)  # Minimum pheromone level
    
    if best_solution is None:
        best_solution = data_items
    
    item_counts = fitness_function([best_solution])
    most_occurred_item = max(item_counts, key=item_counts.get) if item_counts else "item1"
    
    return most_occurred_item, item_counts

@app.get("/")
async def root():
    return {"message": "Frequent Items Mining API", "docs": "/docs"}

@app.post("/mine-frequent-items", response_model=MiningResponse)
async def mine_frequent_items(request: MiningRequest):
    """Mine frequent items using the selected metaheuristic algorithm."""
    import time
    
    try:
        # Generate data
        data_dict = generate_data(request.data_generation)
        
        start_time = time.time()
        
        # Run the selected algorithm
        if request.algorithm == "GA":
            if not request.ga_parameters:
                raise HTTPException(status_code=400, detail="GA parameters required for Genetic Algorithm")
            most_occurred_item, item_counts = genetic_algorithm(data_dict, request.ga_parameters)
            algorithm_params = request.ga_parameters.dict()
            
        elif request.algorithm == "PSO":
            if not request.pso_parameters:
                raise HTTPException(status_code=400, detail="PSO parameters required for Particle Swarm Optimization")
            most_occurred_item, item_counts = particle_swarm_optimization(data_dict, request.pso_parameters)
            algorithm_params = request.pso_parameters.dict()
            
        elif request.algorithm == "SA":
            if not request.sa_parameters:
                raise HTTPException(status_code=400, detail="SA parameters required for Simulated Annealing")
            most_occurred_item, item_counts = simulated_annealing(data_dict, request.sa_parameters)
            algorithm_params = request.sa_parameters.dict()
            
        elif request.algorithm == "ACO":
            if not request.aco_parameters:
                raise HTTPException(status_code=400, detail="ACO parameters required for Ant Colony Optimization")
            most_occurred_item, item_counts = ant_colony_optimization(data_dict, request.aco_parameters)
            algorithm_params = request.aco_parameters.dict()
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {request.algorithm}")
        
        execution_time = time.time() - start_time
        
        return MiningResponse(
            algorithm_used=request.algorithm,
            most_occurred_item=most_occurred_item,
            item_counts=item_counts,
            data_size=len(data_dict),
            parameters=algorithm_params,
            execution_time=execution_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/algorithms")
async def get_available_algorithms():
    """Get list of available algorithms and their parameter schemas."""
    return {
        "algorithms": [
            {
                "name": "GA",
                "display_name": "Genetic Algorithm",
                "description": "Evolutionary algorithm using selection, crossover, and mutation"
            },
            {
                "name": "PSO", 
                "display_name": "Particle Swarm Optimization",
                "description": "Swarm intelligence algorithm inspired by bird flocking behavior"
            },
            {
                "name": "SA",
                "display_name": "Simulated Annealing", 
                "description": "Probabilistic technique inspired by annealing in metallurgy"
            },
            {
                "name": "ACO",
                "display_name": "Ant Colony Optimization",
                "description": "Swarm intelligence algorithm inspired by ant foraging behavior"
            }
        ]
    }

@app.get("/distributions")
async def get_available_distributions():
    """Get list of available data distributions for transaction generation."""
    return {
        "distributions": [
            {
                "name": "gaussian",
                "display_name": "📈 Gaussian (Normal)",
                "description": "Bell-shaped distribution, good for natural phenomena",
                "parameters": ["mean_list_length", "std_list_length"],
                "use_cases": "Most transactions have similar lengths with some variation"
            },
            {
                "name": "uniform",
                "display_name": "📊 Uniform",
                "description": "Equal probability for all values in range",
                "parameters": ["min_list_length", "max_list_length"],
                "use_cases": "All transaction lengths are equally likely"
            },
            {
                "name": "poisson",
                "display_name": "🎯 Poisson",
                "description": "Models count of events in fixed interval",
                "parameters": ["lambda_param"],
                "use_cases": "Modeling rare events or arrivals (e.g., customer purchases)"
            },
            {
                "name": "exponential",
                "display_name": "📉 Exponential",
                "description": "Models time between events, many small values",
                "parameters": ["lambda_param"],
                "use_cases": "Time between purchases, waiting times"
            },
            {
                "name": "binomial",
                "display_name": "🎲 Binomial",
                "description": "Number of successes in fixed number of trials",
                "parameters": ["trials_param", "prob_param"],
                "use_cases": "Success/failure scenarios, survey responses"
            }
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)