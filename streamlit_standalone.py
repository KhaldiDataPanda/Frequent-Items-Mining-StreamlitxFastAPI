import streamlit as st
import random
import numpy as np
import pandas as pd
import math
import time
from typing import Dict, List

# Page configuration
st.set_page_config(
    page_title="Metaheuristic Frequent Items Mining",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .parameter-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .algorithm-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4169e1;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== ALGORITHM IMPLEMENTATIONS ====================

def generate_data(data_size, min_list_length, max_list_length, mean_list_length, 
                  std_list_length, min_item_id, max_item_id, distribution_type,
                  lambda_param, prob_param, trials_param) -> Dict[int, List[str]]:
    """Generate synthetic transaction data based on parameters and selected distribution."""
    data_dict = {}
    
    def generate_list_length():
        """Generate list length based on the selected distribution."""
        if distribution_type == "gaussian":
            length = round(random.gauss(mean_list_length, std_list_length))
        elif distribution_type == "uniform":
            length = random.randint(min_list_length, max_list_length)
        elif distribution_type == "poisson":
            length = np.random.poisson(lambda_param)
        elif distribution_type == "exponential":
            length = round(np.random.exponential(1/lambda_param))
        elif distribution_type == "binomial":
            length = np.random.binomial(trials_param, prob_param)
        else:
            length = round(random.gauss(mean_list_length, std_list_length))
        
        length = max(min_list_length, min(max_list_length, length))
        return length
    
    while len(data_dict) < data_size:
        list_length = generate_list_length()
        items = [f"item{np.random.randint(low=min_item_id, high=max_item_id+1)}" 
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

def genetic_algorithm(dictionary, population_size, num_generations, num_parents, mutation_rate):
    """Run genetic algorithm to find most frequent item."""
    population = generate_initial_population(dictionary, population_size)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for gen in range(num_generations):
        fitness_scores = [fitness_function(candidate) for candidate in population]
        parents = select_parents(population, fitness_scores, num_parents)
        offspring = crossover(parents, population_size - num_parents)
        offspring = mutate(offspring, mutation_rate)
        population = parents + offspring
        
        progress_bar.progress((gen + 1) / num_generations)
        status_text.text(f"Generation {gen + 1}/{num_generations}")
    
    progress_bar.empty()
    status_text.empty()
    
    best_candidate = max(population, key=lambda candidate: sum(fitness_function(candidate).values()))
    item_counts = fitness_function(best_candidate)
    most_occurred_item = max(item_counts, key=item_counts.get)
    
    return most_occurred_item, item_counts

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
        indices.sort(key=lambda i: self.velocity[i], reverse=True)
        self.position = [data_items[i] for i in indices[:len(data_items)]]
        
        current_fitness = self._calculate_fitness()
        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            self.best_position = self.position.copy()

def particle_swarm_optimization(dictionary, num_particles, num_iterations, w, c1, c2):
    """Run PSO algorithm to find most frequent item."""
    data_items = list(dictionary.values())
    particles = [Particle(data_items) for _ in range(num_particles)]
    
    global_best_particle = max(particles, key=lambda p: p.best_fitness)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for iteration in range(num_iterations):
        for particle in particles:
            particle.update_velocity(global_best_particle.best_position, w, c1, c2)
            particle.update_position(data_items)
            
            if particle.best_fitness > global_best_particle.best_fitness:
                global_best_particle = particle
        
        progress_bar.progress((iteration + 1) / num_iterations)
        status_text.text(f"Iteration {iteration + 1}/{num_iterations}")
    
    progress_bar.empty()
    status_text.empty()
    
    item_counts = fitness_function([global_best_particle.best_position])
    most_occurred_item = max(item_counts, key=item_counts.get) if item_counts else "item1"
    
    return most_occurred_item, item_counts

def simulated_annealing(dictionary, initial_temp, cooling_rate, min_temp, max_iterations):
    """Run simulated annealing to find most frequent item."""
    current_solution = list(dictionary.values())
    random.shuffle(current_solution)
    current_fitness = sum(fitness_function(current_solution).values())
    
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    temperature = initial_temp
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for iteration in range(max_iterations):
        if temperature < min_temp:
            break
            
        new_solution = current_solution.copy()
        i, j = random.sample(range(len(new_solution)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        
        new_fitness = sum(fitness_function(new_solution).values())
        
        if new_fitness > current_fitness or random.random() < math.exp((new_fitness - current_fitness) / temperature):
            current_solution = new_solution
            current_fitness = new_fitness
            
            if current_fitness > best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
        
        temperature *= cooling_rate
        
        progress_bar.progress((iteration + 1) / max_iterations)
        status_text.text(f"Iteration {iteration + 1}/{max_iterations} | Temp: {temperature:.2f}")
    
    progress_bar.empty()
    status_text.empty()
    
    item_counts = fitness_function(best_solution)
    most_occurred_item = max(item_counts, key=item_counts.get) if item_counts else "item1"
    
    return most_occurred_item, item_counts

def ant_colony_optimization(dictionary, num_ants, num_iterations, alpha, beta, evaporation_rate, q):
    """Run ACO algorithm to find most frequent item."""
    data_items = list(dictionary.values())
    num_items = len(data_items)
    
    pheromones = [[1.0 for _ in range(num_items)] for _ in range(num_items)]
    
    best_solution = None
    best_fitness = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for iteration in range(num_iterations):
        solutions = []
        
        for ant in range(num_ants):
            solution = []
            available_items = list(range(num_items))
            
            while available_items:
                if len(solution) == 0:
                    next_item = random.choice(available_items)
                else:
                    probabilities = []
                    current_item = solution[-1]
                    
                    for item in available_items:
                        pheromone = pheromones[current_item][item] ** alpha
                        heuristic = (1.0 / (item + 1)) ** beta
                        probabilities.append(pheromone * heuristic)
                    
                    total_prob = sum(probabilities)
                    if total_prob > 0:
                        probabilities = [p / total_prob for p in probabilities]
                        next_item = np.random.choice(available_items, p=probabilities)
                    else:
                        next_item = random.choice(available_items)
                
                solution.append(next_item)
                available_items.remove(next_item)
            
            ant_solution = [data_items[i] for i in solution]
            solutions.append((solution, ant_solution))
        
        for solution_indices, solution_items in solutions:
            fitness = sum(fitness_function([solution_items]).values())
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution_items
            
            pheromone_deposit = q / (1 + fitness)
            for i in range(len(solution_indices) - 1):
                current_idx = solution_indices[i]
                next_idx = solution_indices[i + 1]
                pheromones[current_idx][next_idx] += pheromone_deposit
        
        for i in range(num_items):
            for j in range(num_items):
                pheromones[i][j] *= (1 - evaporation_rate)
                pheromones[i][j] = max(pheromones[i][j], 0.01)
        
        progress_bar.progress((iteration + 1) / num_iterations)
        status_text.text(f"Iteration {iteration + 1}/{num_iterations}")
    
    progress_bar.empty()
    status_text.empty()
    
    if best_solution is None:
        best_solution = data_items
    
    item_counts = fitness_function([best_solution])
    most_occurred_item = max(item_counts, key=item_counts.get) if item_counts else "item1"
    
    return most_occurred_item, item_counts

# ==================== STREAMLIT UI ====================

# Create tabs for different pages
tab1, tab2 = st.tabs(["🚀 Algorithm Runner", "📚 Theory Behind"])

with tab1:
    # Title and description
    st.markdown('<h1 class="main-header">🔍 Metaheuristic Frequent Items Mining</h1>', unsafe_allow_html=True)
    st.markdown("""
    This application uses various **metaheuristic algorithms** to find the most frequently occurring items in synthetically generated transaction data.
    Choose your algorithm and adjust the parameters below to customize your analysis.
    """)

    # Sidebar for parameters
    st.sidebar.title("⚙️ Configuration")

    # Algorithm Selection
    st.sidebar.markdown("### 🎯 Algorithm Selection")
    algorithm_names = {
        "GA": "🧬 Genetic Algorithm",
        "PSO": "🐦 Particle Swarm Optimization", 
        "SA": "🔥 Simulated Annealing",
        "ACO": "🐜 Ant Colony Optimization"
    }
    
    selected_algorithm = st.sidebar.selectbox(
        "Choose Algorithm",
        options=list(algorithm_names.keys()),
        format_func=lambda x: algorithm_names[x],
        help="Select the metaheuristic algorithm to use"
    )

    # Data Generation Parameters
    st.sidebar.markdown("### 📊 Data Generation")
    with st.sidebar.expander("Data Parameters", expanded=True):
        data_size = st.slider("Dataset Size", min_value=100, max_value=10000, value=2000, step=100,
                             help="Number of transactions to generate")
        
        distribution_options = {
            "gaussian": "📈 Gaussian (Normal)",
            "uniform": "📊 Uniform",
            "poisson": "🎯 Poisson", 
            "exponential": "📉 Exponential",
            "binomial": "🎲 Binomial"
        }
        
        selected_distribution = st.selectbox(
            "Distribution Type",
            options=list(distribution_options.keys()),
            format_func=lambda x: distribution_options[x],
            index=0,
            help="Statistical distribution for generating transaction lengths"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            min_list_length = st.number_input("Min List Length", min_value=1, max_value=10, value=1,
                                            help="Minimum items per transaction")
        with col2:
            max_list_length = st.number_input("Max List Length", min_value=1, max_value=20, value=6,
                                            help="Maximum items per transaction")
        
        # Distribution-specific parameters
        if selected_distribution == "gaussian":
            col1, col2 = st.columns(2)
            with col1:
                mean_list_length = st.number_input("Mean Length", min_value=1.0, max_value=20.0, value=3.0, step=0.1)
            with col2:
                std_list_length = st.number_input("Std Dev", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            lambda_param = 3.0
            prob_param = 0.5
            trials_param = 10
            
        elif selected_distribution == "uniform":
            st.info("📊 Uniform distribution uses only Min/Max List Length parameters above")
            mean_list_length = 3.0
            std_list_length = 1.0
            lambda_param = 3.0
            prob_param = 0.5
            trials_param = 10
            
        elif selected_distribution == "poisson":
            lambda_param = st.number_input("Lambda (λ)", min_value=0.1, max_value=10.0, value=3.0, step=0.1,
                                         help="Average rate parameter for Poisson distribution")
            st.info("🎯 Poisson distribution models count events (e.g., items bought)")
            mean_list_length = 3.0
            std_list_length = 1.0
            prob_param = 0.5
            trials_param = 10
            
        elif selected_distribution == "exponential":
            lambda_param = st.number_input("Rate (λ)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                         help="Rate parameter (1/mean) for exponential distribution")
            st.info("📉 Exponential distribution models time between events")
            mean_list_length = 3.0
            std_list_length = 1.0
            prob_param = 0.5
            trials_param = 10
            
        elif selected_distribution == "binomial":
            col1, col2 = st.columns(2)
            with col1:
                trials_param = st.number_input("Trials (n)", min_value=1, max_value=50, value=10)
            with col2:
                prob_param = st.number_input("Probability (p)", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
            st.info("🎲 Binomial distribution models number of successes in fixed trials")
            mean_list_length = 3.0
            std_list_length = 1.0
            lambda_param = 3.0

    # Item Range Parameters
    st.sidebar.markdown("### 🏷️ Item Range")
    with st.sidebar.expander("Item Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            min_item_id = st.number_input("Min Item ID", min_value=1, max_value=100, value=1,
                                        help="Minimum item identifier")
        with col2:
            max_item_id = st.number_input("Max Item ID", min_value=1, max_value=100, value=6,
                                        help="Maximum item identifier")

    # Dynamic Algorithm Parameters
    st.sidebar.markdown(f"### {algorithm_names[selected_algorithm]} Parameters")
    
    algorithm_params = {}
    
    if selected_algorithm == "GA":
        with st.sidebar.expander("Genetic Algorithm Parameters", expanded=True):
            population_size = st.slider("Population Size", min_value=10, max_value=500, value=100, step=10,
                                       help="Number of individuals in each generation")
            num_generations = st.slider("Generations", min_value=10, max_value=500, value=100, step=10,
                                       help="Number of generations to evolve")
            num_parents = st.slider("Number of Parents", min_value=5, max_value=population_size//2, 
                                   value=min(50, population_size//2), step=5,
                                   help="Number of parents selected for reproduction")
            mutation_rate = st.slider("Mutation Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01,
                                     help="Probability of mutation occurring")
            algorithm_params = {
                "population_size": population_size,
                "num_generations": num_generations, 
                "num_parents": num_parents,
                "mutation_rate": mutation_rate
            }
            
    elif selected_algorithm == "PSO":
        with st.sidebar.expander("PSO Parameters", expanded=True):
            num_particles = st.slider("Number of Particles", min_value=10, max_value=200, value=50, step=5,
                                     help="Number of particles in the swarm")
            num_iterations = st.slider("Iterations", min_value=50, max_value=1000, value=200, step=10,
                                     help="Number of iterations to run")
            w = st.slider("Inertia Weight (w)", min_value=0.1, max_value=2.0, value=0.9, step=0.1,
                         help="Controls the influence of previous velocity")
            c1 = st.slider("Cognitive Parameter (c1)", min_value=0.1, max_value=4.0, value=2.0, step=0.1,
                          help="Personal best influence")
            c2 = st.slider("Social Parameter (c2)", min_value=0.1, max_value=4.0, value=2.0, step=0.1,
                          help="Global best influence")
            algorithm_params = {
                "num_particles": num_particles,
                "num_iterations": num_iterations,
                "w": w,
                "c1": c1,
                "c2": c2
            }
            
    elif selected_algorithm == "SA":
        with st.sidebar.expander("Simulated Annealing Parameters", expanded=True):
            initial_temp = st.slider("Initial Temperature", min_value=100.0, max_value=5000.0, value=1000.0, step=50.0,
                                   help="Starting temperature for the annealing process")
            cooling_rate = st.slider("Cooling Rate", min_value=0.80, max_value=0.99, value=0.95, step=0.01,
                                   help="Rate at which temperature decreases")
            min_temp = st.slider("Minimum Temperature", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                               help="Temperature at which to stop the algorithm")
            max_iterations = st.slider("Max Iterations", min_value=100, max_value=5000, value=1000, step=50,
                                     help="Maximum number of iterations")
            algorithm_params = {
                "initial_temp": initial_temp,
                "cooling_rate": cooling_rate,
                "min_temp": min_temp,
                "max_iterations": max_iterations
            }
            
    elif selected_algorithm == "ACO":
        with st.sidebar.expander("Ant Colony Optimization Parameters", expanded=True):
            num_ants = st.slider("Number of Ants", min_value=10, max_value=200, value=50, step=5,
                               help="Number of ants in the colony")
            num_iterations = st.slider("Iterations", min_value=50, max_value=500, value=100, step=10,
                                     help="Number of iterations to run")
            alpha = st.slider("Alpha (α)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                            help="Pheromone importance factor")
            beta = st.slider("Beta (β)", min_value=0.1, max_value=5.0, value=2.0, step=0.1,
                           help="Heuristic importance factor")
            evaporation_rate = st.slider("Evaporation Rate", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                                       help="Rate at which pheromones evaporate")
            q = st.slider("Pheromone Deposit (Q)", min_value=10.0, max_value=500.0, value=100.0, step=10.0,
                        help="Amount of pheromone deposited")
            algorithm_params = {
                "num_ants": num_ants,
                "num_iterations": num_iterations,
                "alpha": alpha,
                "beta": beta,
                "evaporation_rate": evaporation_rate,
                "q": q
            }

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🚀 Run Analysis")
        
        st.markdown(f'<div class="algorithm-box">', unsafe_allow_html=True)
        st.markdown(f"**Selected Algorithm:** {algorithm_names[selected_algorithm]}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔍 Find Most Frequent Item", type="primary", use_container_width=True):
            try:
                with st.spinner("Generating data..."):
                    data_dict = generate_data(
                        data_size, min_list_length, max_list_length, mean_list_length,
                        std_list_length, min_item_id, max_item_id, selected_distribution,
                        lambda_param, prob_param, trials_param
                    )
                
                st.success(f"✅ Generated {len(data_dict)} transactions")
                
                st.info(f"🔄 Running {algorithm_names[selected_algorithm]}...")
                
                start_time = time.time()
                
                # Run selected algorithm
                if selected_algorithm == "GA":
                    most_occurred_item, item_counts = genetic_algorithm(
                        data_dict, 
                        algorithm_params["population_size"],
                        algorithm_params["num_generations"],
                        algorithm_params["num_parents"],
                        algorithm_params["mutation_rate"]
                    )
                    
                elif selected_algorithm == "PSO":
                    most_occurred_item, item_counts = particle_swarm_optimization(
                        data_dict,
                        algorithm_params["num_particles"],
                        algorithm_params["num_iterations"],
                        algorithm_params["w"],
                        algorithm_params["c1"],
                        algorithm_params["c2"]
                    )
                    
                elif selected_algorithm == "SA":
                    most_occurred_item, item_counts = simulated_annealing(
                        data_dict,
                        algorithm_params["initial_temp"],
                        algorithm_params["cooling_rate"],
                        algorithm_params["min_temp"],
                        algorithm_params["max_iterations"]
                    )
                    
                elif selected_algorithm == "ACO":
                    most_occurred_item, item_counts = ant_colony_optimization(
                        data_dict,
                        algorithm_params["num_ants"],
                        algorithm_params["num_iterations"],
                        algorithm_params["alpha"],
                        algorithm_params["beta"],
                        algorithm_params["evaporation_rate"],
                        algorithm_params["q"]
                    )
                
                execution_time = time.time() - start_time
                
                # Display results
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("### 🎯 Results")
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.metric("Most Frequent Item", most_occurred_item)
                    st.metric("Occurrence Count", item_counts.get(most_occurred_item, 0))
                with col_res2:
                    st.metric("Algorithm", selected_algorithm)
                    st.metric("Execution Time", f"{execution_time:.3f}s")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Item counts visualization
                st.markdown("### 📊 Item Frequency Distribution")
                
                # Sort item counts
                sorted_items = dict(sorted(item_counts.items(), key=lambda x: x[1], reverse=True))
                
                # Create DataFrame
                df = pd.DataFrame(list(sorted_items.items()), columns=["Item", "Count"])
                
                # Bar chart
                st.bar_chart(df.set_index("Item"))
                
                # Data table
                with st.expander("📋 View Detailed Counts"):
                    st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Sample transactions
                with st.expander("🔍 View Sample Transactions"):
                    sample_size = min(10, len(data_dict))
                    sample_data = {k: data_dict[k] for k in list(data_dict.keys())[:sample_size]}
                    for idx, items in sample_data.items():
                        st.write(f"**Transaction {idx}:** {', '.join(items)}")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

    with col2:
        st.markdown("### ℹ️ Algorithm Info")
        
        algorithm_descriptions = {
            "GA": """
            **🧬 Genetic Algorithm:**
            - Uses evolutionary principles
            - Selection, Crossover, Mutation
            - Population-based search
            - Good for complex optimization
            """,
            "PSO": """
            **🐦 Particle Swarm Optimization:**
            - Inspired by bird flocking
            - Particles move through solution space
            - Social and cognitive learning
            - Fast convergence
            """,
            "SA": """
            **🔥 Simulated Annealing:**
            - Inspired by metallurgy process
            - Probabilistic acceptance
            - Temperature cooling schedule
            - Escapes local optima
            """,
            "ACO": """
            **🐜 Ant Colony Optimization:**
            - Inspired by ant foraging
            - Pheromone trail communication
            - Indirect coordination
            - Good for path problems
            """
        }
        
        with st.expander("📖 Current Algorithm", expanded=True):
            st.markdown(algorithm_descriptions[selected_algorithm])
        
        with st.expander("⚙️ Parameter Guide"):
            distribution_descriptions = {
                "gaussian": "**📈 Gaussian**: Bell curve - most values near mean, few at extremes",
                "uniform": "**📊 Uniform**: All values equally likely within range",
                "poisson": "**🎯 Poisson**: Models rare events, many small values",
                "exponential": "**📉 Exponential**: Models waiting times, heavily skewed",
                "binomial": "**🎲 Binomial**: Models success counts in fixed trials"
            }
            
            st.markdown(f"""
            **Data Generation:**
            - **Dataset Size**: Number of transactions to generate
            - **Distribution**: {distribution_descriptions.get(selected_distribution, "Selected distribution")}
            - **Item Range**: Available item identifiers
            
            **Algorithm-Specific:**
            - Parameters change based on selected algorithm
            - Each algorithm has different optimization strategies
            """)

with tab2:
    # Theory Behind page
    st.markdown('<h1 class="main-header">📚 Theory Behind Metaheuristic Algorithms</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This page provides detailed explanations of the metaheuristic algorithms used in this application
    for solving the frequent items mining problem.
    """)
    
    # Create sub-tabs for each algorithm and data distributions
    theory_tabs = st.tabs(["🧬 Genetic Algorithm", "🐦 Particle Swarm", "🔥 Simulated Annealing", "🐜 Ant Colony", "📊 Data Distributions"])
    
    with theory_tabs[0]:
        st.markdown("## 🧬 Genetic Algorithm (GA)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Overview
            Genetic Algorithm is an evolutionary optimization technique inspired by the process of natural selection. 
            It mimics the biological evolution process to find optimal or near-optimal solutions to complex problems.
            
            ### Key Components
            
            #### 1. **Population**
            - A collection of candidate solutions (individuals/chromosomes)
            - Each individual represents a possible arrangement of transaction data
            - Population size determines the diversity of solutions explored
            
            #### 2. **Fitness Function**
            - Evaluates the quality of each individual
            - In our case: counts item occurrences across all transactions
            - Higher fitness = better solution (more frequent items found)
            
            #### 3. **Selection**
            - **Tournament Selection**: Randomly select small groups, choose the best from each
            - Ensures that fitter individuals have higher chances of reproduction
            - Maintains diversity while favoring good solutions
            
            #### 4. **Crossover (Reproduction)**
            - Combines genetic material from two parents
            - **Single-point crossover**: Split parents at random point, combine parts
            - Creates offspring that inherit traits from both parents
            
            #### 5. **Mutation**
            - Introduces random changes to maintain diversity
            - Prevents premature convergence to local optima
            - Small probability of random alterations in offspring
            
            ### Algorithm Flow
            ```
            1. Initialize random population
            2. Evaluate fitness of each individual
            3. While not converged:
                a. Select parents using tournament selection
                b. Create offspring through crossover
                c. Apply mutation to offspring
                d. Replace old population with new generation
                e. Evaluate fitness of new population
            4. Return best individual found
            ```
            
            ### Parameters Explained
            - **Population Size**: Larger = more diversity, slower convergence
            - **Generations**: More generations = better solutions, longer runtime
            - **Parents**: Number of individuals selected for reproduction
            - **Mutation Rate**: Higher = more exploration, risk of losing good solutions
            """)
        
        with col2:
            st.markdown("""
            ### Advantages ✅
            - Good global search capability
            - Handles complex, non-linear problems
            - Inherently parallel
            - Robust and flexible
            
            ### Disadvantages ❌
            - Can be slow to converge
            - No guarantee of global optimum
            - Many parameters to tune
            - May converge prematurely
            
            ### Best Used For
            - Complex optimization problems
            - Non-differentiable functions
            - Multi-modal landscapes
            - When solution quality matters more than speed
            """)
    
    with theory_tabs[1]:
        st.markdown("## 🐦 Particle Swarm Optimization (PSO)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Overview
            PSO is inspired by the social behavior of bird flocking or fish schooling. Each particle represents 
            a potential solution that moves through the solution space, influenced by its own experience and 
            the experience of neighboring particles.
            
            ### Key Concepts
            
            #### 1. **Particle**
            - Represents a candidate solution with position and velocity
            - Position: current solution arrangement
            - Velocity: direction and magnitude of movement
            
            #### 2. **Personal Best (pbest)**
            - Best position found by each individual particle
            - Represents particle's own experience and memory
            
            #### 3. **Global Best (gbest)**
            - Best position found by any particle in the swarm
            - Represents collective knowledge of the swarm
            
            #### 4. **Velocity Update**
            The velocity of each particle is updated using:
            ```
            v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))
            ```
            Where:
            - w: inertia weight (controls previous velocity influence)
            - c1: cognitive parameter (personal best influence)
            - c2: social parameter (global best influence)
            - r1, r2: random numbers [0,1]
            
            ### Algorithm Flow
            ```
            1. Initialize particles with random positions and velocities
            2. Evaluate fitness of each particle
            3. Set personal and global bests
            4. While not converged:
                a. Update velocities based on personal and global bests
                b. Update positions based on velocities
                c. Evaluate new positions
                d. Update personal and global bests
            5. Return global best solution
            ```
            
            ### Parameters Explained
            - **Particles**: More particles = better exploration, higher computation
            - **Iterations**: More iterations = better convergence, longer runtime
            - **Inertia (w)**: Higher = more exploration, lower = more exploitation
            - **Cognitive (c1)**: Influence of particle's own experience
            - **Social (c2)**: Influence of swarm's collective knowledge
            """)
        
        with col2:
            st.markdown("""
            ### Advantages ✅
            - Fast convergence
            - Few parameters to adjust
            - Easy to implement
            - Good for continuous optimization
            
            ### Disadvantages ❌
            - May converge prematurely
            - Can get trapped in local optima
            - Less effective for discrete problems
            - Sensitive to parameter settings
            
            ### Best Used For
            - Continuous optimization
            - Fast approximate solutions
            - Real-time applications
            - When simplicity is preferred
            """)
    
    with theory_tabs[2]:
        st.markdown("## 🔥 Simulated Annealing (SA)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Overview
            Simulated Annealing is inspired by the annealing process in metallurgy, where controlled cooling 
            of materials leads to a more stable, lower-energy state. The algorithm uses a temperature parameter 
            that decreases over time, controlling the probability of accepting worse solutions.
            
            ### Key Concepts
            
            #### 1. **Temperature**
            - Controls the probability of accepting worse solutions
            - High temperature: accept many worse solutions (exploration)
            - Low temperature: accept few worse solutions (exploitation)
            
            #### 2. **Cooling Schedule**
            - Determines how temperature decreases over time
            - Common approach: T(t+1) = α * T(t) where α < 1
            - Slower cooling = better solutions, longer runtime
            
            #### 3. **Acceptance Probability**
            For a worse solution with fitness difference Δf:
            ```
            P(accept) = exp(-Δf / T)
            ```
            - Higher temperature = higher acceptance probability
            - Smaller fitness difference = higher acceptance probability
            
            #### 4. **Neighborhood Function**
            - Generates new solutions from current solution
            - In our case: swap two random positions in transaction arrangement
            - Quality of neighborhood affects algorithm performance
            
            ### Algorithm Flow
            ```
            1. Initialize with random solution and high temperature
            2. While temperature > minimum:
                a. Generate neighbor solution
                b. Calculate fitness difference
                c. If better: accept new solution
                d. If worse: accept with probability exp(-Δf/T)
                e. Decrease temperature according to cooling schedule
            3. Return best solution found
            ```
            
            ### Parameters Explained
            - **Initial Temperature**: Higher = more exploration initially
            - **Cooling Rate**: Slower cooling = better solutions, longer time
            - **Minimum Temperature**: Stopping criterion for the algorithm
            - **Max Iterations**: Alternative stopping criterion
            """)
        
        with col2:
            st.markdown("""
            ### Advantages ✅
            - Can escape local optima
            - Theoretical convergence guarantee
            - Simple to implement
            - Works well for many problems
            
            ### Disadvantages ❌
            - Slow convergence
            - Sensitive to cooling schedule
            - No memory of good solutions
            - Single-solution based (no population)
            
            ### Best Used For
            - Avoiding local optima
            - Problems with many local minima
            - When solution quality is critical
            - Combinatorial optimization
            """)
    
    with theory_tabs[3]:
        st.markdown("## 🐜 Ant Colony Optimization (ACO)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Overview
            ACO is inspired by the foraging behavior of ants. Real ants deposit pheromones on paths they traverse,
            and other ants are more likely to follow paths with stronger pheromone trails. This indirect 
            communication leads to the discovery of shortest paths.
            
            ### Key Concepts
            
            #### 1. **Pheromone Trails**
            - Chemical substance deposited by ants
            - Guides future ants toward better solutions
            - Stronger trails = more attractive paths
            
            #### 2. **Heuristic Information**
            - Problem-specific guidance
            - Helps ants make informed decisions
            - Combined with pheromones for decision making
            
            #### 3. **Probability Rule**
            Probability of ant k moving from position i to j:
            ```
            P(i,j) = [τ(i,j)^α * η(i,j)^β] / Σ[τ(i,k)^α * η(i,k)^β]
            ```
            Where:
            - τ: pheromone level
            - η: heuristic information
            - α: pheromone importance
            - β: heuristic importance
            
            #### 4. **Pheromone Update**
            - **Evaporation**: τ(i,j) = (1-ρ) * τ(i,j)
            - **Deposit**: τ(i,j) += Δτ(i,j)
            - Balances exploration and exploitation
            
            ### Algorithm Flow
            ```
            1. Initialize pheromone trails
            2. While not converged:
                a. Each ant constructs a solution
                b. Evaluate all solutions
                c. Update best solution found
                d. Evaporate pheromones
                e. Deposit new pheromones on good paths
            3. Return best solution found
            ```
            
            ### Parameters Explained
            - **Ants**: More ants = better exploration, higher computation
            - **Iterations**: More iterations = better convergence
            - **Alpha (α)**: Higher = pheromones more important
            - **Beta (β)**: Higher = heuristic more important
            - **Evaporation Rate**: Higher = forget bad solutions faster
            - **Q**: Amount of pheromone deposited
            """)
        
        with col2:
            st.markdown("""
            ### Advantages ✅
            - Positive feedback mechanism
            - Can adapt to changes
            - Inherently parallel
            - Good for graph-based problems
            
            ### Disadvantages ❌
            - Many parameters to tune
            - Can converge prematurely
            - Slower than some alternatives
            - Requires careful tuning
            
            ### Best Used For
            - Routing problems
            - Scheduling tasks
            - Graph traversal
            - Discrete optimization
            """)
    
    with theory_tabs[4]:
        st.markdown("## 📊 Data Distributions")
        
        st.markdown("""
        Understanding different probability distributions is crucial for generating realistic synthetic data. 
        Each distribution models different real-world scenarios and affects the characteristics of your dataset.
        """)
        
        # Create distribution comparison
        dist_tabs = st.tabs(["📈 Gaussian", "📊 Uniform", "🎯 Poisson", "📉 Exponential", "🎲 Binomial"])
        
        with dist_tabs[0]:
            st.markdown("""
            ### 📈 Gaussian (Normal) Distribution
            
            **Mathematical Form:**
            ```
            f(x) = (1 / (σ√(2π))) * exp(-(x-μ)²/(2σ²))
            ```
            Where μ is the mean and σ is the standard deviation.
            
            **Characteristics:**
            - Symmetric bell-shaped curve
            - Most values cluster around the mean
            - 68% of values within 1 std dev, 95% within 2 std dev
            - Described by two parameters: mean (μ) and standard deviation (σ)
            
            **Real-World Applications:**
            - Heights and weights of populations
            - Measurement errors
            - Test scores
            - Natural phenomena with random variation
            
            **For Transaction Length:**
            - Most transactions have similar length (near the mean)
            - Some variation is expected (controlled by std dev)
            - Very short or very long transactions are rare
            - Models typical shopping behavior
            
            **Parameter Guidelines:**
            - **Mean**: Set to expected average transaction length (e.g., 3-5 items)
            - **Std Dev**: Controls spread (1.0 = moderate variation, 2.0 = high variation)
            """)
        
        with dist_tabs[1]:
            st.markdown("""
            ### 📊 Uniform Distribution
            
            **Mathematical Form:**
            ```
            f(x) = 1/(b-a)  for a ≤ x ≤ b
            ```
            Where a is minimum and b is maximum.
            
            **Characteristics:**
            - Flat distribution - all values equally likely
            - Constant probability density
            - Bounded by minimum and maximum values
            - No clustering around any particular value
            
            **Real-World Applications:**
            - Random number generation
            - Lottery numbers
            - Dice rolls
            - Situations with no inherent bias
            
            **For Transaction Length:**
            - All transaction lengths between min and max are equally likely
            - No preference for short or long transactions
            - Useful for baseline comparisons
            - Rare in real shopping scenarios
            
            **Parameter Guidelines:**
            - **Min**: Smallest possible transaction length
            - **Max**: Largest possible transaction length
            - Use when you have no prior knowledge about distribution
            """)
        
        with dist_tabs[2]:
            st.markdown("""
            ### 🎯 Poisson Distribution
            
            **Mathematical Form:**
            ```
            P(X=k) = (λ^k * e^(-λ)) / k!
            ```
            Where λ is the average rate of events.
            
            **Characteristics:**
            - Discrete distribution for count data
            - Models number of events in fixed interval
            - Right-skewed for small λ, approaches normal for large λ
            - Mean = Variance = λ
            
            **Real-World Applications:**
            - Number of customers arriving per hour
            - Number of emails received per day
            - Number of defects in manufacturing
            - Rare events occurrences
            
            **For Transaction Length:**
            - Models customer purchase patterns
            - Good for e-commerce datasets
            - Captures "number of items bought" naturally
            - Realistic for many retail scenarios
            
            **Parameter Guidelines:**
            - **Lambda (λ)**: Average number of items per transaction
            - λ = 2-3: Few items per transaction
            - λ = 5-7: Medium-sized transactions
            - λ > 10: Large transactions
            """)
        
        with dist_tabs[3]:
            st.markdown("""
            ### 📉 Exponential Distribution
            
            **Mathematical Form:**
            ```
            f(x) = λ * e^(-λx)  for x ≥ 0
            ```
            Where λ is the rate parameter.
            
            **Characteristics:**
            - Continuous distribution for non-negative values
            - Heavily right-skewed
            - Memoryless property
            - Many small values, few large values
            
            **Real-World Applications:**
            - Time between customer arrivals
            - Product lifetimes
            - Waiting times
            - Decay processes
            
            **For Transaction Length:**
            - Models time-based behavior
            - Heavily skewed toward small transactions
            - Many quick purchases, few large ones
            - Good for impulse-buy scenarios
            
            **Parameter Guidelines:**
            - **Rate (λ)**: Inverse of mean (1/mean)
            - λ = 1.0: Mean transaction length = 1
            - λ = 0.5: Mean transaction length = 2
            - Smaller λ = longer average transactions
            """)
        
        with dist_tabs[4]:
            st.markdown("""
            ### 🎲 Binomial Distribution
            
            **Mathematical Form:**
            ```
            P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
            ```
            Where n is number of trials and p is success probability.
            
            **Characteristics:**
            - Discrete distribution
            - Models number of successes in fixed trials
            - Symmetric when p = 0.5
            - Skewed for p far from 0.5
            
            **Real-World Applications:**
            - Quality control (defective items)
            - Survey responses (yes/no questions)
            - Clinical trials (success/failure)
            - A/B testing results
            
            **For Transaction Length:**
            - Models decision-making process
            - Each item represents a success/purchase decision
            - Good for modeling conversion rates
            - Useful for basket analysis
            
            **Parameter Guidelines:**
            - **Trials (n)**: Maximum possible items (e.g., 10-20)
            - **Probability (p)**: Likelihood of adding an item
            - p = 0.5: Medium-sized transactions
            - p < 0.3: Small transactions
            - p > 0.7: Large transactions
            """)
        
        # Distribution comparison table
        st.markdown("---")
        st.markdown("### 📋 Distribution Comparison")
        
        dist_comparison_data = {
            "Distribution": ["Gaussian", "Uniform", "Poisson", "Exponential", "Binomial"],
            "Type": ["Continuous", "Continuous", "Discrete", "Continuous", "Discrete"],
            "Shape": ["Bell curve", "Flat", "Right-skewed", "Heavy right-skew", "Variable"],
            "Parameters": ["Mean, Std Dev", "Min, Max", "Lambda", "Rate", "Trials, Probability"],
            "Best Use Case": ["Natural variation", "Equal likelihood", "Count events", "Waiting times", "Success counts"],
            "Complexity": ["Low", "Very Low", "Low", "Low", "Medium"]
        }
        
        dist_comparison_df = pd.DataFrame(dist_comparison_data)
        st.dataframe(dist_comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        ### 🎯 Choosing the Right Distribution
        
        **For Transaction Length Modeling:**
        
        - **📈 Gaussian**: When most transactions have similar lengths with natural variation
        - **📊 Uniform**: When all transaction lengths are equally likely (rare in practice)
        - **🎯 Poisson**: When modeling customer purchase patterns or item counts
        - **📉 Exponential**: When modeling time-based behavior or heavily skewed data
        - **🎲 Binomial**: When modeling success rates or binary decision processes
        
        **Consider Your Domain:**
        
        - **E-commerce**: Poisson or Exponential (customer behavior)
        - **Retail**: Gaussian or Poisson (transaction patterns)
        - **Manufacturing**: Binomial or Poisson (quality control)
        - **Finance**: Gaussian or Exponential (risk modeling)
        - **Research**: Start with Gaussian, then experiment
        """)

    # Comparison section
    st.markdown("---")
    st.markdown("## 🔄 Algorithm Comparison")
    
    comparison_data = {
        "Algorithm": ["Genetic Algorithm", "Particle Swarm", "Simulated Annealing", "Ant Colony"],
        "Population-based": ["✅", "✅", "❌", "✅"],
        "Memory": ["✅", "✅", "❌", "✅"],
        "Convergence Speed": ["Medium", "Fast", "Slow", "Medium"],
        "Parameter Sensitivity": ["Medium", "Low", "High", "High"],
        "Global Search": ["Good", "Medium", "Good", "Medium"],
        "Implementation": ["Medium", "Easy", "Easy", "Complex"]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ### Choosing the Right Algorithm
    
    - **For beginners**: Start with **Particle Swarm Optimization** (fewer parameters, faster)
    - **For quality solutions**: Use **Genetic Algorithm** or **Simulated Annealing**
    - **For avoiding local optima**: **Simulated Annealing** works well
    - **For parallel processing**: **Genetic Algorithm** or **Ant Colony Optimization**
    - **For real-time applications**: **Particle Swarm Optimization**
    
    ### Problem-Specific Considerations
    
    In the context of **frequent items mining**:
    - **GA**: Good balance of exploration and exploitation
    - **PSO**: Fast convergence for quick approximate results
    - **SA**: Excellent for avoiding suboptimal item frequency counts
    - **ACO**: Interesting approach treating items as path components
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit 🎈 | Standalone Version (No FastAPI Required) 🚀 | Multiple Metaheuristics 🧬🐦🔥🐜</p>
    <p><em>Ready for Streamlit Cloud Deployment!</em></p>
</div>
""", unsafe_allow_html=True)
