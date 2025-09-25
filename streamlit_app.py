import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time

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

# Create tabs for different pages
tab1, tab2 = st.tabs(["🚀 Algorithm Runner", "📚 Theory Behind"])

# API URL configuration (moved to sidebar)
st.sidebar.title("🔧 API Configuration")
API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000", help="URL of the FastAPI backend")

with tab1:
    # Title and description for Algorithm Runner tab
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
        
        # Distribution Selection
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
            index=0,  # Default to Gaussian
            help="Statistical distribution for generating transaction lengths"
        )
        
        # Basic parameters (always shown)
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
                mean_list_length = st.number_input("Mean List Length", min_value=1.0, max_value=10.0, value=3.0, step=0.1,
                                                 help="Average items per transaction")
            with col2:
                std_list_length = st.number_input("Std List Length", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                                help="Standard deviation for list length")
            lambda_param = 3.0  # Default values for other distributions
            prob_param = 0.5
            trials_param = 10
            
        elif selected_distribution == "uniform":
            st.info("📊 Uniform distribution uses only Min/Max List Length parameters above")
            mean_list_length = 3.0  # Default values
            std_list_length = 1.0
            lambda_param = 3.0
            prob_param = 0.5
            trials_param = 10
            
        elif selected_distribution == "poisson":
            lambda_param = st.number_input("Lambda (λ)", min_value=0.1, max_value=10.0, value=3.0, step=0.1,
                                         help="Average rate parameter for Poisson distribution")
            st.info("🎯 Poisson distribution models count events (e.g., items bought)")
            mean_list_length = 3.0  # Default values
            std_list_length = 1.0
            prob_param = 0.5
            trials_param = 10
            
        elif selected_distribution == "exponential":
            lambda_param = st.number_input("Rate (λ)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                         help="Rate parameter (1/mean) for exponential distribution")
            st.info("📉 Exponential distribution models time between events")
            mean_list_length = 3.0  # Default values
            std_list_length = 1.0
            prob_param = 0.5
            trials_param = 10
            
        elif selected_distribution == "binomial":
            col1, col2 = st.columns(2)
            with col1:
                trials_param = st.number_input("Number of Trials (n)", min_value=1, max_value=20, value=10,
                                             help="Number of independent trials")
            with col2:
                prob_param = st.number_input("Success Probability (p)", min_value=0.01, max_value=0.99, value=0.5, step=0.01,
                                           help="Probability of success in each trial")
            st.info("🎲 Binomial distribution models number of successes in fixed trials")
            mean_list_length = 3.0  # Default values
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

    # Dynamic Algorithm Parameters based on selection
    st.sidebar.markdown(f"### {algorithm_names[selected_algorithm]} Parameters")
    
    # Clear any previous parameter containers to avoid overlap
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
        
        # Display selected algorithm info
        st.markdown(f'<div class="algorithm-box">', unsafe_allow_html=True)
        st.markdown(f"**Selected Algorithm:** {algorithm_names[selected_algorithm]}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔍 Find Most Frequent Item", type="primary", use_container_width=True):
            # Prepare request data based on selected algorithm
            request_data = {
                "algorithm": selected_algorithm,
                "data_generation": {
                    "data_size": data_size,
                    "min_list_length": min_list_length,
                    "max_list_length": max_list_length,
                    "mean_list_length": mean_list_length,
                    "std_list_length": std_list_length,
                    "min_item_id": min_item_id,
                    "max_item_id": max_item_id,
                    "distribution_type": selected_distribution,
                    "lambda_param": lambda_param,
                    "prob_param": prob_param,
                    "trials_param": trials_param
                }
            }
            
            # Add algorithm-specific parameters
            if selected_algorithm == "GA":
                request_data["ga_parameters"] = algorithm_params
            elif selected_algorithm == "PSO":
                request_data["pso_parameters"] = algorithm_params
            elif selected_algorithm == "SA":
                request_data["sa_parameters"] = algorithm_params
            elif selected_algorithm == "ACO":
                request_data["aco_parameters"] = algorithm_params
        
            try:
                with st.spinner(f"🔍 Running {algorithm_names[selected_algorithm]}... This may take a few moments."):
                    # Make API request
                    response = requests.post(f"{API_URL}/mine-frequent-items", 
                                           json=request_data,
                                           timeout=300)  # 5 minute timeout
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.success("✅ Analysis completed successfully!")
                        
                        # Most frequent item with algorithm info
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown(f"### 🏆 Most Frequent Item: **{result['most_occurred_item']}**")
                        st.markdown(f"**Algorithm Used:** {algorithm_names[result['algorithm_used']]}")
                        st.markdown(f"**Occurrences:** {result['item_counts'][result['most_occurred_item']]}")
                        st.markdown(f"**Execution Time:** {result['execution_time']:.3f} seconds")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Item counts visualization
                    if result['item_counts']:
                        st.markdown("### 📊 Item Frequency Distribution")
                        
                        # Create DataFrame for visualization
                        df = pd.DataFrame(list(result['item_counts'].items()), 
                                        columns=['Item', 'Count'])
                        df = df.sort_values('Count', ascending=False)
                        
                        # Bar chart
                        fig = px.bar(df, x='Item', y='Count', 
                                   title="Item Frequency Distribution",
                                   color='Count',
                                   color_continuous_scale='viridis')
                        fig.update_layout(
                            xaxis_title="Items",
                            yaxis_title="Frequency Count",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Pie chart
                        fig_pie = px.pie(df, values='Count', names='Item', 
                                       title="Item Distribution (Pie Chart)")
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Data table
                        st.markdown("### 📋 Detailed Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Items", len(df))
                        with col2:
                            st.metric("Max Frequency", df['Count'].max())
                        with col3:
                            st.metric("Min Frequency", df['Count'].min())
                        with col4:
                            st.metric("Avg Frequency", f"{df['Count'].mean():.2f}")
                    
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        if response.text:
                            st.error(response.text)
                            
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to API. Make sure the FastAPI server is running on the specified URL.")
                st.code(f"python main.py", language="bash")
                
            except requests.exceptions.Timeout:
                st.error("⏰ Request timed out. Try reducing the dataset size or number of generations.")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

    with col2:
        st.markdown("### ℹ️ Algorithm Info")
        
        # Dynamic algorithm description based on selection
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
        
        # API Status
        st.markdown("### 🔧 API Status")
        try:
            health_response = requests.get(f"{API_URL}/health", timeout=5)
            if health_response.status_code == 200:
                st.success("🟢 API is running")
            else:
                st.error("🔴 API error")
        except:
            st.error("🔴 API not accessible")

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
            ACO is inspired by the foraging behavior of ants. Real ants deposit pheromones on paths, 
            creating a communication system that helps the colony find short paths between food sources 
            and their nest. The algorithm uses artificial pheromones to guide the search.
            
            ### Key Concepts
            
            #### 1. **Pheromone Trails**
            - Represent the attractiveness of solution components
            - Stronger trails = more attractive paths
            - Updated based on solution quality
            
            #### 2. **Heuristic Information**
            - Problem-specific guidance (like visibility for ants)
            - In our case: simple heuristic based on item position
            - Combines with pheromones to guide decisions
            
            #### 3. **Probabilistic Decision Rule**
            Probability of choosing component j from component i:
            ```
            P(i,j) = [τ(i,j)]^α * [η(i,j)]^β / Σ[τ(i,k)]^α * [η(i,k)]^β
            ```
            Where:
            - τ(i,j): pheromone level between i and j
            - η(i,j): heuristic information
            - α: pheromone importance factor
            - β: heuristic importance factor
            
            #### 4. **Pheromone Update**
            - **Evaporation**: τ(i,j) = (1-ρ) * τ(i,j)
            - **Deposition**: Add pheromone based on solution quality
            - Balances exploration and exploitation
            
            ### Algorithm Flow
            ```
            1. Initialize pheromone trails
            2. While not converged:
                a. Each ant constructs a solution probabilistically
                b. Evaluate solutions and update best found
                c. Update pheromone trails:
                   - Evaporate existing pheromones
                   - Deposit new pheromones based on solution quality
            3. Return best solution found
            ```
            
            ### Parameters Explained
            - **Ants**: More ants = more parallel exploration
            - **Iterations**: More iterations = better pheromone learning
            - **Alpha (α)**: Pheromone importance (higher = more exploitation)
            - **Beta (β)**: Heuristic importance (higher = more greedy)
            - **Evaporation Rate**: Higher = faster forgetting of bad paths
            - **Q**: Amount of pheromone deposited by each ant
            """)
        
        with col2:
            st.markdown("""
            ### Advantages ✅
            - Good for combinatorial problems
            - Inherently parallel
            - Positive feedback mechanism
            - Adapts to changes dynamically
            
            ### Disadvantages ❌
            - Many parameters to tune
            - Convergence can be slow
            - May converge prematurely
            - Theoretical analysis difficult
            
            ### Best Used For
            - Path optimization problems
            - Combinatorial optimization
            - Dynamic environments
            - When parallel processing available
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
            st.markdown("### 📈 Gaussian (Normal) Distribution")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                **Overview:**
                The Gaussian (Normal) distribution is the most common distribution in nature. It creates a bell-shaped curve 
                where most values cluster around the mean, with fewer values at the extremes.
                
                **Mathematical Formula:**
                ```
                f(x) = (1/σ√(2π)) * e^(-½((x-μ)/σ)²)
                ```
                Where μ is the mean and σ is the standard deviation.
                
                **Parameters:**
                - **Mean (μ)**: Center of the distribution, where most values cluster
                - **Standard Deviation (σ)**: Spread of the distribution
                
                **Characteristics:**
                - Symmetric around the mean
                - 68% of values within 1 standard deviation
                - 95% of values within 2 standard deviations
                - Can generate negative values (bounded by min/max in our case)
                
                **Real-world Examples:**
                - Heights and weights of people
                - Test scores in large populations
                - Measurement errors
                - Natural phenomena with random variation
                """)
            
            with col2:
                st.markdown("""
                **✅ Advantages:**
                - Natural and intuitive
                - Well-understood properties
                - Good for modeling natural variation
                - Central Limit Theorem applies
                
                **❌ Disadvantages:**
                - Can generate values outside bounds
                - May not fit all real-world scenarios
                - Symmetric (no skewness)
                
                **Best for:**
                - General-purpose data generation
                - When you have natural variation
                - Educational examples
                - Baseline comparisons
                """)
        
        with dist_tabs[1]:
            st.markdown("### 📊 Uniform Distribution")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                **Overview:**
                The uniform distribution gives equal probability to all values within a specified range. 
                Every outcome between the minimum and maximum is equally likely.
                
                **Mathematical Formula:**
                ```
                f(x) = 1/(b-a) for a ≤ x ≤ b
                f(x) = 0 otherwise
                ```
                Where a is the minimum and b is the maximum.
                
                **Parameters:**
                - **Minimum (a)**: Lower bound of the distribution
                - **Maximum (b)**: Upper bound of the distribution
                
                **Characteristics:**
                - Flat probability density
                - Equal likelihood for all values in range
                - No preferred central value
                - Sharp cutoffs at boundaries
                
                **Real-world Examples:**
                - Random number generation
                - Fair dice rolls
                - Random sampling
                - When no preference exists for any particular value
                """)
            
            with col2:
                st.markdown("""
                **✅ Advantages:**
                - Simple and predictable
                - Natural bounds
                - No bias toward any value
                - Easy to understand
                
                **❌ Disadvantages:**
                - May not reflect natural processes
                - No central tendency
                - Sharp boundaries unrealistic
                
                **Best for:**
                - Random sampling scenarios
                - When all options are equally likely
                - Testing algorithm robustness
                - Baseline random data
                """)
        
        with dist_tabs[2]:
            st.markdown("### 🎯 Poisson Distribution")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                **Overview:**
                The Poisson distribution models the number of events occurring in a fixed interval of time or space. 
                It's excellent for modeling rare events or counting processes.
                
                **Mathematical Formula:**
                ```
                P(X = k) = (λᵏ * e^(-λ)) / k!
                ```
                Where λ is the average rate of occurrence and k is the number of events.
                
                **Parameters:**
                - **Lambda (λ)**: Average rate of occurrence (both mean and variance)
                
                **Characteristics:**
                - Discrete distribution (whole numbers only)
                - Right-skewed for small λ, approaches normal for large λ
                - Mean = Variance = λ
                - Many small values, few large values
                
                **Real-world Examples:**
                - Number of customers arriving per hour
                - Defects in manufacturing
                - Phone calls received per minute
                - Items purchased in a single transaction
                """)
            
            with col2:
                st.markdown("""
                **✅ Advantages:**
                - Great for count data
                - Models rare events well
                - Single parameter simplicity
                - Realistic for many business scenarios
                
                **❌ Disadvantages:**
                - Only generates integers
                - Right-skewed (may not fit all data)
                - Mean equals variance constraint
                
                **Best for:**
                - Customer behavior modeling
                - Arrival processes
                - Count-based phenomena
                - Sparse event modeling
                """)
        
        with dist_tabs[3]:
            st.markdown("### 📉 Exponential Distribution")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                **Overview:**
                The exponential distribution models the time between events in a Poisson process. 
                It has many small values and few large values, creating a heavily right-skewed distribution.
                
                **Mathematical Formula:**
                ```
                f(x) = λ * e^(-λx) for x ≥ 0
                ```
                Where λ is the rate parameter.
                
                **Parameters:**
                - **Rate (λ)**: Rate parameter, where mean = 1/λ
                
                **Characteristics:**
                - Continuous distribution
                - Heavily right-skewed
                - Memoryless property
                - Decreasing probability density
                
                **Real-world Examples:**
                - Time between customer arrivals
                - Lifespan of electronic components
                - Time until next purchase
                - Waiting times in queues
                """)
            
            with col2:
                st.markdown("""
                **✅ Advantages:**
                - Models waiting times naturally
                - Memoryless property useful
                - Simple single parameter
                - Good for survival analysis
                
                **❌ Disadvantages:**
                - Heavily skewed
                - May generate very large values
                - Not suitable for symmetric data
                
                **Best for:**
                - Time-based modeling
                - Reliability analysis
                - Queue theory applications
                - Inter-arrival time modeling
                """)
        
        with dist_tabs[4]:
            st.markdown("### 🎲 Binomial Distribution")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                **Overview:**
                The binomial distribution models the number of successes in a fixed number of independent trials, 
                each with the same probability of success.
                
                **Mathematical Formula:**
                ```
                P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
                ```
                Where n is trials, p is success probability, and C(n,k) is combinations.
                
                **Parameters:**
                - **Trials (n)**: Number of independent trials
                - **Probability (p)**: Probability of success in each trial
                
                **Characteristics:**
                - Discrete distribution
                - Bell-shaped for large n and moderate p
                - Mean = n*p, Variance = n*p*(1-p)
                - Bounded between 0 and n
                
                **Real-world Examples:**
                - Number of defective items in a batch
                - Survey responses (yes/no questions)
                - Marketing campaign success rates
                - Quality control testing
                """)
            
            with col2:
                st.markdown("""
                **✅ Advantages:**
                - Models success/failure scenarios
                - Natural upper bound
                - Well-understood properties
                - Flexible with two parameters
                
                **❌ Disadvantages:**
                - Requires fixed number of trials
                - Assumes independence
                - May be complex to parameterize
                
                **Best for:**
                - Success rate modeling
                - Quality control scenarios
                - Survey data simulation
                - A/B testing scenarios
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
    <p>Built with Streamlit 🎈 | Powered by FastAPI ⚡ | Multiple Metaheuristics 🧬🐦🔥🐜</p>
</div>
""", unsafe_allow_html=True)