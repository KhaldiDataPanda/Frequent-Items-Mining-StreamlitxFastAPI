# Metaheuristic Frequent Items Mining

A comprehensive web application that uses various **metaheuristic algorithms** to find the most frequently occurring items in synthetically generated transaction data. The application features multiple optimization algorithms with dynamic parameter interfaces and detailed theoretical explanations.

## 🌟 Features

- **Multiple Metaheuristic Algorithms**: Choose from GA, PSO, SA, and ACO
- **Dynamic Parameter Interface**: UI adapts based on selected algorithm
- **Interactive Web Interface**: User-friendly Streamlit app with sliders and controls
- **FastAPI Backend**: RESTful API supporting multiple optimization algorithms
- **Real-time Visualization**: Interactive charts showing item frequency distributions
- **Theory Behind Page**: Comprehensive explanations of each algorithm
- **Multiple Data Distributions**: Choose from 5 statistical distributions for realistic data generation
- **Synthetic Data Generation**: Generate transaction datasets with configurable parameters and distributions
- **Algorithm Comparison**: Performance metrics and execution time tracking

<img src="Figs/Most%20Frequent%20Item%20Mining1.png" alt="Application Interface" style="width:800px;">
*Main application interface showing algorithm selection, dynamic parameters, and configuration options*

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or navigate to the project directory**:
    ```bash
    cd Frequent-Items-Mining-GA-PSO-main
    ```

2. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

#### Method 1: Using the provided batch file (Windows)
```bash
run_app.bat
```

#### Method 2: Manual startup

1. **Start the FastAPI backend** (in terminal 1):
    ```bash
    python main.py
    ```
    The API will be available at `http://localhost:8000`

2. **Start the Streamlit frontend** (in terminal 2):
    ```bash
    streamlit run streamlit_app.py
    ```
    The web app will open in your browser at `http://localhost:8501`

## 📊 Usage Guide

### Algorithm Selection

Choose from four powerful metaheuristic algorithms:

1. **🧬 Genetic Algorithm (GA)**: Evolutionary approach with selection, crossover, and mutation
2. **🐦 Particle Swarm Optimization (PSO)**: Swarm intelligence inspired by bird flocking
3. **🔥 Simulated Annealing (SA)**: Probabilistic technique inspired by metallurgy
4. **🐜 Ant Colony Optimization (ACO)**: Swarm intelligence based on ant foraging behavior

### Data Generation Parameters

- **Dataset Size**: Number of transactions to generate (100-10,000)
- **Distribution Type**: Statistical distribution for transaction length generation
  - **📈 Gaussian (Normal)**: Bell curve distribution (default)
  - **📊 Uniform**: Equal probability for all values in range
  - **🎯 Poisson**: Models count events and rare occurrences
  - **📉 Exponential**: Models waiting times and heavily skewed data
  - **🎲 Binomial**: Models success counts in fixed number of trials
- **List Length Parameters**: (vary by distribution)
  - **Min/Max Length**: Range boundaries for all distributions
  - **Mean/Std Dev**: For Gaussian distribution
  - **Lambda**: Rate parameter for Poisson/Exponential
  - **Trials/Probability**: For Binomial distribution
- **Item Range**: ID range for generated items (e.g., item1 to item6)

### Dynamic Algorithm Parameters

The interface automatically adapts based on your algorithm choice:

#### Genetic Algorithm Parameters
- **Population Size**: Number of candidate solutions per generation (10-500)
- **Generations**: Number of evolution iterations (10-500)  
- **Number of Parents**: Solutions selected for reproduction
- **Mutation Rate**: Probability of random changes (0.01-1.0)

#### Particle Swarm Parameters
- **Number of Particles**: Swarm size (10-200)
- **Iterations**: Number of optimization iterations (50-1000)
- **Inertia Weight (w)**: Previous velocity influence (0.1-2.0)
- **Cognitive Parameter (c1)**: Personal best influence (0.1-4.0)
- **Social Parameter (c2)**: Global best influence (0.1-4.0)

#### Simulated Annealing Parameters
- **Initial Temperature**: Starting temperature (100-5000)
- **Cooling Rate**: Temperature reduction rate (0.80-0.99)
- **Minimum Temperature**: Stopping temperature (0.1-10.0)
- **Max Iterations**: Maximum optimization steps (100-5000)

#### Ant Colony Parameters
- **Number of Ants**: Colony size (10-200)
- **Iterations**: Number of optimization cycles (50-500)
- **Alpha (α)**: Pheromone importance factor (0.1-5.0)
- **Beta (β)**: Heuristic importance factor (0.1-5.0)
- **Evaporation Rate**: Pheromone decay rate (0.1-0.9)
- **Pheromone Deposit (Q)**: Amount of pheromone deposited (10-500)

### Running Analysis

1. **Select Algorithm**: Choose your preferred metaheuristic algorithm
2. **Configure Parameters**: Use the sidebar sliders to adjust algorithm-specific settings  
3. **Run Analysis**: Click "🔍 Find Most Frequent Item"
4. **View Results**: 
    - Most frequent item and its count
    - Algorithm used and execution time
    - Interactive bar chart and pie chart
    - Detailed frequency table
    - Summary statistics

<img src="Figs/Most%20Frequent%20Item%20Mining2.png" alt="Analysis Results" style="width:800px;">
*Results visualization showing the most frequent item, algorithm performance metrics, and interactive charts*

### Theory Behind Page

Navigate to the "📚 Theory Behind" tab to explore:
- Detailed explanations of each algorithm
- Mathematical formulations and pseudocode
- Advantages and disadvantages comparison
- Parameter tuning guidelines
- Algorithm selection recommendations
- **Data Distributions Theory**: Complete guide to all 5 statistical distributions
- Distribution selection recommendations for different use cases

<img src="Figs/Most%20Frequent%20Item3.png" alt="Theory Behind Page" style="width:800px;">
*Comprehensive theory section with detailed algorithm explanations, mathematical formulations, and educational content*

## 🏗️ Architecture

### Backend (main.py)
- **FastAPI Framework**: RESTful API with automatic documentation
- **Multiple Algorithm Engines**: GA, PSO, SA, and ACO implementations
- **Dynamic Parameter Handling**: Algorithm-specific parameter validation
- **Data Generation**: Synthetic transaction data with configurable parameters
- **API Endpoints**:
  - `POST /mine-frequent-items`: Main analysis endpoint supporting all algorithms
  - `GET /algorithms`: Available algorithms and descriptions
  - `GET /health`: Health check endpoint
  - `GET /docs`: Interactive API documentation

### Frontend (streamlit_app.py)
- **Multi-tab Interface**: Algorithm Runner and Theory Behind pages
- **Dynamic Parameter Controls**: UI adapts to selected algorithm to prevent overlap
- **Algorithm Selection**: Dropdown with visual algorithm indicators
- **Comprehensive Visualization**: Plotly charts for data presentation
- **Theory Documentation**: In-depth algorithm explanations with examples
- **Real-time API Communication**: HTTP requests to backend with algorithm selection
- **Error Handling**: User-friendly error messages and status indicators

## 🧬 Algorithm Implementations

### Genetic Algorithm (GA)
- **Selection**: Tournament selection with configurable tournament size
- **Crossover**: Single-point crossover creating diverse offspring
- **Mutation**: Random position swapping with probabilistic control
- **Population Management**: Elitism preserving best solutions

### Particle Swarm Optimization (PSO)
- **Particle Movement**: Velocity and position updates based on personal and global bests
- **Social Learning**: Particles learn from swarm's collective knowledge
- **Inertia Control**: Balances exploration and exploitation dynamically
- **Boundary Handling**: Proper constraint management for discrete problems

### Simulated Annealing (SA)
- **Temperature Schedule**: Exponential cooling with configurable rate
- **Acceptance Criterion**: Metropolis criterion for solution acceptance
- **Neighborhood Generation**: Random swap operations for solution perturbation
- **Convergence Control**: Multiple stopping criteria (temperature and iterations)

### Ant Colony Optimization (ACO)
- **Pheromone Management**: Dynamic trail updating with evaporation
- **Probabilistic Construction**: Solutions built using pheromone and heuristic information
- **Collective Learning**: Colony-wide knowledge accumulation
- **Parameter Balancing**: Alpha and beta parameters control exploration vs exploitation

## 📁 Project Structure

```
Frequent-Items-Mining-GA-PSO-main/
├── main.py                 # FastAPI backend server
├── streamlit_app.py        # Streamlit frontend application
├── requirements.txt        # Python dependencies
├── run_app.bat             # Windows batch file to start both services
├── MiniPorj.ipynb          # Original Jupyter notebook
└── README.md               # This file
```

## 🔧 API Documentation

Once the FastAPI server is running, visit `http://localhost:8000/docs` for interactive API documentation with:
- Request/response schemas
- Try-it-out functionality
- Parameter descriptions
- Example payloads

## 🎯 Example Use Cases

1. **Market Basket Analysis**: Find frequently bought together items using different optimization approaches
2. **Algorithm Performance Comparison**: Compare GA, PSO, SA, and ACO on the same dataset
3. **Educational Tool**: Learn about various metaheuristic algorithms with hands-on experience
4. **Research Platform**: Experiment with algorithm parameters and synthetic datasets
5. **Optimization Benchmarking**: Test algorithm convergence speed and solution quality
6. **Parameter Sensitivity Analysis**: Study how different parameters affect algorithm performance

## 🛠️ Customization

### Adding New Algorithms
- **Differential Evolution**: Add another evolutionary algorithm
- **Harmony Search**: Implement music-inspired metaheuristic
- **Firefly Algorithm**: Add nature-inspired swarm intelligence
- **Tabu Search**: Implement memory-based local search

### Enhancing Existing Features
- **Algorithm Hybridization**: Combine multiple algorithms
- **Advanced Visualizations**: Add convergence plots and parameter sensitivity analysis
- **Export Functionality**: Save results and algorithm comparisons to CSV/JSON
- **Real Data Support**: Upload actual transaction datasets
- **Multi-objective Optimization**: Extend to handle multiple objectives simultaneously

### Performance Optimization
- **Parallel Processing**: Implement parallel execution for population-based algorithms
- **GPU Acceleration**: Use CUDA for intensive computations
- **Caching**: Implement intelligent caching for repeated analyses
- **Database Integration**: Store results, configurations, and algorithm performance metrics

## 🐛 Troubleshooting

### Common Issues

1. **API Connection Error**:
    - Ensure FastAPI server is running on the correct port
    - Check firewall settings
    - Verify the API URL in Streamlit sidebar

2. **Performance Issues**:
    - Reduce dataset size or number of generations
    - Lower population size for faster execution
    - Consider using smaller item ranges

3. **Installation Problems**:
    - Update pip: `python -m pip install --upgrade pip`
    - Use virtual environment: `python -m venv venv && venv\Scripts\activate`
    - Check Python version compatibility

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Original algorithm implementation from the Jupyter notebook
- Streamlit and FastAPI communities for excellent documentation
- Plotly for interactive visualization capabilities

## 📞 Support

For questions, issues, or suggestions:
- Open an issue in the repository
- Check the API documentation at `/docs`
- Review the troubleshooting section above

---

**Happy Mining! 🔍✨**