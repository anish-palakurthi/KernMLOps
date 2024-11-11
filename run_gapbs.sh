#!/bin/bash

# GAPBS Benchmark Runner
# This script automates the installation and running of the GAP Benchmark Suite

set -e  # Exit on any error

# Configuration variables - modify these as needed
GRAPH_SIZES=(16 20 22)  # 2^N vertices
NUM_ITERATIONS=16       # Number of iterations per benchmark
NUM_THREADS=8          # Number of OpenMP threads
LOG_FILE="gapbs_benchmark_$(date +%Y%m%d_%H%M%S).log"
INSTALL_DIR="$HOME/gapbs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logger function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$LOG_FILE"
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check for C++ compiler
    if ! command -v g++ &> /dev/null; then
        error "g++ is required but not installed. Please install build-essential."
    fi
    
    # Check for OpenMP
    if ! echo "#include <omp.h>" | g++ -fopenmp -x c++ - -c -o /dev/null &> /dev/null; then
        warn "OpenMP not available. Installing libomp-dev..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y libomp-dev
        elif command -v yum &> /dev/null; then
            sudo yum install -y libomp-devel
        else
            error "Could not install OpenMP. Please install manually."
        fi
    fi
    
    # Check for git
    if ! command -v git &> /dev/null; then
        error "git is required but not installed."
    fi
    
    log "All requirements satisfied."
}

# Install GAPBS
install_gapbs() {
    log "Installing GAPBS..."
    
    if [ -d "$INSTALL_DIR" ]; then
        warn "Installation directory already exists. Removing..."
        rm -rf "$INSTALL_DIR"
    fi
    
    git clone https://github.com/sbeamer/gapbs.git "$INSTALL_DIR" || error "Failed to clone repository"
    cd "$INSTALL_DIR" || error "Failed to enter installation directory"
    
    # Compile
    log "Compiling GAPBS..."
    make clean
    make -j "$(nproc)" || error "Compilation failed"
    
    log "Installation completed successfully."
}

# Set up environment
setup_environment() {
    log "Setting up environment..."
    
    export OMP_NUM_THREADS=$NUM_THREADS
    export OMP_SCHEDULE=static
    export KMP_AFFINITY=compact,1
    
    # Increase stack size
    ulimit -s unlimited || warn "Failed to set unlimited stack size"
    
    log "Environment configured."
}

# Run benchmarks
run_benchmarks() {
    local benchmarks=("bfs" "pr" "cc" "bc" "sssp" "tc")
    local graph_types=("rmat" "uniform")
    
    log "Starting benchmark suite..."
    
    for size in "${GRAPH_SIZES[@]}"; do
        log "Testing graphs of size 2^$size vertices..."
        
        for type in "${graph_types[@]}"; do
            log "Generating $type graph..."
            
            # Check if generator exists
            if [ ! -f "./generator" ]; then
                error "Generator executable not found. Please ensure it is compiled."
            fi
            
            if [ "$type" == "rmat" ]; then
                ./generator -g "$size" -s "$((size+4))" || error "Failed to generate RMAT graph"
            else
                ./generator -g "$size" -s "$((size+4))" -r || error "Failed to generate uniform random graph"
            fi
            
            for bench in "${benchmarks[@]}"; do
                log "Running $bench benchmark..."
                
                # Run benchmark with timing
                TIMEFORMAT="%R"
                runtime=$( { time ./"$bench" -g "$size" -n "$NUM_ITERATIONS" -v; } 2>&1 )
                
                # Log results
                echo "Benchmark: $bench, Graph: $type, Size: 2^$size, Runtime: ${runtime}s" >> "$LOG_FILE"
            done
        done
    done
}

# Generate report
generate_report() {
    log "Generating benchmark report..."
    
    echo -e "\nBenchmark Summary Report" > "report_$(date +%Y%m%d_%H%M%S).txt"
    echo "=========================" >> "report_$(date +%Y%m%d_%H%M%S).txt"
    echo "System Information:" >> "report_$(date +%Y%m%d_%H%M%S).txt"
    echo "CPU: $(lscpu | grep "Model name" | sed 's/Model name: *//')" >> "report_$(date +%Y%m%d_%H%M%S).txt"
    echo "Memory: $(free -h | awk '/^Mem:/ {print $2}')" >> "report_$(date +%Y%m%d_%H%M%S).txt"
    echo "Threads: $NUM_THREADS" >> "report_$(date +%Y%m%d_%H%M%S).txt"
    echo -e "\nBenchmark Results:" >> "report_$(date +%Y%m%d_%H%M%S).txt"
    grep "Benchmark:" "$LOG_FILE" >> "report_$(date +%Y%m%d_%H%M%S).txt"
    
    log "Report generated."
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    cd "$INSTALL_DIR" || return
    make clean
    rm -f benchmark.graph*
    log "Cleanup completed."
}

# Main execution
main() {
    echo "GAP Benchmark Suite Runner"
    echo "========================="
    
    check_requirements
    install_gapbs
    setup_environment
    run_benchmarks
    generate_report
    cleanup
    
    log "Benchmark suite completed successfully!"
    log "Results are available in $LOG_FILE"
    log "Detailed report available in report_$(date +%Y%m%d_%H%M%S).txt"
}

# Trap Ctrl+C and call cleanup
trap cleanup INT

# Run main function
main