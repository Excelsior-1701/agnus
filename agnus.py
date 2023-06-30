import math
import numpy as np
import timeit
import os
#os.environ["NUMBA_ENABLE_CUDASIM"] = "1"; # For Debugging
#os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"; # For Debugging
import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from timeit import default_timer as timer

num_agents_default = 20000
size_field_default = 5000
chance_start_infected_default = .005
infection_time_default = 100
infection_range_default = 10
infection_chance_default = .15
speed_mult_default = 2
epochs_default = 500

RANDOM_SEED = 314
np.random.seed(RANDOM_SEED)

@cuda.jit(device=True)
def is_healthy(agent_array):
    return agent_array[4] == 0

@cuda.jit(device=True)
def is_infected(agent_array):
    return agent_array[4] == 1

@cuda.jit(device=True)
def is_recovered(agent_array):
    return agent_array[4] == 2

agent_lookup = {
    "x": 0,
    "y": 1,
    "direction": 2,
    "speed": 3,
    "state": 4,
    "infection_counter": 5
}

@cuda.jit
def run_epoch(agents, rng_states, reset):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < agents.shape[0]:
        agent = agents[pos]
        
        # Movement
        agents[pos][0] += math.cos(agents[pos][2] * 2 * math.pi) * agents[pos][3]
        agents[pos][1] += math.sin(agents[pos][2] * 2 * math.pi) * agents[pos][3]
        # Bounds Checking
        if agents[pos][0] > size_field:
            agents[pos][0] -= size_field
        elif agents[pos][0] < -1 * size_field:
            agents[pos][0] += size_field
        if agents[pos][1] > size_field:
            agents[pos][1] -= size_field
        elif agents[pos][1] < -1 * size_field:
            agents[pos][1] += size_field
        # Infection Counter
        if round(agents[pos][4]) == 1:
            agents[pos][5] += 1
            if agents[pos][5] > infection_time:
                agents[pos][4] = 2
        
        # Look for nearby infections
        if round(agents[pos][4]) == 0:
            for a in range(agents.shape[0]):
                if (
                    agents[a][4] == 1 and # Cell is infected
                    math.sqrt((agents[a][0] - agents[pos][0])**2 + (agents[a][1] - agents[pos][1])**2) < infection_range
                    # Cell is nearby
                ):
                # Now see if infected
                    if xoroshiro128p_uniform_float32(rng_states, pos) < infection_chance:
                        agents[pos][4] = 1
                        
        # Check Status
        if reset[0] == 1:
            agents[pos][6] = 1
            agents[pos][7] = 1
            agents[pos][8] = 1
        agents[pos][6] = int(agents[pos][6]) << 1
        agents[pos][7] = int(agents[pos][7]) << 1
        agents[pos][8] = int(agents[pos][8]) << 1
        if is_healthy(agent):
            agents[pos][6] += 1
        elif is_infected(agent):
            agents[pos][7] += 1
        elif is_recovered(agent):
            agents[pos][8] += 1
        
        
population = {"uninfected": [], 
              "infected": [], 
              "recovered": []}
epoch_count = []
num_states = 3
def summary(agents, epoch):
    uninfected = 0
    infected = 0
    recovered = 0
    for agent in agents:
        if round(agent[4]) == 0:
            uninfected += 1
        elif round(agent[4]) == 1:
            infected += 1
        elif round(agent[4]) == 2:
            recovered +=1
    population["uninfected"].append(uninfected)
    population["infected"].append(infected)
    population["recovered"].append(recovered)
    epoch_count.append(epoch)
    
    
'''Each of these gets one epoch of data'''
base = 6
@cuda.jit
def run_summary(results, agents):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < results.shape[0] * results.shape[1]:
        
        '''Kernels are assigned per epoch, then per state'''
        epoch = pos % results.shape[0]
        epoch_flag = 2 ** (epoch)
        state = pos // results.shape[0]
        
        count = 0
        for agent in agents:
            if int(agent[state + base]) & epoch_flag == epoch_flag:
                count += 1
        
        results[epoch][state] = count
        
epoch_block_size = 50 # Epcohs between data collection
def execute(epochs, agents, rng_states):
    reset = np.asarray([1])
    pop_breakdown = None
    count = 0 # Used to get last set of data
    for i in range(epochs):
        count += 1
        with cuda.pinned(agents):
            stream = cuda.stream()
            d_ary = cuda.to_device(agents, stream=stream)
            run_epoch[blockspergrid, threadsperblock](agents, rng_states, reset)
            reset[0] = 0
        if i % epoch_block_size == 0 and i != 0:
            reset[0] = 1
            count = 0
            results = np.zeros((epoch_block_size,num_states))
            run_summary[blockspergrid, threadsperblock](results, agents)
            results = np.rot90(results, k=-1)
            if pop_breakdown is None:
                pop_breakdown = results
            else:
                pop_breakdown = np.concatenate((pop_breakdown, results), axis=1)
    if count != 0:
        results = np.zeros((count,num_states))
        run_summary[blockspergrid, threadsperblock](results, agents)
        results = np.rot90(results, k=-1)
        if pop_breakdown is None:
            pop_breakdown = results
        else:
            pop_breakdown = np.concatenate((pop_breakdown, results), axis=1)
    return pop_breakdown


def build_agents(count):
    agents = []
    # Develop Initial base set
    for i in range(num_agents):
        x = np.random.uniform(0, size_field)
        y = np.random.uniform(0, size_field)
        direction = np.random.rand() # Tau angle
        speed = np.random.rand() * speed_mult
        if np.random.rand() < chance_start_infected:
            stage = 1
            infected = 0
        else:
            stage = 0
            infected = -1

        agents.append([x, y, direction, speed, stage, infected, 1, 1, 1])
    agents = np.array(agents)
    return agents


class Simulation:
    def define_simulation(self, agents=num_agents_default, 
                          size=size_field_default, 
                          csi=chance_start_infected_default,
                          inf_time =infection_time_default,
                          inf_range =infection_range_default,
                          inf_chance=infection_chance_default,
                          speed=speed_mult_default):
        global num_agents
        num_agents = agents
        global size_field 
        size_field = size
        global chance_start_infected
        chance_start_infected = csi
        global infection_time
        infection_time = inf_time
        global infection_range
        infection_range = inf_range
        global infection_chance
        infection_chance = inf_chance
        global speed_mult
        speed_mult = speed
        
    def __init__(self):
        self.current_seed = RANDOM_SEED
        
    def set_seed(self, number):
        self.current_seed = number
        np.random.seed(number)
        
    def configure_kernels(self):
        global threadsperblock
        global blockspergrid
        global rng_states 
        threadsperblock = 32
        blockspergrid = (num_agents + (threadsperblock - 1)) // threadsperblock
        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=RANDOM_SEED)
        
    def run_once(self, epochs=epochs_default):
        ts = timer()
        agents = build_agents(num_agents)
        pop_breakdown = execute(epochs, agents, rng_states)
        te = timer()
        print("Completed in:", te - ts)
        return pop_breakdown
        
    def monte_carlo(self, runs, epochs=epochs_default):
        breakdowns = []
        ts = timer()
        for i in range(runs):
            np.random.seed(RANDOM_SEED + i)
            agents = build_agents(num_agents)
            pop_breakdown = execute(epochs, agents, rng_states)
            breakdowns.append(pop_breakdown)
        te = timer()
        print("Completed in:", te - ts)
        return breakdowns