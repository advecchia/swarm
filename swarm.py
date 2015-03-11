#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Default libraries
import os
import sys
from copy import copy, deepcopy
import argparse
from time import time
import pickle
from random import random

# Development libraries
sys.path.append(os.path.join(os.getcwd(),os.path.dirname(__file__), 'src'))
from learning_ant_agent import LearningAntAgent
from learning_agent import LearningAgent
from ant import Ant
from constants import *

import traci
 
def parse():
    """Read the input and put the words in a dictionary.
    """

    parser = argparse.ArgumentParser(description="Execute an simple algorithm for swarm intelligence.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Simulation configuration group
    groupSim = parser.add_argument_group("Simulation group","The parameters that define the basics of simulation execution.")
    groupSim.add_argument(MIN_PORT, PORT, dest=PORT, nargs=1, metavar="number", default=DEFAULT_PORT, help="Flag to inform the port to communicate with traci-hub or SUMO.")

    # Swarm configuration group
    groupSwarm = parser.add_argument_group("Swarm configuration group","The parameters that define the properties for the Swarm algorithm.")
    groupSwarm.add_argument(MIN_BETA, BETA, MIN_EVAPORATION_RATE, EVAPORATION_RATE, dest=EVAPORATION_RATE, nargs=1, metavar="number", default=DEFAULT_EVAPORATION_RATE, help="The evaporation rate of pheromone.")
    groupSwarm.add_argument(MIN_THETA, THETA, MIN_RESPONSE_THRESHOLD, RESPONSE_THRESHOLD, dest=RESPONSE_THRESHOLD, nargs=1, metavar="number", default=DEFAULT_RESPONSE_THRESHOLD, help="The response threshold for a plan change for an ant.")

    # Learning configuration group
    groupLearning = parser.add_argument_group("Learning configuration group","The parameters that define the properties for the Learning algorithm.")
    groupLearning.add_argument(MIN_ALPHA, ALPHA, MIN_LEARNING_RATE, LEARNING_RATE, dest=LEARNING_RATE, nargs=1, metavar="number", default=DEFAULT_LEARNING_RATE, help="The alpha parameter(learning rate) of q-learning. A [0,1] value.")
    groupLearning.add_argument(MIN_GAMMA, GAMMA, MIN_DISCOUNT_FACTOR, DISCOUNT_FACTOR, dest=DISCOUNT_FACTOR, nargs=1, metavar="number", default=DEFAULT_DISCOUNT_FACTOR, help="The gamma parameter(discount rate) of q-learning. A [0,1) value.")
    groupLearning.add_argument(MIN_EPSILON, EPSILON, MIN_CURIOSITY, CURIOSITY, dest=CURIOSITY, nargs=1, metavar="number", default=DEFAULT_CURIOSITY, help="The probability of a random choice(agent curiosity). A [0,1] value.")
    groupLearning.add_argument(MIN_EPSILON_DECAY, EPSILON_DECAY, MIN_CURIOSITY_DECAY, CURIOSITY_DECAY, dest=CURIOSITY_DECAY, nargs=1, metavar="number", default=DEFAULT_CURIOSITY_DECAY, help="The rate at which epsilon decays(agent curiosity decay) after every random choice. Following: epsilon' = epsilon * (1 - epsilonDecay). A [0,1) value.")
    groupLearning.add_argument(MIN_EXPLORATION_PERIOD, EXPLORATION_PERIOD, dest=EXPLORATION_PERIOD, nargs=1, metavar="number", default=DEFAULT_EXPLORATION_PERIOD, help="The amount of time the agent is allowed to explore. A [0,infinite) value.")
    groupLearning.add_argument(MIN_QVALUE, QVALUE, dest=QVALUE, nargs=1, metavar="number", default=DEFAULT_QVALUE, help="The initial default value for an action in the algorithm. Q(S,A)=default_qvalue. A [0,1] value.")
    groupLearning.add_argument(MIN_REWARD_EXPONENT, REWARD_EXPONENT, dest=REWARD_EXPONENT, nargs=1, metavar="number", default=DEFAULT_REWARD_EXPONENT, help="Exponent used for calculating the reward of an action. Let Ol be the average occupancy for the lane l in the last period, and OMl be the max occupancy for the lane l. The reward is calculated as the sum for all lanes l of (1-Ol/OMl)^reward_exponent. A [0,infinite) value.")
    groupLearning.add_argument(MIN_BEST_QTABLE_FILE, BEST_QTABLE_FILE, dest=BEST_QTABLE_FILE, nargs=1, metavar="file-path", default=DEFAULT_BEST_QTABLE_FILE, help='Captures the file containing the best policy(q-table) for an network to be simulated.')

    # Hybrid configuration group
    groupHybrid = parser.add_argument_group("Hybrid configuration group","The parameters that define the properties for the Hybrid algorithm.")
    groupHybrid.add_argument(MIN_MEMORY_WINDOW, MEMORY_WINDOW, dest=MEMORY_WINDOW, nargs=1, metavar="number", default=DEFAULT_MEMORY_WINDOW, help="The size of memory of each traffic lights, i.e. the number of memorized plans. A [1,infinite) value.")
    #groupHybrid.add_argument(MIN_MEMORY_LOSS_FACTOR, MEMORY_LOSS_FACTOR, dest=MEMORY_LOSS_FACTOR, nargs=1, metavar="number", default=DEFAULT_MEMORY_LOSS_FACTOR, help="The time memory for acquired plans, i.e. the time in steps for the plan be forgotten. A [0,infinite) value.")
    groupHybrid.add_argument(MIN_MEMORY_LOSS_FACTOR, MEMORY_LOSS_FACTOR, dest=MEMORY_LOSS_FACTOR, nargs=1, metavar="number", default=DEFAULT_MEMORY_LOSS_FACTOR, help="A factor that is applied to the model quality simulating the loss of memory of the agent. A [0,1] value.")
    groupHybrid.add_argument(MIN_RHO, RHO, dest=RHO, nargs=1, metavar="number", default=DEFAULT_RHO, help="This parameter is an adjustment coefficient for the model’s quality.")
    groupHybrid.add_argument(MIN_OMEGA, OMEGA, dest=OMEGA, nargs=1, metavar="number", default=DEFAULT_OMEGA, help="This parameter specifies the relative importance of rewards and transitions for the model’s quality.")
    args = parser.parse_args()
    params = vars(args)

    #"""
    # Removes the parameter not inserted in the input
    for key,value in params.items():
        if value is None:
            del params[key]
        elif not isinstance(value, list):
            params[key] = []
            params[key].append(value)
    #"""
    return params

def main():
    # Parse the input command.
    params = parse()

    # For the use of SUMO with traci-hub.
    port = int(params[PORT].pop()) 
    if port >= 0:
        traci.init(port)
    else:
        traci.init(DEFAULT_PORT)

    # Swarm
    er = float(params[EVAPORATION_RATE].pop())
    rt = float(params[RESPONSE_THRESHOLD].pop())

    # Learning
    lr = float(params[LEARNING_RATE].pop())
    df = float(params[DISCOUNT_FACTOR].pop())
    c = float(params[CURIOSITY].pop())
    cd = float(params[CURIOSITY_DECAY].pop())
    ep = int(params[EXPLORATION_PERIOD].pop())
    qv = float(params[QVALUE].pop())
    re = float(params[REWARD_EXPONENT].pop())

    # Hybrid
    mw = int(params[MEMORY_WINDOW].pop())
    ml = float(params[MEMORY_LOSS_FACTOR].pop())
    r = float(params[RHO].pop())
    o = float(params[OMEGA].pop())

    log = open("log.txt", "w+")
    log.write("Input Statistics\n")
    log.write("MEMORY_WINDOW: "+str(mw)+" EVAPORATION_RATE: "+str(er)+" RESPONSE_THRESHOLD: "+str(rt)+
              "\nLEARNING_RATE: "+str(lr)+" DISCOUNT_FACTOR: "+str(df)+" CURIOSITY: "+str(c)+
              "\nCURIOSITY_DECAY: "+str(cd)+" EXPLORATION_PERIOD: "+str(ep)+" QVALUE: "+str(qv)+
              "\nREWARD_EXPONENT: "+str(re)+" MEMORY_LOSS_FACTOR: "+str(ml)+" RHO: "+str(r)+
              "\nOMEGA: "+str(o)+"\n")

    # Initializes the structure of ant system
    #agents = [Ant(id=id, memory_window=mw, evaporation_rate=er, response_threshold=rt) for id in traci.trafficlights.getIDList()]
    agents = [LearningAgent(id=id, memory_window=mw, learning_rate=lr, discount_factor=df, curiosity=c, curiosity_decay=cd, exploration_period=ep, qvalue=qv, reward_exponent=re, memory_loss_factor=ml) for id in traci.trafficlights.getIDList()]
    #agents = [LearningAntAgent(id=id, memory_window=mw, evaporation_rate=er, response_threshold=rt, learning_rate=lr, discount_factor=df, curiosity=c, curiosity_decay=cd, exploration_period=ep, qvalue=qv, reward_exponent=re, memory_loss_factor=ml, omega=o, rho=r) for id in traci.trafficlights.getIDList()]
    #print "LearningAntAgent List created"
    
    # Open the input file that contains a list of qtables for each traffic lights
    # the data is converted to pickle object for compatibility.
    if params.has_key(BEST_QTABLE_FILE):
        bqf = params[BEST_QTABLE_FILE].pop()
        try:
            with open(bqf,"r") as f:
                try:
                    # Read the pickle object data from input file and converts 
                    #its content to a python list object.
                    qtables = pickle.load(f)
                # There is no pickle data at all in the file.
                except EOFError:
                    qtables = []
                # Load the qtable data for each traffic light
                map(lambda agent: agent.load_qtable(qtables), agents)
        # There is no file to load
        except IOError as message:
            print "Warning: Failed to open best qtable input file."
            print "\nMessage: ", message

    # For stop execution
    total_departed = 0
    total_arrived = 0
    # For time execution calculus
    time_total = 0
    # For each iteration of the simulation.
    time_step = 1
    while True:

        # Double of end flow time.
        if time_step > 36000:
            traci.close()
            print "\nMessage: traci closed for double time limit."
            break

        # All vehicles have arrived after the flow beginning.
        if time_step > 3600 and total_arrived == total_departed:
            traci.close()
            print "\nMessage: traci normally closed."
            break

        try:
            start = time()
            #Executes one step
            traci.simulationStep()

            total_departed += traci.simulation.getDepartedNumber()
            total_arrived += traci.simulation.getArrivedNumber()

            # Executes the algorithm
            # Apply the update method to the agents
            map(lambda agent: agent.update(time_step), agents)

            time_step += 1
            time_diff = time() - start
            time_total += time_diff
            sys.stdout.write("\rTimestep #%d took %5.3f ms, Total %5.3f ms " % (time_step, time_diff, time_total))
            sys.stdout.flush()

        except traci.FatalTraCIError as message:
            print "\nMessage:", message
            traci.close()
            break

    # Save the learning qtables, even if there is no input file.
    if params.has_key(BEST_QTABLE_FILE):
        with open(bqf,"w+") as f:
            qtables = map(lambda agent: agent.save_qtable(), agents)
            # Converts the python list data in a pickle object to save at output file.
            pickle.dump(qtables, f)
    # TODO: Need a correction where all the algorithms can save their data to an output file.
    # the below code works correctly, but the todo need to be corrected first
#    else:
#        uniqueId = str(abs(hash(random())))
#        with open("qtables"+uniqueId+".txt","w+") as f:
#            qtables = map(lambda agent: agent.save_qtable(), agents)
#            # Converts the python list data in a pickle object to save at output file.
#            pickle.dump(qtables, f)

    # Save the log file
    log.write("\n".join(map(lambda agent: str(agent), agents)))
    #print "\n".join(map(lambda agent: str(agent), agents))
    log.close()
    exit()

if __name__ == '__main__':
    main()
