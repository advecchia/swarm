#!/usr/bin/env python
#-*- coding: utf-8 -*-

import traci
from agent import Agent
from math import sqrt,pow
from collections import deque, defaultdict
from copy import deepcopy
from random import choice
import tls_util

class Ant(Agent):
    """An Ant agent, based in a bio-inspired approach of Ant System
    """

    """An enumeration of the possible changes in traffic light plane time,
    it will generate the all possible plans for ant agent plan's change.
    """
    TIME_CHANGE_FACTORS = [0.10, 0.25, 0.5] #[0.25, 0.33, 0.5]

    """The default value of threshold for each traffic plan,
    it will be used for evaluate the necessity of a plan change.
    """
    DEFAULT_THRESHOLD = 0.5

    #def __init__(self, id, memory_window, evaporation_rate, response_threshold, **kwds):
    def __init__(self, evaporation_rate, response_threshold, **kwds):
        #super(Ant, self).__init__(id, memory_window)
        super(Ant, self).__init__(**kwds)

        """The pheromone evaporation rate.
        
        dissipation rate beta
        """
        self.evaporation_rate = evaporation_rate
        
        """The response threshold is an indicative of success in a particular action of the ant. 
        
        theta_ij
        """
        self.response_threshold = []
        self.initial_theta = response_threshold
        
        """Keeps the pheromone density in time for each lane. {"lane":{"step":"density"}}

        d_l,t
        """
        self.pheromone_density = defaultdict(dict)
        
        """Keeps the pheromone density accumulated in time for each phase of the program. {"phase_index":{"step sum":"no vehicles"}} """
        self.pheromone_density_accumulated = defaultdict(dict)
        self.init_pheromone_density_accumulated()

        """Keeps the tendency of the current program to be reused by ant on time. 
        
        T__theta_ij(s_j)
        """
        self.program_tendency = []

        """A list of possible plans to an ant agent."""
        # Add the current plan to make it a possible plan
        self.possible_plans = [[deepcopy(self.program_default), self.DEFAULT_THRESHOLD]] #self.program_current
        # Create the new possible plans
        self.init_possible_plans()
        # Init the change's plan tendency
        self.init_program_tendency()
        # Init the agent's response threshold for each plan
        self.init_response_threshold(response_threshold)
        #print "Ant agent created"

    def get_max_plan_id(self):
        return max(map(lambda pid: int(pid), [p[0].id for p in self.possible_plans]))

    def add_plan(self, plan):
        self.possible_plans.append([plan, self.DEFAULT_THRESHOLD])
        self.response_threshold.insert(len(self.possible_plans), self.initial_theta)
        self.program_tendency.insert(len(self.possible_plans), self.DEFAULT_THRESHOLD)

    def remove_plan(self, id):
        """ Remove a plan when the memory is overloaded.
        """
        plan_to_remove = filter(lambda p: p[0].id==id, [p for p in self.possible_plans])
        if plan_to_remove:
            index = self.get_plan_index(plan_to_remove[0][0])
            self.program_tendency.remove(self.program_tendency[index])
            self.response_threshold.remove(self.response_threshold[index])
            self.possible_plans.remove(plan_to_remove[0])

    def get_plan_index(self, plan):
        """Return the plan index in possible plans, that match
        with the current program.
        """
        return [idx for (idx,l) in enumerate(self.possible_plans) if l[0].id == plan.id][-1]

    @property
    def current_plan_index(self):
        """Return the 'first' current plan index in possible plans, that match
        with the current program.
        """
        return [idx for (idx,l) in enumerate(self.possible_plans) if l[0].id == self.program_current.id][-1]

    def init_pheromone_density_accumulated(self):
        """Initializes the dictionary that keeps the accumulated density of pheromone.
        """
        for idx, phase in enumerate(self.program_current.phases):
                self.pheromone_density_accumulated[idx][0] = 0

    def init_program_tendency(self):
        """Initializes the values of the execution's tendency of determined plan for the agent.
        """
        for idx, l in enumerate(self.possible_plans):
            self.program_tendency.insert(idx, l[1])

    def init_response_threshold(self, response_threshold):
        """Initializes the values of the response threshold for all possible plans for the agent.
        """
        for idx, l in enumerate(self.possible_plans):
            self.response_threshold.insert(idx, response_threshold)

    def init_possible_plans(self):
        """Creates a set of the all possible plans that the ant agent can choose.

        The number of plans can vary according with the number of time factors.
        """
        # Create a new program
        new_program = self.program_default # self.init_program_current()
        #new_program = deepcopy(self.program_current)
        program_id = int(new_program.id)
        green_phases = [(idx, phase) for idx, phase in enumerate(new_program.phases)
                           if phase.has_green]
        # Take the plan time cycle duration
        cycle_length = sum(phase.duration for phase in new_program.phases
                       if phase.has_green)

        # Iterate over all the relevant phases
        for (idx, phase) in green_phases:
            other_phases = [(oidx, ophase) for oidx, ophase in enumerate(new_program.phases)
                                if ophase.has_green and oidx != idx]
            # and for all time factors
            for factor in self.TIME_CHANGE_FACTORS:
                # Make a copy of the new program and update it
                copy_program = deepcopy(new_program)
                program_id += 1
                copy_program.id = str(program_id)

                # Calculate the time increment for selected phase
                increment = 0
                for (oidx, ophase) in other_phases:
                    decrement = ophase.duration * factor
                    # Decrement the other phases
                    copy_program.phases[oidx].duration -= decrement
                    increment += decrement
                # Increment the selected phase
                copy_program.phases[idx].duration += increment

                # Compensate errors in time cycle
                phases_duration = [phase.duration for phase in new_program.phases
                                   if phase.has_green]
                error = cycle_length - sum(phases_duration)

                ophases_len = len(other_phases)
                if error > 0:
                    # The cycle is upward
                    if copy_program.phases[idx].has_max_duration:
                        for (oidx, phase) in other_phases:
                            copy_program.phases[oidx].duration += int(error/ophases_len)
                    else:
                        copy_program.phases[idx].duration += error
                elif error < 0:
                    # The cycle is downward
                    if copy_program.phases[idx].has_min_duration:
                        for (oidx, phase) in other_phases:
                            copy_program.phases[oidx].duration += int(error/ophases_len)
                    else:
                        copy_program.phases[idx].duration += error                
                # Save the new program
                self.possible_plans.append([copy_program, self.DEFAULT_THRESHOLD])

    def act(self):
        """Choose a new plan based on agent's tendency to changes plans. 
        """
        super(Ant, self).act()
        current_agent_tendency = self.program_tendency[self.current_plan_index]
        programs_indexes = [i for (i, l) in enumerate(self.possible_plans) if l[1] >= current_agent_tendency]
        # If there is any index
        data = ["\n"]
        if programs_indexes:
            data.append("Acting")
            # Choose at random one of the best possible plans
            index = choice(programs_indexes)
            program = self.possible_plans[index][0]
            # Keeps the same phase index for traffic light
            program.current_phase_index = traci.trafficlights.getPhase(self.id) #self.program_current.current_phase_index
            self.program_current.fill_with(program, self.lanes)
            #self.update_program_current(self.program_current)
            #traci.trafficlights.setProgram(self.id, program.id)
        else:
            data.append("There's no best plan to choose.")
        data.append("TlsId: %s CurrentPlan %s: "%(self.id, self.current_plan_index))
        #print "\n".join(data)
        print "Ant actuate"

    def update(self, step):
        """Updates the necessary attributes of this class. And call the action.
        """
        super(Ant, self).update(step)
        self.update_pheromone_density()
        self.update_pheromone_density_accumulated()

        if self.is_time_to_act():
            self.update_response_threshold()
            self.program_keep_tendency()
            self.act()
            self.update_program_current(self.program_current)

            data = ["\n"]
            #data.append("Step: "+str(step))
            #data.append("CycleTime: "+str(self.current_cycle_time))
            data.append("Stimuli: "+str([values for (p,values) in self.possible_plans]))
            data.append("Threshold: "+str(self.response_threshold)) #[self.current_plan_index]
            #data.append("ProgramCurId: "+str(self.program_current.id))
            #data.append("ProgramCurrent: "+str(self.program_current))
            data.append("Tendency: "+str(self.program_tendency))
            #data.append("PheromeDensityAccum: "+str(self.pheromone_density_accumulated))
            #print "\n".join(data)

    def update_program_current(self, program):
        super(Ant, self).update_program_current(program)
        traci.trafficlights.setProgram(self.id, program.id)

    def update_pheromone_density_accumulated(self):
        """Updates the density of pheromone for each phase of the current program.
        """
        #Takes the value for the last step (deque last element)
        step = self.memory_window[-1]
        #TODO: maybe not the best choice, but works
        #phase_index = self.program_current.current_phase_index
        phase_index = traci.trafficlights.getPhase(self.id)
        # Get the last key for accumulated density
        accumulated_steps = self.pheromone_density_accumulated[phase_index].keys()[-1]
        last_density = self.pheromone_density_accumulated[phase_index][accumulated_steps]
        occupancy = 0

        #Take the lanes that are not in green state and accumulate their occupancy
        #for lane_id in set(self.lanes).difference(self.program_current.current_phase.green_lanes):
        for lane_id in set(self.lanes).intersection(self.program_current.phases[phase_index].green_lanes):
            occupancy += self.pheromone_density[lane_id][step] #self.vehicular_occupancy[lane_id][step]

        #Increases the number of steps and adds the new value of pheromone for the phase
        self.pheromone_density_accumulated[phase_index][accumulated_steps+1] = last_density + occupancy #round(last_density + occupancy, 6)

        #Clear the last position of the dictionary for memory reduction
        del self.pheromone_density_accumulated[phase_index][accumulated_steps]

    def update_pheromone_density(self):
        """ time need to be the overall time passed, don't need to be passed on function
            [sum t=0_w ( EVAPORATION_RATE ^-t)*(vehicularDensity(t)) ] / [sum t=0_w ( EVAPORATION_RATE ^-t)]
            t timestep 
            w overall time
        """
        for lane_id in set(self.lanes):
            # Auxiliary variables initialization
            num = den = 0
            for w, time_step in enumerate(self.memory_window):
                evaporation = 1.0 / pow(self.evaporation_rate, w)
                num += evaporation * self.vehicular_occupancy[lane_id][time_step]
                den += evaporation

            try:
                density = num / den
            except ZeroDivisionError:
                #Maybe, there is no steps in the array
                density = 0

            self.pheromone_density[lane_id][time_step] = density

    def program_current_stimulus(self):
        """Stimulus associated to the prioritized semaphore plan
            S de um plano j
            Sj = sum k=0_n (feromoneDensity(IN_k,t)) * D_k
            k fase plano semaforico
            n numero de fases do plano semaforico j
            
            D_k = (tempo_fim - tempo_inicio) / tempo_ciclo
            ou seja o tempo de sinal verde da fase
            
        stimulus s_j
        """
        stimulus = 0    #For plan j
        for idx, phase in enumerate(self.program_current.phases):
            max_steps = max(self.pheromone_density_accumulated[idx].keys())
            try:
                density = float(self.pheromone_density_accumulated[idx][max_steps])/max_steps
            except ZeroDivisionError:
                density = 0
            stimulus += density * self.program_current.phase_time_portion(idx)

        # Updates the current stimulus's plan.
        try:
            self.possible_plans[self.current_plan_index][1] = stimulus
        except IndexError as message:
            print "Warning: There is no current plan for ant agent."
            print "\nMessage:", message

        return stimulus

    def program_keep_tendency(self):
        """Trend of the agent to change the plan, according to the relation to the stimulus and the response threshold.
            T(Lij, Sj) = (Sj ^ 2) / (Sj ^ 2) + (Lij ^ 2)
            Lij its the response threshold of the agent i with relation to plan j
            Sj its the stimulus for the task j
        """
        #tasks tinha valor default 2 anteriormente, como definido no livro do bonabeau
        #tasks = len(self.program_current.current_phase.green_lanes)
        tasks = 2
        stimulus = pow(self.program_current_stimulus(),tasks)
        tendency = stimulus / float(stimulus + pow(self.response_threshold[self.current_plan_index],tasks))

        try:
            self.program_tendency[self.current_plan_index] = round(tendency, 6)
            #self.program_tendency[self.current_plan_index] = tendency
        except IndexError as message:
            print "Warning: There is no current plan for ant agent."
            print "\nMessage:", message
        data = ["\n"]
        data.append("Index: "+str(self.current_plan_index))
        #data.append("Tasks: "+str(tasks))
        data.append("Stimulus: "+str(stimulus))
        data.append("Tendency: "+str(tendency))
        #print " # ".join(data)

    def update_response_threshold(self):
        """The threshold for keeping or not the current semaphore plan.
        """
        deviation = self.deviation()
        delta = 1 - (2 * deviation)
        #tls_util.bounded(5, -1, 1)
        #Updates the response threshold based in the learning coefficient and in the time interval.
        current_cycle_time = round(self.program_current.phase_cycle_time() / 1000)
        #response = delta * round(delta * len(self.memory_window), 6)
        #response = delta * len(self.memory_window)
        #response = delta * current_cycle_time
        response = delta * (current_cycle_time/self.memory_window[-1])
        self.response_threshold[self.current_plan_index] -= response
        data = ["\n"]
        data.append("Deviation: "+str(deviation))
        data.append("Delta: "+str(delta))
        data.append("CurCycleTime: "+str(current_cycle_time))
        data.append("Response: "+str(response))
        data.append("ActualResponse: "+str(self.response_threshold[self.current_plan_index]))
        #print " # ".join(data)
        
        
    def deviation(self):
        """Sample standard deviation for pheromone trail accumulated in controlled lane is calculated.
            Then return the average for all deviation calculated.
        """
        deviation = []
        for lane_id in set(self.lanes):
            deviation_aux = 0
            lane_occupancy = [self.vehicular_occupancy[lane_id][step] for step in self.memory_window]
            average = self.average(lane_occupancy)
            for occupancy in lane_occupancy:
                deviation_aux += pow(occupancy-average, 2)
            try:
                #The -1 is if has only one element, then the average is zero 
                deviation_aux = sqrt(float(deviation_aux)/float(len(lane_occupancy)-1))
            except ZeroDivisionError:
                #Exists only one element
                deviation_aux = 0
            deviation.append(deviation_aux)

        #Returns the average of the standard deviations
        return self.average(deviation)

    def __repr__(self):
        data = [super(Ant, self).__repr__()]
        data.append("\nAnt Class")
        data.append("evaporation rate="+str(self.evaporation_rate))
        data.append("response threshold="+str(self.response_threshold))
        data.append("pheromone density="+str(self.pheromone_density))
        data.append("pheromone density accumulated="+str(self.pheromone_density_accumulated))
        data.append("program tendency="+str(self.program_tendency))
        return "\n".join(data)

    def __str__(self):
        data = [super(Ant, self).__str__()]
        data.append("\nAnt Class")
        data.append("evaporation rate="+str(self.evaporation_rate))
        data.append("response threshold="+str(self.response_threshold))
        data.append("pheromone density="+str(self.pheromone_density))
        data.append("pheromone density accumulated="+str(self.pheromone_density_accumulated))
        data.append("program tendency="+str(self.program_tendency))
        return "\n".join(data)
