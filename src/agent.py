#!/usr/bin/env python
#-*- coding: utf-8 -*-

import traci
from tls_util import Program
from collections import deque, defaultdict
from copy import deepcopy
# TODO: verify the need to change this constant.
# The size of a default vehicle, used in density calculus.
VEHICLE_SIZE = 5

class Agent(object):
    """A simple agent, inspired in a traffic light.
    """

    # Three cycles before to actuate
    COUNT_TO_ACTUATE = 3
    # Three cycles plus before to actuate, usually to wait before net occupation
    INITIAL_ACT_DELAY = 3

    def __init__(self, id, memory_window, **kwds):

        """The agent identifier. """
        self.id = id

        """The controlled lanes """
        #self.lanes = tuple(filter(lambda x: x.find(':') == -1, traci.trafficlights.getControlledLanes(self.id)))
        # Maybe the lanes will be duplicated, if it occurs represents that the 
        # lane at traffic lights turns into more than one lane. 
        self.lanes = tuple(traci.trafficlights.getControlledLanes(self.id))

        """The controlled links. """
        self.links = traci.trafficlights.getControlledLinks(self.id)

        """The current program that is not yet modified. """
        self.program_current = self.init_program_current()
        self.program_default = deepcopy(self.program_current)

        """Keeps the number of vehicles in time for each lane. {"lane":{"step":"no vehicles"}} """
        self.stopped_vehicles = defaultdict(dict)

        """Keeps the occupancy of vehicles in time for each lane. {"lane":{"step":"occupancy"}} """
        self.vehicular_occupancy = defaultdict(dict)

        """Keeps the density of vehicles in time for each lane. {"lane":{"step":"density"}} """
        self.vehicular_density = defaultdict(dict)

        """Keeps a list of 'WINDOW_SIZE' length for the current time steps to evaluate. 

        time interval w
        """
        if memory_window > 1:
            self.memory_window = deque(maxlen=memory_window)
        else:
            self.memory_window = deque(maxlen=1)
            print ("Warning: The memory window %d not contained in [1,infinite), so its value has past to 1.\n"
                             % memory_window)

        """ Keeps the initial traffic light cycle time, and after use it to actuate. (/1000 is ms) """
        self.current_cycle_time = round(self.program_current.phase_cycle_time() / 1000)

        """ Number of cycle to wait before take an action """
        self.count_to_actuate = self.COUNT_TO_ACTUATE + self.INITIAL_ACT_DELAY

        #TODO: Need to be corrected because of the multiple heritage
        # Is used to take action control for Learning Ant Agent 
        self.act_hack = False
        #print "Agent created"

    def update_stopped_vehicles(self, step):
        """Update the stopped vehicles for each lane.
        """
        for lane_id in self.lanes:
            self.stopped_vehicles[lane_id][step] = traci.lane.getLastStepHaltingNumber(lane_id)

    def update_vehicular_occupancy(self, step):
        """Update the vehicular occupancy for each lane.
        """
        # http://sumo.sourceforge.net/doc/current/docs/userdoc/Simulation/Output/Lane-_or_Edge-based_Traffic_Measures.html
        for lane_id in self.lanes:
            self.vehicular_occupancy[lane_id][step] = round(traci.lane.getLastStepOccupancy(lane_id), 6)

    def update_vehicular_density(self, step):
        """Update the vehicular density for each lane.
        """
        # http://sumo.sourceforge.net/doc/current/docs/userdoc/Simulation/Output/Lane-_or_Edge-based_Traffic_Measures.html
        for lane_id in self.lanes:
            self.vehicular_density[lane_id][step] = round(traci.lane.getLastStepVehicleNumber(lane_id)/float(traci.lane.getLength(lane_id)/VEHICLE_SIZE), 6)

    def update(self, step):
        """Updates the necessary attributes of this class.
        """
        self.current_cycle_time -= 1
        if self.current_cycle_time <= 0:
            self.current_cycle_time = round(self.program_current.phase_cycle_time() / 1000)
            self.count_to_actuate -= 1

        self.memory_window.append(step)
        self.update_stopped_vehicles(step)
        self.update_vehicular_occupancy(step)
        self.update_vehicular_density(step)

    def is_time_to_act(self):
        """Verify if is time to act.
        """
        if self.act_hack:
            return False

        if self.count_to_actuate <= 0:
            return True
        else:
            return False

    def act(self):
        """Because this agent has no action, this only restarts the counter's action.
        """
        self.count_to_actuate = self.COUNT_TO_ACTUATE
        print "Agent actuate"

    def init_program_current(self):
        """Capture the initial signal program and return it.
        """
        programs = traci.trafficlights.getCompleteRedYellowGreenDefinition(self.id)
        current_id = traci.trafficlights.getProgram(self.id)
        current = next(prog for prog in programs if prog._subID == current_id)
        program = Program.from_sumo(current, self.lanes)
        # Updates the limits of phases time.
        #program.set_phases_duration_limits()
        return program

    def update_program_current(self, program):
        """Update the traffic lights in real time at simulation.
        
        Notice that this will occurs only after the current program 
        terminates its actual time phase.
        """
        #print traci.trafficlights.getNextSwitch(self.id), traci.trafficlights.getPhase(self.id)
        if program.current_phase_index != traci.trafficlights.getPhase(self.id):
            program.current_phase_index = traci.trafficlights.getPhase(self.id)
        traci.trafficlights.setCompleteRedYellowGreenDefinition(self.id, program)
#        else:
#            print "warning: phase indexes are not equal, the program is not changed: ProgramCurrent="+str(program.current_phase_index)+" SumoCurrent="+str(traci.trafficlights.getPhase(self.id))

    def average(self, items):
        """Arithmetic average.
        """
        if items == []:
            return 0
            #raise ValueError("Cannot calculate the arithmetic mean of an empty iterable.")
        return float(sum(items))/len(items)

    def __repr__(self):
        data = []
        data.append("\nAgent Class")
        data.append("id="+str(self.id))
        data.append("memory window="+str(self.memory_window))
        data.append("lanes="+str(";".join(self.lanes)))
        data.append("links="+str(self.links))
        data.append("program current="+str(self.program_current))
        data.append("stopped vehicles="+str(self.stopped_vehicles))
        data.append("vehicular occupancy="+str(self.vehicular_occupancy))
        data.append("vehicular density="+str(self.vehicular_density))
        return "\n".join(data)

    def __str__(self):
        data = []
        data.append("\nAgent Class")
        data.append("id="+str(self.id))
        data.append("memory window="+str(self.memory_window))
        data.append("lanes="+str(";".join(self.lanes)))
        data.append("links="+str(self.links))
        data.append("program current="+str(self.program_current))
        data.append("stopped vehicles="+str(self.stopped_vehicles))
        data.append("vehicular occupancy="+str(self.vehicular_occupancy))
        data.append("vehicular density="+str(self.vehicular_density))
        return "\n".join(data)