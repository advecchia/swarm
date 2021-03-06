#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""Utility classes for implementing traffic light agents.

copy of the file Sumo/Controle Semaforico/QLearning/tlcontrol/tlutils.py
"""

from collections import deque
import traci

# Export the utility classes
__all__ = ['Lane', 'Program', 'Phase']


VEHICLE_LENGTH = 5.0


def bounded(num, lower, upper):
    """Returns a num if it is in [lower, upper], or the closest extreme."""
    if num < lower:
        return lower
    elif num > upper:
        return upper
    else:
        return num


class AveragingWindow(object):
    """Keeps the average of a maximum number of data points.
    """

    def __init__(self, window_size):
        """Sets the maximum number of data points handled.
        """
        self.window_size = window_size
        self.points = deque()

        self.sum = 0
        self.count = 0.0
        self._average = None

    @property
    def average(self):
        """Average of the currently stored data points, or None for no points.
        """
        return self._average

    def add_point(self, value):
        """Adds a data point to the moving average, returns the removed one.
        """
        # Adds the new value and stores it on the queue
        self.sum += value
        self.points.append(value)

        if self.count < self.window_size:
            # Increments the count while adding new points
            self.count += 1
            removed_value = None
        if self.count >= self.window_size:
            # Substitutes a point when full
            removed_value = self.points.popleft()
            self.sum -= removed_value

        # Updates the average
        self._average = self.sum / self.count

        return removed_value


class Lane(object):
    """Accumulates information about a lane during the simulation.
    """

    @property
    def mean_halting_vehicles(self):
        """Mean number of halting vehicles on lane, for the observed window of time."""
        return self.__halting_window[0].average

    @property
    def halting_vehicles_decrease(self):
        """Difference between the halting vehicles in the two last observed windows."""
        return (self.__halting_window[1].average or 0) - self.__halting_window[0].average

    @property
    def normalized_mean_halting_vehicles(self):
        """Mean halting vehicles normalized to [0,3], but usually on [0,1].

        Even in extreme cases this number will rarely be larger than 1.
        """
        return 3 * (self.__halting_window[0].average / self.capacity)

    @property
    def normalized_halting_vehicles_decrease(self):
        """Normalized difference between halting vehicles in the two last windows.

        The normalization is the same as in normalized_mean_halting_vehicles.
        """
        return 3 * (self.halting_vehicles_decrease + self.capacity) / (2 * self.capacity)

    @property
    def mean_occupancy(self):
        """Mean occupancy of lane, for the observed window of time."""
        return self.__occupancy_window[0].average

    @property
    def occupancy_decrease(self):
        """Difference between the occupancy in the two last observed windows."""
        return (self.__occupancy_window[1].average or 0) - self.__occupancy_window[0].average

    @property
    def normalized_occupancy_decrease(self):
        """Occupancy decrease normalized to [0,1]."""
        return (self.occupancy_decrease + 1) / 2

    @property
    def mean_speed(self):
        """Mean speed on lane, for the observed window of time."""
        return self.__speed_window.average

    @property
    def max_speed(self):
        """Max speed allowed on lane."""
        return traci.lane.getMaxSpeed(self.id)

    def __init__(self, id, window_size):
        """Initializes the lane, obtaining information from SUMO.

        All mean values will be obtain from a moving window of
        window_size timesteps. All decreases are the difference
        between the last two consecutive windows of window_size
        timesteps (i.e. last - previous).
        """
        self.id = id
        self.__occupancy_window = [AveragingWindow(window_size) for n in (0,1)]
        self.__halting_window = [AveragingWindow(window_size) for n in (0,1)]
        self.__speed_window = AveragingWindow(window_size)

        self.length = traci.lane.getLength(id)
        self.capacity = self.length / VEHICLE_LENGTH

    def update(self):
        """Update the values with information from SUMO.

        This should be called once a timestep.
        """
        # Update occupancy
        occupancy = traci.lane.getLastStepOccupancy(self.id)
        overflow = self.__occupancy_window[0].add_point(occupancy)
        if overflow: self.__occupancy_window[1].add_point(overflow)

        # Update halting vehicles
        halting_number = traci.lane.getLastStepHaltingNumber(self.id)
        overflow = self.__halting_window[0].add_point(halting_number)
        if overflow: self.__halting_window[1].add_point(overflow)

        # Update speed
        speed = traci.lane.getLastStepMeanSpeed(self.id)
        self.__speed_window.add_point(speed)


class Program(traci.trafficlights.Logic, object):
    """Represents a sequence of phases, with their own duration.

    Since this is a subclass of traci.trafficlights.Logic,
    it can be used with methods from traci.trafficlights.
    """

    @staticmethod
    def from_sumo(program, lanes):
        """Create a program from data straight from TraCI."""
        instance = Program("", [])
        instance.fill_with(program, lanes)
        return instance

    def __init__(self, id, phases, current_phase_index=0, type=0, sub_parameter=0):
        super(Program, self).__init__(id, type, sub_parameter,
                                           current_phase_index, phases)
        try:
            self.__lanes = phases[0].state.keys()
        except IndexError:
            self.__lanes = []

    @property
    def id(self):
        """ID of the program within a traffic light."""
        return self._subID
    @id.setter
    def id(self, subID):
        """Updates the ID of the program within a traffic light."""
        self._subID = subID

    @property
    def phases(self):
        """List of phases on this traffic light.

        Be careful not to invalidate the current phase index!!
        """
        return self._phases
    @phases.setter
    def phases(self, val):
        self._phases = val

    @property
    def current_phase_index(self):
        """Index of the currently active phase.

        Raises an IndexError on modification if the given
        index doesn't correspond to any phases.
        """
        return self._currentPhaseIndex

    @current_phase_index.setter
    def current_phase_index(self, idx):
        self._phases[idx]
        self._currentPhaseIndex = idx

    @property
    def current_phase(self):
        """Currently active phase.

        Raises a ValueError on modification if the given phase
        is not contained on this program.
        """
        return self._phases[self._currentPhaseIndex]

    @current_phase.setter
    def current_phase(self, val):
        self._currentPhaseIndex = self._phases.index(val)
        self.set_modified()

    def set_phases_duration_limits(self):
        """Update the min and max time limits for all phases
        """
        #
        # Calculate the limits for green phases
        #
        green_phases = filter(lambda phase: phase.has_green, self.phases)
        green_phases_duration = map(lambda phase: phase.duration, green_phases)
        # Sum the list of phase duration that has green
        total_green_time = reduce(lambda ph1, ph2: ph1 + ph2, green_phases_duration)

        min_green_phase = min(green_phases_duration)
        max_green_phase = max(green_phases_duration)
        # Calculates the average of the three and a half values of the phases duration.
        min_green_duration = 0.75*(total_green_time/len(green_phases))
        # Calculates a max_duration using the smaller and greater values of the 
        # program, this will generate an approximative mean value.
        #max_green_duration = min_green_phase + (len(green_phases)-1) * (max_green_phase*0.75)
        max_green_duration = min_green_duration + (len(green_phases)-1) * (max_green_phase/2.0)
        # Update the phases
        map(lambda phase: phase.set_max_duration(max_green_duration), green_phases)
        map(lambda phase: phase.set_min_duration(min_green_duration), green_phases)

        #
        # Calculate the limits for not green phases
        #
        other_phases = filter(lambda phase: phase not in green_phases, self.phases)
        other_phases_duration = map(lambda phase: phase.duration, other_phases)
        # Sum the list of phase duration that has not green
        total_other_time = reduce(lambda ph1, ph2: ph1 + ph2, other_phases_duration)

        min_other_phase = min(other_phases_duration)
        max_other_phase = max(other_phases_duration)
        # Calculates the average of the three and a half values of the phases duration.
        min_other_duration = 0.75*(total_other_time/len(other_phases))
        # Calculates a max_duration using the smaller and greater values of the 
        # program, this will generate an approximative mean value.
        #max_other_duration = min_other_phase + (len(other_phases)-1) * (max_other_phase*0.75)
        max_other_duration = min_other_duration + (len(other_phases)-1) * (max_other_phase/2.0)
        # Update the phases
        map(lambda phase: phase.set_max_duration(max_other_duration), other_phases)
        map(lambda phase: phase.set_min_duration(min_other_duration), other_phases)
        #print [(phase.state, phase.green_lanes, phase.min_duration, phase.duration,phase.max_duration) 
        #       for phase in self.phases]

    def set_current_phase_index(self, idx):
        """Changes the current phase, throws an IndexError if there's no phase with given index."""
        self._phases[idx]
        self._currentPhaseIndex = idx

    def phase_cycle_time(self):
        """The cycle time of the program.
        """
        cycle_time = 0
        for phase in self.phases:
            cycle_time += phase.duration
        return cycle_time

    def phase_time_portion(self, phase_index=None):
        """Proportion of time used by the current phase in the cycle.
        """
        time_portion = 0
        cycle_time = self.phase_cycle_time()
        current_phase_duration = 0

        if phase_index in range(len(self.phases)):
            current_phase = self.phases[phase_index]
        else:
            current_phase = self.current_phase

        if current_phase.has_green:
            current_phase_duration = current_phase.duration

        try:
            time_portion = float(current_phase_duration)/float(cycle_time)

        #Keeps the time_portion as zero
        except ZeroDivisionError:
            time_portion = 0

        return time_portion

    def fill_with(self, from_sumo, lanes=None):
        """Fills with the given data from traci, and may change the controlled lanes."""
        self._subID = from_sumo._subID
        self._type = from_sumo._type
        self._subParameter = from_sumo._subParameter
        self._currentPhaseIndex = from_sumo._currentPhaseIndex
        self._phases = [Phase.from_sumo(data, lanes or self.__lanes)
                        for data in from_sumo._phases]

    def __repr__(self):
        class_name = self.__class__.__module__ + '.' + self.__class__.__name__
        return ("<%s id=%s current_phase_index=%d phases=%s>"
                % (class_name, self.id, self.current_phase_index, self.phases))

class Phase(traci.trafficlights.Phase, object):
    """A configuration of green/yellow/red signs for all controlled lanes of a traffic light.

    A Phase also has a current duration, as well as a minimum and maximum value for it.
    This class may be used to group lanes by the phase in which they have a green light.
    """

    DEFAULT_MIN =  15000 #20000
    DEFAULT_MAX = 120000
    DEFAULT_MIN_NOGREEN = 5000
    DEFAULT_MAX_NOGREEN = 60000

    @staticmethod
    def from_sumo(phase, lanes):
        instance = Phase([], "", 0)
        instance.fill_with(phase, lanes)
        instance.__set_duration(None, None)
        return instance

    def __init__(self, controlled_lanes, state_string, duration,
                 min_duration=None, max_duration=None):
        """Initializes a phase from the list of incoming lanes, a state string and the duration values.

        The state string consists of characters from gGyYrR, which refer to the lane
        in the same position of controlled_lanes, e.g. controlled_lanes = ["1","2"] and
        state_string = "rG" means that the lane "1" is red while lane "2" is green.

        Obviously, controlled_lanes and state_string must have the same length,
        otherwise a ValueError is raised.
        """
        super(Phase, self).__init__(duration, 0, 0, state_string)

        if len(controlled_lanes) != len(state_string):
            raise ValueError("The list of controlled lanes and the state "
                             "string must have the same length.")

        self.fill_with(self, controlled_lanes)

        self.__set_duration(min_duration, max_duration)

    def __set_duration(self, min_duration, max_duration):
        self.__min_duration = min_duration or (self.DEFAULT_MIN
                                               if self.has_green
                                               else self.DEFAULT_MIN_NOGREEN)
        self.__max_duration = max_duration or (self.DEFAULT_MAX
                                               if self.has_green
                                               else self.DEFAULT_MAX_NOGREEN)

    @property
    def state(self):
        """Dict from incoming lane ID to its current state (char from gGyYrR)."""
        return self.__state

    @property
    def has_green(self):
        """True if some of the incoming lanes has a green light."""
        return len(self.__green_lanes) > 0

    @property
    def green_lanes(self):
        """Tuple of incoming lanes with green light on this phase.
        """
        return tuple(l for l in self.__green_lanes)

    @property
    def min_duration(self):
        """Least time this phase must be active, in milliseconds."""
        return self.__min_duration
    # The setter is not used because its affect the map function
    def set_min_duration(self, val):
        self.__min_duration = val

    @property
    def max_duration(self):
        """Most time this phase must be active, in milliseconds."""
        return self.__max_duration
    # The setter is not used because its affect the map function
    def set_max_duration(self, val):
        self.__max_duration = val

    @property
    def duration(self):
        """Current duration, in milliseconds, always bounded by min_duration and max_duration."""
        return self._duration
    @duration.setter
    def duration(self, val):
        self._duration = bounded(val, self.min_duration, self.max_duration)

    @property
    def has_max_duration(self):
        """Verify if the duration has at max boundary."""
        return self.duration >= self.max_duration
    @property
    def has_min_duration(self):
        """Verify if the duration has at max boundary."""
        return self.duration <= self.min_duration

    def fill_with(self, from_sumo, lanes=None):
        """Uses a phase from traci to fill this, and may change the lanes.
        """

        if lanes is not None:
            self.__controlled_lanes = lanes
        if len(self.__controlled_lanes) != len(from_sumo._phaseDef):
            raise ValueError("Length mismatch between controlled lanes "
                             "and phase definition.")

        self._duration = from_sumo._duration
        self._duration1 = from_sumo._duration1
        self._duration2 = from_sumo._duration2
        self._phaseDef = from_sumo._phaseDef

        lanes_and_states = zip(self.__controlled_lanes, from_sumo._phaseDef)
        self.__state = dict(lanes_and_states)
        self.__green_lanes = set(l for l, s in lanes_and_states if s in "gG")


    def __repr__(self):
        class_name = self.__class__.__module__ + '.' + self.__class__.__name__
        return ("<%s green_lanes=%s duration=%dms>"
                % (class_name, self.green_lanes, self.duration))