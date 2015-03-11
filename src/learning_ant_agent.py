#!/usr/bin/env python
#-*- coding: utf-8 -*-

from random import choice
from copy import deepcopy
import traci
from ant import Ant
from learning_agent import LearningAgent
from rlcd import RLContextDetection

class LearningAntAgent(Ant,LearningAgent):#, Learning
    """A Learning Ant agent
    
    This approach capture the task resolution for Ant agent, upgrades it with a
    Reinforcement Learning for non-stationary environment that Detect its Context for
    learn new models in an unsupervised machine learning simulation.
    """
    def __init__(self, memory_loss_factor, omega, rho, **kwds):
        super(LearningAntAgent, self).__init__(**kwds)

        #"""The factor that influences loss of the learned plans. """
        #self.memory_loss_factor = memory_loss_factor
        """Limited memory of the best programs acquired. A set of tuples {("value","program")} """
        self.program_memory = RLContextDetection(omega, rho, len(self.qtable.states), memory_loss_factor)#(omega, rho)
        self.program_memory.add_items(self.possible_plans)
        #self.program_memory.add_item(self.program_current)
        #print "LearningAntAgent created"


    def update(self, step):
        """Updates the necessary attributes of this class. And call the action.
        """
        # Wtih this hack, the Ant and Learning Agents do not executes their action phase
        # so I can make here their actions. This is necessary because the multiple
        # heritage is not working in a good way.
        self.act_hack = True
        super(LearningAntAgent, self).update(step)
        self.act_hack = False

        if self.is_time_to_act():
            # For Ant agent
            self.update_response_threshold()
            self.program_keep_tendency()

            self.act()
            self.update_program_current(self.program_current)

        """state machine
                ANT-AGENT
                LEARNING-AGENT
                LEARNING-ANT-AGENT

            for each step
                #reads the traffic light state (Agent)
                #saves his state (Agent)
                #calculate and analyse the pheromone density (Ant)
                #calculate and analyse the stimulus (Ant)  
                #updates the threshold (Ant)
                Avalia a qualidade do mesmo?
                #take states and actions (Learning)

                saves a plane (LearningAntAgent)
                Reforça planos valor_plano + fator_aprendizado 
                Esquece planos valor_plano * memory_loss_factor < random()
                compares saved plans (LearningAntAgent)
                if good changes occur
                    updates a plan (LearningAntAgent)
                    update the simulation state (Agent)
        """
        pass

    def act(self):
        #super(LearningAntAgent, self).act()
        #
        # for agent act()
        #
        self.count_to_actuate = self.COUNT_TO_ACTUATE

        #
        # for Learning Agent act()
        #
        # Takes the tuple that points to an new_state in qtable
        new_state = self.discretize_state()

        # Chooses and executes the new action
        new_action = self.qtable.act(new_state)

        # Update the qtable in the N>1 actions
        if self.last_state:
            #print "P.Ids: ", map(lambda pid: int(pid), [p[0].id for p in self.possible_plans])
            reward = self.calculate_reward()
            self.qtable.observe(self.last_state, self.last_action, new_state, reward)

            #
            # for Learning Ant Agent act()
            #
            # Detect a new context and actuate 
            self.program_memory.update(self.last_state, self.last_action, new_state, reward)
            #
            # for Ant act()
            #
            current_agent_tendency = self.program_tendency[self.current_plan_index]
            programs_indexes = [i for (i, p) in enumerate(self.possible_plans) if p[1] >= current_agent_tendency]
            plans = []
            for index in programs_indexes:
                plans.append(self.possible_plans[index][0])
            plans.append(self.program_memory.get_best_item())
            plan = choice(plans)
            plan.current_phase_index = traci.trafficlights.getPhase(self.id)
            self.program_current.fill_with(plan, self.lanes)
            if self.program_memory.is_current_item_worst(self.program_current.id):
                item = deepcopy(self.program_default)
                item.id = str(int(self.get_max_plan_id()) + 1)
                # If we create a new plan, we add it to ant plan's
                self.add_plan(item)
                self.program_current = deepcopy(item)
                # If the number of plans are overload, we remove it from ant plan's
                # taking it by plan id
                plan_id = self.program_memory.add_item(item)
                if plan_id:
                    #print "Drop. Plan: ",plan_id
                    self.remove_plan(plan_id)
            # Update the various attributes for the model
            self.program_memory.update_transition_probability(self.program_current.id, self.last_state, self.last_action, new_state)
            self.program_memory.update_reward_model(self.program_current.id, self.last_state, self.last_action, reward)
            self.program_memory.update_transition_estimate(self.last_state, self.last_action)
            #self.program_memory.debug()
            #"""

        if new_action is not None:
            cycle_length = sum(phase.duration for phase in self.program_current.phases
                           if phase.has_green)
            (inc_idx, increment) = new_action

            # Increments the current phase
            self.program_current.phases[inc_idx].duration += increment

            #"""Decrement the other phases to equate the cycle time
            decr_phases = [(idx, phase) for idx, phase in enumerate(self.program_current.phases)
                            if phase.has_green and idx != inc_idx]
            decr_phases_len = len(decr_phases)
            decrement = int(increment / decr_phases_len)
            for (decr_idx, phase) in decr_phases:
                self.program_current.phases[decr_idx].duration -= decrement

            # Compensates errors on float->int conversion
            phs = [phase.duration for phase in self.program_current.phases
                           if phase.has_green]
            error = cycle_length - sum(phs)
            if error > 0:
                # The cycle is higher than expected
                if self.program_current.phases[inc_idx].has_max_duration:
                    for (decr_idx, phase) in decr_phases:
                        self.program_current.phases[decr_idx].duration += int(error/decr_phases_len)
                else:
                    self.program_current.phases[inc_idx].duration += error
            elif error < 0:
                # The cycle is lower than expected
                if self.program_current.phases[inc_idx].has_min_duration:
                    for (decr_idx, phase) in decr_phases:
                        self.program_current.phases[decr_idx].duration += int(error/decr_phases_len)
                else:
                    self.program_current.phases[inc_idx].duration += error
            # decrement ends here"""

        self.last_state = new_state
        self.last_action = new_action
        self.last_step_action = self.memory_window[-1]
        # Learning Agent act()
        print "SA-CD actuate"

    def update_program_current(self, program):
        super(LearningAntAgent, self).update_program_current(program)

    def __repr__(self):
        #data = [super(LearningAntAgent, self).__repr__()]
        data = []
        #data.append("\nLearningAntAgent Class")
        data.append("program memory="+str(self.program_memory.debug()))
        return "\n".join(data)

    def __str__(self):
        #data = [super(LearningAntAgent, self).__repr__()]
        data = []
        #data.append("\nLearningAntAgent Class")
        data.append("program memory="+str(self.program_memory.debug()))
        return "\n".join(data)
