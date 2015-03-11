#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""Classes implementing the general q-learning algorithm.
"""

from random import random, choice

class QTable(object):
    """A table of state-action qvalues, with the learning logic.
    """

    DEFAULT_QVALUE = 0

    def __init__(self, states, actions, learning_rate,
                 discount_factor, default_qvalue=None):
        """Initializes the table and checks parameter restrictions.

        In case a parameter restriction isn't met, a ValueError is raised:
          - 0 <= learning_rate <= 1
          - 0 <= discount_factor < 1
        """
        # Verifies and initializes learning_rate
        if 0 <= learning_rate and learning_rate <= 1:
            self.__learning_rate = learning_rate
        else:
            raise ValueError("Invalid learning rate of %d not contained in [0,1]."
                             % learning_rate)

        # Verifies and initializes discount_factor
        if 0 <= discount_factor and discount_factor < 1:
            self.__discount_factor = discount_factor
        else:
            raise ValueError("Invalid discount factor of %d not contained in [0,1)."
                             % discount_factor)

        # Initializes the table of qvalues
        if default_qvalue is None:
            default_qvalue = self.DEFAULT_QVALUE

        self.__table = dict( (s, dict((a, default_qvalue) for a in actions))
                             for s in states )
        self.__states = tuple(states)
        self.__actions = tuple(actions)
        """ Force the random choice to be biased. 
            
        bias > 0 increment time biased
        bias < 0 decrement time biased
        bias = 0 not biased
        """
        self.__bias = (0, 0)

    def __getitem__(self, pair):
        return self.__table[pair[0]][pair[1]]

    def subtable(self, state):
        """Obtain the part of the table describing the given state.

        The result is a dict from actions to their q-value
        on the given state.
        """
        return self.__table[state]

    @property
    def table(self):
        """Return the qtable."""
        return self.__table

    @table.setter
    def table(self, table):
        """Update the qtable."""
        self.__table = table

    @property
    def bias(self):
        """Return the bias value."""
        return self.__bias

    @bias.setter
    def bias(self, value):
        """ Update the biased behavior value. """
        self.__bias = value

    @property
    def states(self):
        """Return a tuple with all possible states."""
        return self.__states
    @states.setter
    def states(self, states):
        """Update a tuple with all possible states."""
        self.__states = states

    @property
    def actions(self):
        """Return a tuple with all possible actions."""
        return self.__actions

    @actions.setter
    def actions(self, actions):
        """Update a tuple with all possible actions."""
        self.__actions = actions

    def observe(self, state, action, new_state, reward):
        """Update q-values according to the observed behavior."""
        max_future = max( self[new_state, new_action]
                          for new_action in self.__actions )
        old_val = self[state, action]

        change = reward + (self.__discount_factor * max_future) - old_val
        self.__table[state][action] = old_val + (self.__learning_rate * change)

    def act(self, state):
        """Return the recommended action for this state.

        The choice of action may include random exploration.
        """
        raise NotImplementedError("The basic QTable has no policy.")

    def __repr__(self):
        data = []
        data.append("\nQTable")
        data.append("learning rate="+str(self.__learning_rate))
        data.append("discount factor="+str(self.__discount_factor))
        data.append("actions="+str(";".join([str(a) for a in self.__actions])))
        data.append("states="+str(";".join([str(s) for s in self.__states])))
        data.append("table="+str(self.table))
        return "\n".join(data)

    def __str__(self):
        data = []
        data.append("\nQTable")
        data.append("learning rate="+str(self.__learning_rate))
        data.append("discount factor="+str(self.__discount_factor))
        data.append("actions="+str(";".join([str(a) for a in self.__actions])))
        data.append("states="+str(";".join([str(s) for s in self.__states])))
        data.append("table="+str(self.table))
        return "\n".join(data)

class EpsilonGreedyQTable(QTable):
    """QTable with epsilon-greedy strategy.

    With a given probability, called curiosity, chooses a random action.
    Otherwise, does a greedy choice (action with greatest qvalue, in case
    of a tie randomly picks one of the best).

    The curiosity decays a given percentage after each random choice.
    """

    DEFAULT_CURIOSITY_DECAY = 0

    def __init__(self, states, actions, learning_rate, discount_factor, curiosity,
                 curiosity_decay=None, default_qvalue=None):
        """Initializes the table and checks for parameter restrictions.

        In case a parameter restriction isn't met, a ValueError is raised:
          - all restrictions from QTable apply
          - 0 <= curiosity_decay < 1
        """
        super(EpsilonGreedyQTable, self).__init__(states, actions, learning_rate,
                                                  discount_factor, default_qvalue)

        self.__curiosity = curiosity
        self.__first_random_choice = True
        self.__last_random_choice = None

        # Verify and initialize curiosity_decay
        if curiosity_decay is None:
            curiosity_decay = self.DEFAULT_CURIOSITY_DECAY

        if 0 <= curiosity_decay and curiosity_decay < 1:
            self.__curiosity_factor = 1 - curiosity_decay
        else:
            raise ValueError("Invalid curiosity decay %d not contained in [0,1)."
                             % curiosity_decay)

    def act(self, state):
        if random() < self.__curiosity:
            print "RANDOM CHOICE="+str(self.__curiosity)
            return self.random_choice(state)
        else:
            print "GREEDY CHOICE="+str(self.__curiosity)
            return self.greedy_choice(state)

    def random_choice(self, state):
        """Makes a uniformly random choice between all actions for this state."""
        self.__curiosity *= self.__curiosity_factor

        if self.__last_random_choice is None:
            choices = self.actions
        else:
            choices = self.actions + [self.__last_random_choice]

        if self.__curiosity <= 0.5: # Wait for the half time of e-greedy search strategy
            print "BIAS="+str(self.bias[1])
            # Take random choices or positive/negative only actions
            if self.bias[1] > 0:
                choices = filter(lambda (p,a): a>=0 and p==self.bias[0], filter(lambda c: c is not None, choices))
                choices += tuple([None])
            elif self.bias[1] < 0:
                choices = filter(lambda (p,a): a<0 and p==self.bias[0], filter(lambda c: c is not None, choices))
                choices += tuple([None])
        return choice(choices)

    def greedy_choice(self, state):
        """Makes a uniformly random choice between the actions with best q-value.

        The analyzed q-values are those related with actions on this state."""
        best_qval = max(self.subtable(state).values())
 
        best_actions = [act for (act, qval) in self.subtable(state).items()
                        if qval == best_qval]
 
        return choice(best_actions)

    def __repr__(self):
        data = [super(EpsilonGreedyQTable, self).__repr__()]
        data.append("\nEpsilon Greedy")
        data.append("curiosity="+str(self.__curiosity))
        data.append("curiosity factor="+str(self.__curiosity_factor))
        data.append("first random choice="+str(self.__first_random_choice))
        data.append("last random choice="+str(self.__last_random_choice))
        return "\n".join(data)

    def __str__(self):
        data = [super(EpsilonGreedyQTable, self).__str__()]
        data.append("\nEpsilon Greedy")
        data.append("curiosity="+str(self.__curiosity))
        data.append("curiosity factor="+str(self.__curiosity_factor))
        data.append("first random choice="+str(self.__first_random_choice))
        data.append("last random choice="+str(self.__last_random_choice))
        return "\n".join(data)

class EpsilonFirstQTable(EpsilonGreedyQTable):
    """QTable with epsilon-greedy strategy limited to the first N actions."""

    def __init__(self, states, actions, learning_rate, discount_factor, curiosity,
                 exploration_period, curiosity_decay=None, default_qvalue=None):
        super(EpsilonFirstQTable, self).__init__(states, actions, learning_rate,
                                                 discount_factor, curiosity,
                                                 curiosity_decay, default_qvalue)
        if exploration_period > 0:
            self.__remaining_exploration = exploration_period
        else:
            raise ValueError("Invalid non-positive exploration period %d."
                             % exploration_period)

    def act(self, state):
        if self.__remaining_exploration > 0:
            self.__remaining_exploration -= 1
            return super(EpsilonFirstQTable, self).act(state)
        else:
            return self.greedy_choice(state)

    def __repr__(self):
        data = [super(EpsilonFirstQTable, self).__repr__()]
        data.append("\nEpsilon First")
        data.append("remaining exploration="+str(self.__remaining_exploration))
        return "\n".join(data)

    def __str__(self):
        data = [super(EpsilonFirstQTable, self).__str__()]
        data.append("\nEpsilon First")
        data.append("remaining exploration="+str(self.__remaining_exploration))
        return "\n".join(data)

class Learning(EpsilonGreedyQTable):
    def __init__(self, states, actions, learning_rate, discount_factor, curiosity,
                 exploration_period, curiosity_decay=None, qvalue=None):
        super(Learning, self).__init__(states, actions, learning_rate, 
                                       discount_factor, curiosity, curiosity_decay, 
                                       qvalue)
        #print "Learning"

    def __repr__(self):
        data = [super(Learning, self).__repr__()]
        data.append("\nLearning Class")
        return "\n".join(data)

    def __str__(self):
        data = [super(Learning, self).__str__()]
        data.append("\nLearning Class")
        return "\n".join(data)
