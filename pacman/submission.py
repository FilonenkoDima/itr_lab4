from util import manhattanDistance
from game import Directions
import random, util
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState: GameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
    You may reference the pseudocode for Alpha-Beta pruning here:
    en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 36 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state, depth = '1'):
        self.index = 0
        self.depth = int(depth)                                                                       # used to depth limit the expectimax search, default depth is 2

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write your algorithm Algorithm instead of returning Directions.STOP
        def max_turn(state, depth):                                                                   # starting from the maxturn. This is the pacman's turn
            legal = state.getLegalPacmanActions()                                                     # consider all the legal moves
            if len(legal) == 0 or state.isWin() or state.isLose() or depth == self.depth:             # if no legal moves or Win or Lose or the depth is reached then return the evaluation. When the depth is reached make the node as a leaf node.
                return (self.myEvaluation(state), None)                                               # get the evaluation of the state

            temp = -(float("inf"))
            final_action = None

            for action in legal:
                successor_value = exp_turn(state.generatePacmanSuccessor(action), 1, depth)[0]        #for all the possible successor states apply the exp turn to estimate the chance and get the evaluation of each successor state.

                if temp < successor_value:
                    temp, final_action = successor_value, action                                      #find the successor with the maximum score

            return (temp, final_action)                                                               # take action to go to the best successor.

        def exp_turn(state, agent, depth):                                                            # exp turn to estimate the value of possible states based on moves of pacman and ghosts.

            legal = state.getLegalActions(agent)                                                      # get legal actions
            if len(legal) == 0:
                return (self.myEvaluation(state), None)                                               #if no legal action possible just return the value of state

            temp = 0
            final_action = None

            for action in legal:
                if agent == (state.getNumAgents() - 1):                                               # get the total number agents
                    successor_value = max_turn(state.generateSuccessor(agent, action), depth + 1)[0]  # pacman's turn if the second ghost has called exp - turn
                else:
                    successor_value = exp_turn(state.generateSuccessor(agent, action), agent + 1, depth)[0]  # call exp-turn for the second ghost

                p = successor_value/len(legal)                                                        # The probability of getting to the state
                temp = temp + p                                                                       # The total chance of the state

            return (temp, final_action)                                                               # return the chance and best action for that state.

        return max_turn(state, 0)[1]                                                                  # start estimating from the pacmans current state.


    def myEvaluation(self, state):


        newPos = state.getPacmanPosition()                                                           # get the current position of the pacman
        Food = state.getFood()                                                                       # get the grid of the food


        foodList = [self.manhattanDistance(newPos, f) for f in Food.asList()]                        # calculate the manhattan distance between pellets and pacman

        foodScore = 0

        for f in foodList:
            foodScore += 1.0/float(f)                                                                # pellets which are closer get higher score then pellets which are farther away


        newGhostStates = state.getGhostStates()                                                     # Getting the ghost states Note: Only used to get the the scared timer so that the pacman dosent run away from the ghosts when they are scared.
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]                  # scared times for making the pacman eat scared ghosts

        GhostPositions = state.getGhostPositions()                                                  # getting the positions of the ghosts.
        GhostDistance = [self.manhattanDistance(newPos, g) for g in GhostPositions]                 # calculate the manhattan distance between ghost and pacman.
        GhostDistance.sort()                                                                        # sort the distance to the ghosts

        GhostScore = 0
        if min(GhostDistance) == 0:                                                                 # if the ghost is very near give high penalty
            GhostScore = 100000000
        else:
            for g in GhostDistance:                                                                # add to the ghost score
                if g < 3 and g != 0:                                                               # if the ghost if more than distance of 3 ignore the ghost (if the ghost is very near handled earlier, handled divide be zero error)
                    GhostScore + 1.0/g                                                             # nearer the ghost more is the penalty

        scaredtimeSum = sum(newScaredTimes)                                                        # take sum of scared times for both ghosts

        return state.getScore() + foodScore - 28 * GhostScore + 1.2 * scaredtimeSum                # final score

    def manhattanDistance(self, x, y):
        return abs(x[0] - y[0]) + abs(x[1] - y[1])
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState: GameState) -> float:
  """
    Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.
  """

  # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
  raise Exception("Not implemented yet")
  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
