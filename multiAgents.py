# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from asyncio import SelectorEventLoop
from multiprocessing import managers
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        distToFood = 9999999
        ghostPos = successorGameState.getGhostPositions()

         # use manhattan distance to find food most efficiently
        for food in foodList:
            if distToFood > manhattanDistance(food, newPos):
                distToFood = manhattanDistance(food, newPos)

        # avoid ghosts
        for pos in ghostPos: 
            distance = manhattanDistance(pos, newPos)

            # ghost is too close - you lose
            if distance < 1: 
                return -9999999

        return successorGameState.getScore() + (1.0 / distToFood) # using reciprocal of dist to food

def scoreEvaluationFunction(currentGameState):
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # minimax:
        # max: state's under agent's control - pacman
        # min: state's under opponent's control - ghost
        # find the best achievable utility against a ratioal adversary
        # (minimize the possible loss for worst case scenario resulting from opponent's possible moves)

        # [score, action]
        minimaxOutput = self.minimax(gameState, 0, 0)
        return minimaxOutput[1]

    def minimax(self, gameState, agentIndex, depth):

        # if win or lose or reached maximum depth, return [score, action]
        if gameState.isWin() or gameState.isLose() or (depth == self.depth):
            return gameState.getScore(), 0

        else:
            # if agent is pacman, return max layer
            if agentIndex == 0:
                return self.maxVal(gameState, agentIndex, depth)

            # if agent is ghost, return min layer (min layers represent ghosts)
            else:
                return self.minVal(gameState, agentIndex, depth)

    def maxVal(self, gameState, agentIndex, depth):

        maxVal = -9999999
        maxAction = 0
        agentActions = gameState.getLegalActions(agentIndex)

        for action in agentActions:

            # create successor in tree
            nextAgent = gameState.generateSuccessor(agentIndex, action)
            nextIndex = agentIndex + 1
            nextDepth = depth

            # if successor is pacman, update index and depth appropriately 
            if nextIndex == gameState.getNumAgents():
                nextIndex = 0
                nextDepth += 1

            # calls minimax() and stores score
            minimaxVal = self.minimax(nextAgent, nextIndex, nextDepth)[0]

            # finds and stores max score and stores action
            if maxVal < minimaxVal:
                maxVal = minimaxVal
                maxAction = action

        # [score, action]
        return maxVal, maxAction

    def minVal(self, gameState, agentIndex, depth):

        minVal = 9999999
        minAction = 0
        agentActions = gameState.getLegalActions(agentIndex)

        # create successor in tree
        for action in agentActions:
            nextAgent = gameState.generateSuccessor(agentIndex, action)
            nextIndex = agentIndex + 1
            nextDepth = depth

            # if successor is pacman, update index and depth appropriately
            if nextIndex == gameState.getNumAgents():
                nextIndex = 0
                nextDepth += 1

            # calls minimax() and stores score
            minimaxVal = self.minimax(nextAgent, nextIndex, nextDepth)[0]
            
            # finds and stores min score and stores action
            if minVal > minimaxVal:
                minVal = minimaxVal
                minAction = action

        # [score, action]
        return minVal, minAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        # alpha-beta pruning:
        # seeks to decrease the number of nodes evaluated by minimax
        # stops evaluating a move when at least one better move has been found already

        # [action, score]
        abOutput = self.alphabeta(gameState, 0, 0, -9999999, 9999999)
        return abOutput[0]

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):

        # if win or lose or reached maximum depth, return [action, score]
        if gameState.isWin() or gameState.isLose() or (depth == self.depth):
            return 0, gameState.getScore()

        else:
            # if agent is pacman, return max layer
            if agentIndex == 0:
                return self.maxVal(gameState, agentIndex, depth, alpha, beta)

            # if agent is ghost, return min layer (min layers represent ghosts)
            else:
                return self.minVal(gameState, agentIndex, depth, alpha, beta)

    def maxVal(self, gameState, agentIndex, depth, alpha, beta):

        maxVal = -9999999
        maxAction = 0
        agentActions = gameState.getLegalActions(agentIndex)

        for action in agentActions:

            # create successor in tree
            nextAgent = gameState.generateSuccessor(agentIndex, action)
            nextIndex = agentIndex + 1
            nextDepth = depth

            # if successor is pacman, update index and depth appropriately 
            if nextIndex == gameState.getNumAgents():
                nextIndex = 0
                nextDepth += 1

            # calls alphabeta() and stores score in [action, score] pair
            abVal = self.alphabeta(nextAgent, nextIndex, nextDepth, alpha, beta)[1]

            # finds and stores max score and stores action
            if maxVal < abVal:
                maxVal = abVal
                maxAction = action

            # updates alpha to maximum value
            if maxVal > alpha:
                alpha = maxVal

            # if beta is smaller, no need to evaluate other nodes
            if maxVal > beta:
                return maxAction, maxVal

        # [action, score]
        return maxAction, maxVal

    def minVal(self, gameState, agentIndex, depth, alpha, beta):

        minVal = 9999999
        minAction = 0
        agentActions = gameState.getLegalActions(agentIndex)

        for action in agentActions:

            # create successor in tree
            nextAgent = gameState.generateSuccessor(agentIndex, action)
            nextIndex = agentIndex + 1
            nextDepth = depth

            # if successor is pacman, update index and depth appropriately 
            if nextIndex == gameState.getNumAgents():
                nextIndex = 0
                nextDepth += 1

            # calls alphabeta() and stores score in [action, score] pair
            abVal = self.alphabeta(nextAgent, nextIndex, nextDepth, alpha, beta)[1]

            # finds and stores min score and stores action
            if minVal > abVal:
                minVal = abVal
                minAction = action

            # updates beta to minimum value
            if minVal < beta:
                beta = minVal

            # if alpha is larger, no need to evaluate other nodes
            if minVal < alpha:
                return minAction, minVal

        # [action, score]
        return minAction, minVal
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        # expectimax:
        # compute the score for average opponent move and our best move 
        # max nodes same as minimax
        # chance nodes compute weighted average

        # [action, score]
        expectiOutput = self.expectimax(gameState, 0, 0)
        return expectiOutput[0]

    def expectimax(self, gameState, agentIndex, depth):

        # if win or lose or reached maximum depth, return [action, score]
        if gameState.isWin() or gameState.isLose() or (depth == self.depth):
            return 0, gameState.getScore()

        else:
            # if agent is pacman, return max layer
            if agentIndex == 0:
                return self.maxVal(gameState, agentIndex, depth)

            # if agent is ghost, return expectation value
            else:
                return self.expectedVal(gameState, agentIndex, depth)

    def maxVal(self, gameState, agentIndex, depth):

        maxVal = -9999999
        maxAction = 0
        agentActions = gameState.getLegalActions(agentIndex)

        for action in agentActions:

            # create successor in tree
            nextAgent = gameState.generateSuccessor(agentIndex, action)
            nextIndex = agentIndex + 1
            nextDepth = depth

            # if successor is pacman, update index and depth appropriately 
            if nextIndex == gameState.getNumAgents():
                nextIndex = 0
                nextDepth += 1

            # calls expectimax() and stores score in [action, score] pair
            expectiVal = self.expectimax(nextAgent, nextIndex, nextDepth)[1]

            # finds and stores max score and stores action
            if maxVal < expectiVal:
                maxVal = expectiVal
                maxAction = action

        # [action, score]
        return maxAction, maxVal

    def expectedVal(self, gameState, agentIndex, depth):

        expVal = 0
        expAction = 0
        agentActions = gameState.getLegalActions(agentIndex)


        for action in agentActions:
    
            # probability of action is 1 in total actions
            prob = 1.0 / len(agentActions)

            # create successor in tree
            nextAgent = gameState.generateSuccessor(agentIndex, action)
            nextIndex = agentIndex + 1
            nextDepth = depth

            # if successor is pacman, update index and depth appropriately 
            if nextIndex == gameState.getNumAgents():
                nextIndex = 0
                nextDepth += 1

            # calls expectimax() and stores score in [action, score] pair
            expectiVal = self.expectimax(nextAgent, nextIndex, nextDepth)[1]
            
            # v += p * em-value(s)
            expVal += prob * expectiVal

        # [action, score]
        return expAction, expVal


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 

        pacmanPos stores pacman's position, score stores the total score

        foodDists is an array to hold the distances between pacman and a piece of food

        foodList stores a list of available food

        This function prioritizes getting food as quickly as possible, regardless of ghosts. Manhattan distance between pacman and 
        each piece of food is stored. This function follows the highest score while getting food efficiently.

        I was getting an error when max() was getting passed an empty array, so I added the "if not foodDists" statement to make sure
        foodDists didn't remain empty
    """
    "*** YOUR CODE HERE ***"
    
    pacmanPos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
 
    foodDists = []

    food = currentGameState.getFood()
    foodList = food.asList()

    for foods in foodList:
        distToFood = manhattanDistance(pacmanPos, foods)
        negativeFood = -1.0 * distToFood
        foodDists.append(negativeFood)

    if not foodDists:
        foodDists.append(0)

    return score + max(foodDists)

# Abbreviation
better = betterEvaluationFunction
