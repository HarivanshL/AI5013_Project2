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


from util import manhattanDistance
from game import Directions
import random, util

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


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        food_list = newFood.asList()
        if not food_list:
            return float("inf")
        # distance from the closest food to pacman
        min_food_distance = min(manhattanDistance(newPos, food) for food in food_list)

        # distance from the closest ghost to pacman
        min_ghost_distance = min(manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates)

        # if pacman is too close to dangereous ghost, give penalty
        if min_ghost_distance < 3 and newScaredTimes == 0:
            return -float("inf")
        
        newCapsules = successorGameState.getCapsules()
        # distance from the closest capsule to pacman
        min_capsule_distance = 0
        if newCapsules:
            min_capsule_distance = min(manhattanDistance(newPos, capsule) for capsule in newCapsules)

        return (successorGameState.getScore()
                - min_food_distance * 2 # smaller distance from food is better
                + min_ghost_distance * (0.5 if newScaredTimes == 0 else 2) # larger distance from ghost is better. if ghost is scared add more score
                - len(food_list) * 20 # make pacman eat food actively when there is a few food left
                - min_capsule_distance * 5 # smaller distance from capsule is better
                )

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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        def minimax(gameState: GameState, depth, agentIndex):
            # if depth reached to max depth or game is end return the score
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            # Pacman's turn
            if agentIndex == 0:
                value = -float("inf")
                best_action = None
                # search all possible actions
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    next_value = minimax(successor, depth, agentIndex+1) # move to ghosts' turn
                    if next_value > value:
                        value = next_value
                        best_action = action

                # if depth is 0 return the best action
                if depth == 0:
                    return best_action if best_action is not None else Directions.STOP

                return value
            # Ghosts' turn
            else:
                value = float("inf")
                next_agent = agentIndex + 1

                # if it is the last ghost, next will be pacman's turn. increase the depth
                if next_agent == gameState.getNumAgents():
                    next_agent = 0
                    depth += 1

                # search all possible actions
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = min(value, minimax(successor, depth, next_agent)) # move to next agnet's tunr (it could be ghost or pacman)
                    
                return value
        
        return minimax(gameState, 0, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(gameState: GameState, depth, alpha, beta, agentIndex):
            # if depth reached to max depth or game is end return the score
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            # Pacman's turn
            if agentIndex == 0:
                value = -float("inf")
                best_action = None
                # search all possible actions
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    next_value = alphabeta(successor, depth, alpha, beta, agentIndex+1) # move to ghosts' turn
                    if next_value > value:
                        value = next_value
                        best_action = action
                    
                    # if value is bigger than beta, do pruning
                    if value > beta:
                        break
                    # else change alpha to maximum value
                    alpha = max(alpha, value)

                # if depth is 0 return the best action
                if depth == 0:
                    return best_action if best_action is not None else Directions.STOP

                return value
            
            # Ghosts' turn
            else:
                value = float("inf")
                next_agent = agentIndex + 1

                # if it is the last ghost, next will be pacman's turn. increase the depth
                if next_agent == gameState.getNumAgents():
                    next_agent = 0
                    depth += 1

                # search all possible actions
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = min(value, alphabeta(successor, depth, alpha, beta, next_agent)) # move to next agnet's tunr (it could be ghost or pacman)
                    
                    # if value is less than alpha, do pruning
                    if value < alpha:
                        break
                    # else change beta to minimum value
                    beta = min(beta, value)
                    
                return value
            
        return alphabeta(gameState, 0, -float("inf"), float("inf"), 0)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(gameState: GameState, depth, agentIndex):
            # if depth reached to max depth or game is end return the score
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            # Pacman's turn
            if agentIndex == 0:
                value = -float("inf")
                best_action = None
                # search all possible actions
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    next_value = expectimax(successor, depth, agentIndex+1) # move to ghosts' turn
                    if next_value > value:
                        value = next_value
                        best_action = action

                # if depth is 0 return the best action
                if depth == 0:
                    return best_action if best_action is not None else Directions.STOP

                return value
            
            # Ghosts' turn
            else:
                value = float("inf")
                next_agent = agentIndex + 1

                # if it is the last ghost, next will be pacman's turn. increase the depth
                if next_agent == gameState.getNumAgents():
                    next_agent = 0
                    depth += 1

                total = 0 # total value at next actions
                action_count = 0 # the num of actions
                
                # search all possible actions
                for action in gameState.getLegalActions(agentIndex):
                    successor = gameState.generateSuccessor(agentIndex, action)
                    total += expectimax(successor, depth, next_agent)
                    action_count += 1

                value = total / action_count # choosing uniformly at random from their legal moves
                return value
        
        return expectimax(gameState, 0, 0)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Evaluation function for Pacman’s current state.

    This function scores a game state based on several factors:
        - Encourages eating food by penalizing distance to the closest food and reducing score as food decreases.
        - Avoids dangerous ghosts by applying a heavy penalty if too close.
        - Accounts for scared ghosts, encouraging chasing them when beneficial.
        - Encourages collecting capsules by penalizing distance to the nearest capsule.
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    #Evaluate states instead of actions

    "*** YOUR CODE HERE ***"
    #Current score
    current_score = currentGameState.getScore()

    #Pacman state
    pacman_pos= currentGameState.getPacmanPosition()

    #Food calculations

    #Grabs all the food in the current game
    current_food = currentGameState.getFood()
    food_list = current_food.asList()
    if not food_list:
        return float("inf")
    
    #Calculates the distances for all the foods
    food_distances = [manhattanDistance(pacman_pos, food) for food in food_list]
    #Picks the closest food distance
    closest_food = min(food_distances)

    #Ghost calculations

    #Grabs the current ghost states for the ghosts in the game
    current_ghost_states = currentGameState.getGhostStates()

    #Scared ghosts
    currentScaredGhosts = [ghostState for ghostState in current_ghost_states if ghostState.scaredTimer]
    #Not scared ghosts
    currentNonScaredGhosts = [ghostState for ghostState in current_ghost_states if not ghostState.scaredTimer]


    #Sets scared distance to 0 so it is not factored in if there are no scared ghosts
    min_scared_distance =0
    # distance from the closest scared ghost to pacman distance
    if len(currentScaredGhosts) >0:
        #Picks the closest scared ghost
        min_scared_distance = min(manhattanDistance(pacman_pos, ghost.getPosition()) for ghost in currentScaredGhosts)
    

    #Sets not scared distance to 0 so it is not factored in if there are scared ghosts
    min_not_scared_distance = 0
    if len(currentNonScaredGhosts) >0:
        #Picks the closest not scared ghost distance
        min_not_scared_distance = min(manhattanDistance(pacman_pos, ghost.getPosition()) for ghost in currentNonScaredGhosts)

    # if pacman is too close to dangereous ghost, give penalty
    if min_scared_distance < 3 and currentScaredGhosts == 0:
        return -float("inf")
    
    #Grabs the capsules for the current game state
    newCapsules = currentGameState.getCapsules()
    # distance from the closest capsule to pacman

    #Sets the capsule distance to 0 so it is not factored in if there are no longer any capsules
    min_capsule_distance = 0
    
    #Finds the closest capsule distance
    if newCapsules:
        min_capsule_distance = min(manhattanDistance(pacman_pos, capsule) for capsule in newCapsules)

    return (current_score
            - closest_food * 2 # smaller distance from food is better
            - min_scared_distance * 2 #smaller penalty for ghosts that are scared
            - min_not_scared_distance * 4 # larger distance from ghost is better. if ghost is scared add more score
            - len(food_list) * 20 # make pacman eat food actively when there is a few food left
            - min_capsule_distance * 5 # smaller distance from capsule is better
            )

# Abbreviation
better = betterEvaluationFunction
