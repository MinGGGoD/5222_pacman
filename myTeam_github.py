# myTeam.py
# ---------------
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


# myTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from ast import Raise
from typing import Dict, List, Tuple

from numpy import true_divide
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os
from capture import GameState, noisyDistance
from game import Directions, Actions, AgentState, Agent
from util import nearestPoint
import sys,os

# the folder of current file.
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

from lib_piglet.utils.pddl_solver import pddl_solver
from lib_piglet.domains.pddl import pddl_state
from lib_piglet.utils.pddl_parser import Action

CLOSE_DISTANCE = 4
MEDIUM_DISTANCE = 15
LONG_DISTANCE = 25


#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
                             first = 'MixedAgent', second = 'MixedAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########                                       

class MixedAgent(CaptureAgent):
    """
    MixedAgent implements a hybrid of classical planning and approximate Q-learning
    for Pacman Capture the Flag.  This implementation follows the design of
    Abhinav Creed et al.: classical planning via PDDL for selecting high-level
    goals (e.g. eat a capsule or food) and approximate Q-learning for low-level
    action selection.  When high-level planning is not applicable or the agent
    becomes stuck, the agent falls back to pure approximate Q-learning.
    """
    # Shared weights for approximate Q-learning across agents.  If no weight
    # file is found, the following defaults are used.  Additional weight
    # dictionaries from the original implementation are retained for
    # compatibility but are not utilised by the approximate planner.
    QLWeights: dict = {
        "offensiveWeights": {
            'closest-food': -1,
            'bias': 1,
            '#-of-ghosts-1-step-away': -100,
            'successorScore': 100,
            'chance-return-food': 10,
        },
        "defensiveWeights": {
            'numInvaders': -1000,
            'onDefense': 100,
            'teamDistance': 2,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -2,
        },
        "escapeWeights": {
            'onDefense': 1000,
            'enemyDistance': 30,
            'stop': -100,
            'distanceToHome': -20,
        },
        # Weights for approximate Q-learning (Approach Five).  These weights
        # correspond to the features described in the blog: closest-food,
        # bias, number of ghosts within one step, and whether food is eaten.
        "approximateWeights": {
            'closest-food': -3.0,
            'bias': 1.0,
            '#-of-ghosts-1-step-away': -20.0,
            'eats-food': 5.0,
        },
    }
    # Path to weight file for saving/loading approximate weights
    QLWeightsFile: str = BASE_FOLDER + '/QLWeightsMyTeam.txt'

    # Shared dictionary storing current high-level action names per agent
    CURRENT_ACTION: Dict[int, str] = {}

    def registerInitialState(self, gameState: GameState):
        """
        Initialise the agent for a new game.  This sets up the PDDL solver
        using our custom domain, builds adjacency maps for planning, and
        loads weights for approximate Q-learning if available.  It also
        initialises Q-learning parameters such as epsilon, alpha and discount.
        """
        # Initialise PDDL solvers for both pacman and ghost roles.  The
        # classical planner uses different domain files depending on
        # whether the agent is currently a pacman (offensive role) or a
        # ghost (defensive role).  Loading both upfront avoids repeated
        # file access during play.  Note that the ghost domain is kept
        # extremely simple and may not always be used for planning.
        self.pacman_solver = pddl_solver(BASE_FOLDER + '/pacman_domain.pddl')
        self.ghost_solver = pddl_solver(BASE_FOLDER + '/ghost_domain.pddl')
        # Current solver used for plan execution.  This is updated when
        # generating a new high-level plan.
        self.current_solver = self.pacman_solver
        # High-level plan and associated goal tracking
        self.highLevelPlan: List[Tuple[Action, pddl_state]] = []
        self.currentNegativeGoalStates: List[Tuple[str]] = []
        self.currentPositiveGoalStates: List[Tuple[str]] = []
        self.currentActionIndex: int = 0

        self.startPosition = gameState.getAgentPosition(self.index)
        # Register initial state in CaptureAgent (for distance calculators etc.)
        CaptureAgent.registerInitialState(self, gameState)

        # Build adjacency mapping of maze once per game.  Each location is
        # represented as a string "loc_x_y" and stored in self.pddl_locs.
        self.buildAdjacency(gameState)

        # Initialise Q-learning parameters.  Training mode will update
        # approximate weights and enable exploration via epsilon.
        self.trainning: bool = False  # Set to True when training weights
        self.epsilon: float = 0.1     # Exploration probability
        self.alpha: float = 0.2       # Learning rate
        self.discountRate: float = 0.9

        # Load approximate weights if available
        self.approxWeights: dict = MixedAgent.QLWeights.get('approximateWeights', {}).copy()
        if os.path.exists(MixedAgent.QLWeightsFile):
            try:
                with open(MixedAgent.QLWeightsFile, 'r') as f:
                    loaded = eval(f.read())
                    if 'approximateWeights' in loaded:
                        # Update only existing keys to avoid stray entries
                        for feat, val in loaded['approximateWeights'].items():
                            self.approxWeights[feat] = float(val)
                        print(f"Loaded approximate weights: {self.approxWeights}")
            except Exception as e:
                print(f"Failed to load QL weights: {e}")

        # Initialise last action to STOP and high-level action
        self.lastAction: str = Directions.STOP
        MixedAgent.CURRENT_ACTION[self.index] = ''
        # For stuck detection: keep last few actions
        self.recentActions: List[str] = []

    def buildAdjacency(self, gameState: GameState) -> None:
        """
        Pre-compute the adjacency relationships between all legal positions
        (positions without walls).  Each location is mapped to a string
        identifier loc_x_y for PDDL objects.  The adjacency is stored in
        self.adjacent for building the initial state and objects for the
        classical planner.
        """
        self.adjacent: Dict[str, List[str]] = {}
        self.coord_to_loc: Dict[Tuple[int, int], str] = {}
        self.pddl_locs: List[str] = []
        walls = gameState.getWalls()
        width, height = walls.width, walls.height
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    loc_str = f"loc_{x}_{y}"
                    self.coord_to_loc[(x, y)] = loc_str
                    self.pddl_locs.append(loc_str)
        # Build adjacency for each location
        for (x, y), loc_str in self.coord_to_loc.items():
            neighbours = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.coord_to_loc:
                    neighbours.append(self.coord_to_loc[(nx, ny)])
            self.adjacent[loc_str] = neighbours

    def final(self, gameState: GameState):
        """
        Save the approximate Q-learning weights at the end of a game when
        training is enabled.  Weights are stored under the 'approximateWeights'
        key so that they can be loaded for subsequent games.
        """
        if self.trainning:
            # Update shared QLWeights dictionary
            MixedAgent.QLWeights['approximateWeights'] = self.approxWeights.copy()
            with open(MixedAgent.QLWeightsFile, 'w') as f:
                f.write(str(MixedAgent.QLWeights))
            print("Saved approximate weights to file.")

    def build_pddl_state(self, gameState: GameState) -> Tuple[List[Tuple[str, str]], List[Tuple]]:
        """
        Construct objects and initial state for the PDDL planner from the
        current gameState.  The objects include the pacman, ghosts and all
        legal locations.  The initial state encodes the positions of pacman,
        ghosts, food pellets and capsules, and whether pacman is carrying
        food.  The adjacency relations between locations are also provided
        explicitly as part of the initial state for the planner.

        Returns a tuple of (objects, initState) where objects is a list of
        tuples (objectName, typeName) and initState is a list of tuples
        representing predicates.
        """
        objects: List[Tuple[str, str]] = []
        initState: List[Tuple] = []

        # Pacman object
        objects.append(('pacman', 'pacman'))

        # Opponent ghosts objects (only visible ghosts are included)
        for opp in self.getOpponents(gameState):
            ghost_pos = gameState.getAgentPosition(opp)
            if ghost_pos is not None:
                g_name = f'ghost{opp}'
                objects.append((g_name, 'ghost'))
                loc_name = self.coord_to_loc[ghost_pos]
                initState.append(('ghostAt', loc_name))

        # Location objects
        for loc_str in self.pddl_locs:
            objects.append((loc_str, 'location'))

        # Pacman position
        pac_pos = gameState.getAgentPosition(self.index)
        if pac_pos is not None:
            pac_loc = self.coord_to_loc[pac_pos]
            initState.append(('at', 'pacman', pac_loc))

        # Food positions on the entire map (we include both sides to simplify)
        foodGrid = self.getFood(gameState)
        for x, y in foodGrid.asList():
            loc_name = self.coord_to_loc.get((x, y))
            if loc_name:
                initState.append(('foodAt', loc_name))

        # Capsules positions
        for x, y in gameState.getCapsules():
            loc_name = self.coord_to_loc.get((x, y))
            if loc_name:
                initState.append(('capsuleAt', loc_name))

        # Carrying predicate if we are carrying food
        agentState = gameState.getAgentState(self.index)
        if hasattr(agentState, 'numCarrying') and agentState.numCarrying > 0:
            initState.append(('carrying',))

        # Adjacency relations
        for loc_str, neighbours in self.adjacent.items():
            for n in neighbours:
                initState.append(('adjacent', loc_str, n))

        return objects, initState

    def get_pddl_goal(self, gameState: GameState) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Determine the positive and negative goal predicates for the PDDL
        planner based on the current game state.  This function follows
        a simple decision tree:

        1. If a visible ghost is within CLOSE_DISTANCE and capsules exist,
           the goal is to eat a capsule (capsulesEaten becomes true).
        2. Otherwise, if food pellets remain, the goal is to pick up at
           least one food pellet (carrying becomes true).
        3. If no food is left, there is no goal (empty list).

        Negative goals are not used in this simplified implementation.
        """
        positiveGoal: List[Tuple] = []
        negativeGoal: List[Tuple] = []

        pac_pos = gameState.getAgentPosition(self.index)
        # Determine if ghost is close
        ghost_close = False
        for opp in self.getOpponents(gameState):
            ghost_pos = gameState.getAgentPosition(opp)
            if ghost_pos is not None:
                dist = self.getMazeDistance(pac_pos, ghost_pos)
                if dist <= CLOSE_DISTANCE:
                    ghost_close = True
                    break

        # If ghost close and there is at least one capsule
        if ghost_close and len(gameState.getCapsules()) > 0:
            positiveGoal.append(('capsulesEaten',))
        else:
            # Check if there is food
            if len(self.getFood(gameState).asList()) > 0:
                # Want to carry food
                positiveGoal.append(('carrying',))
        return positiveGoal, negativeGoal

    def getHighLevelPlan(self, objects: List[Tuple[str, str]], initState: List[Tuple], positiveGoal: List[Tuple], negativeGoal: List[Tuple]) -> List[Tuple[Action, pddl_state]]:
        """
        Deprecated wrapper retained for compatibility.  This method
        previously solved planning problems using a single domain.
        Planning is now delegated to `solvePlan` with an explicit solver
        argument.  It is preserved here to avoid breaking external calls,
        but will simply invoke `solvePlan` using the currently selected
        solver.

        Objects, initial state and goals are forwarded directly.
        """
        return self.solvePlan(self.current_solver, objects, initState, positiveGoal, negativeGoal)

    def solvePlan(self, solver: pddl_solver, objects: List[Tuple[str, str]], initState: List[Tuple], positiveGoal: List[Tuple], negativeGoal: List[Tuple]) -> List[Tuple[Action, pddl_state]]:
        """
        Solve a PDDL planning problem using the provided solver instance.
        The solver must have been initialised with a corresponding domain
        file.  The returned plan is a list of (Action, resulting_state)
        tuples.  If no plan can be found, an empty list is returned.
        """
        solver.parser_.reset_problem()
        solver.parser_.set_objects(objects)
        # When using piglet's parser, initial state is set via set_state
        solver.parser_.set_state(initState)
        # Negative and positive goals are set separately
        # If the solver does not support neg goals, passing an empty list
        solver.parser_.set_negative_goals(negativeGoal)
        solver.parser_.set_positive_goals(positiveGoal)
        print("[DEBUG] about to call solver.solve()")
        plan = solver.solve()
        print(f"[DEBUG] solver.solve() returned plan of length {len(plan) if plan else 0}")
        return plan or []


    def getApproxFeatures(self, gameState: GameState, action: str) -> util.Counter:
        """
        Extract approximate Q-learning features from the current gameState and
        candidate action.  The features correspond to those described in the
        blog: bias (constant 1), distance to the closest food, number of
        ghosts adjacent (distance 1), and whether food is eaten in the next
        state.
        """
        features = util.Counter()
        # Always include bias
        features['bias'] = 1.0

        # Compute successor position and state
        succState = gameState.generateSuccessor(self.index, action)
        succPos = succState.getAgentPosition(self.index)

        # Closest food distance (normalised)
        foodList = self.getFood(succState).asList()
        if len(foodList) > 0:
            dists = [self.getMazeDistance(succPos, food) for food in foodList]
            min_dist = min(dists)
            # Normalise by map size
            width = gameState.getWalls().width
            height = gameState.getWalls().height
            features['closest-food'] = float(min_dist) / float(width + height)
        else:
            features['closest-food'] = 0.0

        # Number of ghosts one step away
        ghostCount = 0
        for opp in self.getOpponents(gameState):
            ghostPos = succState.getAgentPosition(opp)
            if ghostPos is not None:
                if self.getMazeDistance(succPos, ghostPos) <= 1:
                    ghostCount += 1
        features['#-of-ghosts-1-step-away'] = ghostCount

        # Eats food feature: 1 if we eat a food pellet in this transition
        # Compare food count before and after
        oldFoodCount = len(self.getFood(gameState).asList())
        newFoodCount = len(self.getFood(succState).asList())
        features['eats-food'] = 1.0 if newFoodCount < oldFoodCount else 0.0

        return features

    def getApproxQValue(self, features: util.Counter) -> float:
        """
        Compute the approximate Q-value as the dot product between
        feature values and weights.
        """
        q = 0.0
        for f, val in features.items():
            weight = self.approxWeights.get(f, 0.0)
            q += weight * val
        return q

    def updateApproxWeights(self, features: util.Counter, reward: float, nextState: GameState) -> None:
        """
        Update approximate Q-learning weights using the observed reward and
        estimated value of the next state.  Implements the standard update
        rule: w_f <- w_f + alpha * (reward + discount * max_a' Q(s',a') - Q(s,a)) * featureValue.
        """
        # Compute maximum Q-value for next state
        legal = nextState.getLegalActions(self.index)
        if len(legal) == 0:
            nextMax = 0.0
        else:
            nextMax = max(self.getApproxQValue(self.getApproxFeatures(nextState, a)) for a in legal)
        currentQ = self.getApproxQValue(features)
        diff = (reward + self.discountRate * nextMax) - currentQ
        for f, val in features.items():
            # Weight update
            newWeight = self.approxWeights.get(f, 0.0) + self.alpha * diff * val
            self.approxWeights[f] = newWeight

    def getApproxReward(self, gameState: GameState, successorState: GameState) -> float:
        """
        Compute the reward for approximate Q-learning.  The reward scheme
        follows the blog description: difference of score multiplied by 10
        for successfully returning home with food, +10 when food is eaten in
        the next state, and -100 if the agent is eaten by a ghost.
        """
        reward = 0.0
        # Score difference times 10
        reward += (successorState.getScore() - gameState.getScore()) * 10.0
        # Food eaten: check if food count decreased
        if len(self.getFood(successorState).asList()) < len(self.getFood(gameState).asList()):
            reward += 10.0
        # Death penalty: if we were pacman and become a ghost (got eaten), penalise
        before_state = gameState.getAgentState(self.index)
        after_state = successorState.getAgentState(self.index)
        if before_state.isPacman and not after_state.isPacman:
            reward -= 100.0
        return reward

    def detectStuck(self) -> bool:
        """
        Detect if the agent is stuck by checking recent actions.  If the
        agent has repeated the same move pattern over the last several
        timesteps, consider it stuck and temporarily fall back to Q-learning
        without planning.  This implements the stuck detection described
        in the blog (approach five).
        """
        # We consider the agent stuck if the last 8 actions repeat a pattern
        pattern_length = 4
        max_len = pattern_length * 2
        if len(self.recentActions) < max_len:
            return False
        last_moves = self.recentActions[-max_len:]
        return last_moves[:pattern_length] == last_moves[pattern_length:]

    def getApproxAction(self, gameState: GameState) -> str:
        """
        Choose a low-level action using approximate Q-learning.  With
        probability epsilon, select a random legal action (exploration);
        otherwise choose the action with the maximum Q-value.  Ties are
        broken randomly.
        """
        legalActions = gameState.getLegalActions(self.index)
        # Avoid stop if other actions available
        if Directions.STOP in legalActions and len(legalActions) > 1:
            legalActions.remove(Directions.STOP)

        # Exploration
        if self.trainning and random.random() < self.epsilon:
            return random.choice(legalActions)

        # Exploitation: pick the best action
        bestVal = float('-inf')
        bestActions = []
        for action in legalActions:
            feats = self.getApproxFeatures(gameState, action)
            val = self.getApproxQValue(feats)
            if val > bestVal:
                bestVal = val
                bestActions = [action]
            elif val == bestVal:
                bestActions.append(action)
        return random.choice(bestActions)

    def chooseAction(self, gameState: GameState) -> str:
        """
        Main control loop for the agent.  The agent uses classical planning
        to decide a high-level goal when acting as pacman and not stuck.
        Based on this goal, the agent sets an intention (eat capsule or food).
        Then approximate Q-learning is used to select the actual low-level
        action.  If no plan is found or the agent is stuck, the agent
        directly relies on approximate Q-learning.
        """
        # Determine if we are currently pacman or ghost
        agentState = gameState.getAgentState(self.index)
        isPacman = agentState.isPacman

        # Track recent actions for stuck detection
        if self.lastAction:
            self.recentActions.append(self.lastAction)
            if len(self.recentActions) > 8:
                self.recentActions.pop(0)
        stuck = self.detectStuck()

        highLevelAction = None
        # Attempt high-level planning only if we are pacman and not stuck
        if isPacman and not stuck:
            # Build objects and initial state for the planner
            objects, initState = self.build_pddl_state(gameState)
            # Determine goals based on current state
            posGoal, negGoal = self.get_pddl_goal(gameState)
            # Use the pacman solver to compute a plan
            print(f"[DEBUG] time {gameState.data.timeleft} calling solvePlan with posGoal={posGoal}, negGoal={negGoal}")
            plan = self.solvePlan(self.pacman_solver, objects, initState, posGoal, negGoal)
            print(f"[DEBUG] returned plan length={len(plan)} at time {gameState.data.timeleft}")

            if len(plan) > 0:
                # Record solver and plan for subsequent execution
                self.highLevelPlan = plan
                self.current_solver = self.pacman_solver
                highLevelAction = plan[0][0].name
            else:
                # Reset high-level plan if no plan found
                self.highLevelPlan = []
                highLevelAction = None
        else:
            # When not planning (either ghost or stuck), clear high-level plan
            self.highLevelPlan = []

        # Low-level action via approximate Q-learning
        action = self.getApproxAction(gameState)

        # Perform Q-learning update during training
        if self.trainning:
            successorState = gameState.generateSuccessor(self.index, action)
            feats = self.getApproxFeatures(gameState, action)
            reward = self.getApproxReward(gameState, successorState)
            self.updateApproxWeights(feats, reward, successorState)

        # Update last action and record high-level intention for coordination
        self.lastAction = action
        MixedAgent.CURRENT_ACTION[self.index] = highLevelAction or ''
        return action

    #------------------------------- PDDL and High-Level Action Functions ------------------------------- 
    
    
    def getHighLevelPlan(self, objects, initState, positiveGoal, negtiveGoal) -> List[Tuple[Action,pddl_state]]:
        """
        Wrapper for high-level planning retained for backwards compatibility.
        By default this invokes solvePlan using the pacman solver.  External
        callers should migrate to using `solvePlan` directly with the
        appropriate solver instance.
        """
        return self.solvePlan(self.pacman_solver, objects, initState, positiveGoal, negtiveGoal)

    def get_pddl_state(self,gameState:GameState) -> Tuple[List[Tuple],List[Tuple]]:
        """
        This function collects pddl :objects and :init states from simulator gameState.
        """
        # Collect objects and states from the gameState

        states = []
        objects = []


        # Collect available foods on the map
        foodLeft = self.getFood(gameState).asList()
        if len(foodLeft) > 0:
            states.append(("food_available",))
        myPos = gameState.getAgentPosition(self.index)
        myObj = "a{}".format(self.index)
        cloestFoodDist = self.closestFood(myPos,self.getFood(gameState), gameState.getWalls())
        if cloestFoodDist != None and cloestFoodDist <=CLOSE_DISTANCE:
            states.append(("near_food",myObj))

        # Collect capsule states
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0 :
            states.append(("capsule_available",))
        for cap in capsules:
            if self.getMazeDistance(cap,myPos) <=CLOSE_DISTANCE:
                states.append(("near_capsule",myObj))
                break
        
        # Collect winning states
        currentScore = gameState.data.score
        if gameState.isOnRedTeam(self.index):
            if currentScore > 0:
                states.append(("winning",))
            if currentScore> 3:
                states.append(("winning_gt3",))
            if currentScore> 5:
                states.append(("winning_gt5",))
            if currentScore> 10:
                states.append(("winning_gt10",))
            if currentScore> 20:
                states.append(("winning_gt20",))
        else:
            if currentScore < 0:
                states.append(("winning",))
            if currentScore < -3:
                states.append(("winning_gt3",))
            if currentScore < -5:
                states.append(("winning_gt5",))
            if currentScore < -10:
                states.append(("winning_gt10",))
            if currentScore < -20:
                states.append(("winning_gt20",))

        # Collect team agents states
        agents : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getTeam(gameState)]
        for agent_index, agent_state in agents :
            agent_object = "a{}".format(agent_index)
            agent_type = "current_agent" if agent_index == self.index else "ally"
            objects += [(agent_object, agent_type)]

            if agent_index != self.index and self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(agent_index)) <= CLOSE_DISTANCE:
                states.append(("near_ally",))
            
            if agent_state.scaredTimer>0:
                states.append(("is_scared",agent_object))

            if agent_state.numCarrying>0:
                states.append(("food_in_backpack",agent_object))
                if agent_state.numCarrying >=20 :
                    states.append(("20_food_in_backpack",agent_object))
                if agent_state.numCarrying >=10 :
                    states.append(("10_food_in_backpack",agent_object))
                if agent_state.numCarrying >=5 :
                    states.append(("5_food_in_backpack",agent_object))
                if agent_state.numCarrying >=3 :
                    states.append(("3_food_in_backpack",agent_object))
                
            if agent_state.isPacman:
                states.append(("is_pacman",agent_object))
            
            

        # Collect enemy agents states
        enemies : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getOpponents(gameState)]
        noisyDistance = gameState.getAgentDistances()
        typeIndex = 1
        for enemy_index, enemy_state in enemies:
            enemy_position = enemy_state.getPosition()
            enemy_object = "e{}".format(enemy_index)
            objects += [(enemy_object, "enemy{}".format(typeIndex))]

            if enemy_state.scaredTimer>0:
                states.append(("is_scared",enemy_object))

            if enemy_position != None:
                for agent_index, agent_state in agents:
                    if self.getMazeDistance(agent_state.getPosition(), enemy_position) <= CLOSE_DISTANCE:
                        states.append(("enemy_around",enemy_object, "a{}".format(agent_index)))
            else:
                if noisyDistance[enemy_index] >=LONG_DISTANCE :
                    states.append(("enemy_long_distance",enemy_object, "a{}".format(self.index)))
                elif noisyDistance[enemy_index] >=MEDIUM_DISTANCE :
                    states.append(("enemy_medium_distance",enemy_object, "a{}".format(self.index)))
                else:
                    states.append(("enemy_short_distance",enemy_object, "a{}".format(self.index)))                                                                                                                                                                                                 


            if enemy_state.isPacman:
                states.append(("is_pacman",enemy_object))
            typeIndex += 1
            
        return objects, states
    
    def stateSatisfyCurrentPlan(self, init_state: List[Tuple],positiveGoal, negtiveGoal):
        if self.highLevelPlan is None or len(self.highLevelPlan) == 0:
            # No plan, need a new plan
            self.currentNegativeGoalStates = negtiveGoal
            self.currentPositiveGoalStates = positiveGoal
            return False
        
        if positiveGoal != self.currentPositiveGoalStates or negtiveGoal != self.currentNegativeGoalStates:
            return False
        # Use the solver associated with the current high-level plan for
        # precondition/effect matching.  When planning is not active,
        # current_solver defaults to the pacman solver.
        solver = getattr(self, 'current_solver', self.pacman_solver)

        # Check if the current state satisfies the effect of the current action
        if solver.matchEffect(init_state, self.highLevelPlan[self.currentActionIndex][0]):
            # Action completed.  Move to the next action if its preconditions are satisfied
            if self.currentActionIndex < len(self.highLevelPlan) - 1 and solver.satisfyPrecondition(init_state, self.highLevelPlan[self.currentActionIndex+1][0]):
                self.currentActionIndex += 1
                self.lowLevelPlan = []
                return True
            else:
                # Either no further actions or next action not applicable
                return False

        # Otherwise check if we can continue executing the current action
        if solver.satisfyPrecondition(init_state, self.highLevelPlan[self.currentActionIndex][0]):
            return True

        # Preconditions not satisfied; need a new plan
        return False
    
    def getGoals(self, objects: List[Tuple], initState: List[Tuple]):
        # Check a list of goal functions from high priority to low priority if the goal is applicable
        # Return the pddl goal states for selected goal function
        if (("winning_gt10",) in initState):
            return self.goalDefWinning(objects, initState)
        else:
            return self.goalScoring(objects, initState)

    def goalScoring(self,objects: List[Tuple], initState: List[Tuple]):
        # If we are not winning more than 5 points,
        # we invate enemy land and eat foods, and bring then back.

        positiveGoal = []
        negtiveGoal = [("food_available",)] # no food avaliable means eat all the food

        for obj in objects:
            agent_obj = obj[0]
            agent_type = obj[1]
            
            if agent_type == "enemy1" or agent_type == "enemy2":
                negtiveGoal += [("is_pacman", agent_obj)] # no enemy should standing on our land.
        
        return positiveGoal, negtiveGoal

    def goalDefWinning(self,objects: List[Tuple], initState: List[Tuple]):
        # If winning greater than 5 points,
        # this example want defend foods only, and let agents patrol on our ground.
        # The "win_the_game" pddl state is only reachable by the "patrol" action in pddl,
        # using it as goal, pddl will generate plan eliminate invading enemy and patrol on our ground.

        positiveGoal = [("defend_foods",)]
        negtiveGoal = []
        
        return positiveGoal, negtiveGoal

    #------------------------------- Heuristic search low level plan Functions -------------------------------
    def getLowLevelPlanHS(self, gameState: GameState, highLevelAction: str) -> List[Tuple[str,Tuple]]:
        # This is a function for plan low level actions using heuristic search.
        # You need to implement this function if you want to solve low level actions using heuristic search.
        # Here, we list some function you might need, read the GameState and CaptureAgent code for more useful functions.
        # These functions also useful for collecting features for Q learnning low levels.

        map = gameState.getWalls() # a 2d array matrix of obstacles, map[x][y] = true means a obstacle(wall) on x,y, map[x][y] = false indicate a free location
        foods = self.getFood(gameState) # a 2d array matrix of food,  foods[x][y] = true if there's a food.
        capsules = self.getCapsules(gameState) # a list of capsules
        foodNeedDefend = self.getFoodYouAreDefending(gameState) # return food will be eatan by enemy (food next to enemy)
        capsuleNeedDefend = self.getCapsulesYouAreDefending(gameState) # return capsule will be eatan by enemy (capsule next to enemy)
        Raise(NotImplementedError("Heuristic Search low level "))
        return [] # You should return a list of tuple of move action and target location (exclude current location).
    
    def posSatisfyLowLevelPlan(self,gameState: GameState):
        if self.lowLevelPlan == None or len(self.lowLevelPlan)==0 or self.lowLevelActionIndex >= len(self.lowLevelPlan):
            return False
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos,self.lowLevelPlan[self.lowLevelActionIndex][0])
        if nextPos != self.lowLevelPlan[self.lowLevelActionIndex][1]:
            return False
        return True

    #------------------------------- Q-learning low level plan Functions -------------------------------

    """
    Iterate through all q-values that we get from all
    possible actions, and return the action associated
    with the highest q-value.
    """
    def getLowLevelPlanQL(self, gameState:GameState, highLevelAction: str) -> List[Tuple[str,Tuple]]:
        values = []
        legalActions = gameState.getLegalActions(self.index)
        rewardFunction = None
        featureFunction = None
        weights = None
        learningRate = 0

        ##########
        # The following classification of high level actions is only a example.
        # You should think and use your own way to design low level planner.
        ##########
        if highLevelAction == "attack":
            # The q learning process for offensive actions are complete, 
            # you can improve getOffensiveFeatures to collect more useful feature to pass more information to Q learning model
            # you can improve the getOffensiveReward function to give reward for new features and improve the trainning process .
            rewardFunction = self.getOffensiveReward
            featureFunction = self.getOffensiveFeatures
            weights = self.getOffensiveWeights()
            learningRate = self.alpha
        elif highLevelAction == "go_home":
            # The q learning process for escape actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getEscapeReward
            featureFunction = self.getEscapeFeatures
            weights = self.getEscapeWeights()
            learningRate = 0 # learning rate set to 0 as reward function not implemented for this action, do not do q update, 
        else:
            # The q learning process for defensive actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getDefensiveReward
            featureFunction = self.getDefensiveFeatures
            weights = self.getDefensiveWeights()
            learningRate = 0 # learning rate set to 0 as reward function not implemented for this action, do not do q update 

        if len(legalActions) != 0:
            prob = util.flipCoin(self.epsilon) # get change of perform random movement
            if prob and self.trainning:
                action = random.choice(legalActions)
            else:
                for action in legalActions:
                        if self.trainning:
                            self.updateWeights(gameState, action, rewardFunction, featureFunction, weights,learningRate)
                        values.append((self.getQValue(featureFunction(gameState, action), weights), action))
                action = max(values)[1]
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos,action)
        return [(action, nextPos)]


    """
    Iterate through all features (closest food, bias, ghost dist),
    multiply each of the features' value to the feature's weight,
    and return the sum of all these values to get the q-value.
    """
    def getQValue(self, features, weights):
        return features * weights
    
    """
    Iterate through all features and for each feature, update
    its weight values using the following formula:
    w(i) = w(i) + alpha((reward + discount*value(nextState)) - Q(s,a)) * f(i)(s,a)
    """
    def updateWeights(self, gameState, action, rewardFunction, featureFunction, weights, learningRate):
        features = featureFunction(gameState, action)
        nextState = self.getSuccessor(gameState, action)

        reward = rewardFunction(gameState, nextState)
        for feature in features:
            correction = (reward + self.discountRate*self.getValue(nextState, featureFunction, weights)) - self.getQValue(features, weights)
            weights[feature] =weights[feature] + learningRate*correction * features[feature]
        
    
    """
    Iterate through all q-values that we get from all
    possible actions, and return the highest q-value
    """
    def getValue(self, nextState: GameState, featureFunction, weights):
        qVals = []
        legalActions = nextState.getLegalActions(self.index)

        if len(legalActions) == 0:
            return 0.0
        else:
            for action in legalActions:
                features = featureFunction(nextState, action)
                qVals.append(self.getQValue(features,weights))
            return max(qVals)
    
    def getOffensiveReward(self, gameState: GameState, nextState: GameState):
        # Calculate the reward. 
        currentAgentState:AgentState = gameState.getAgentState(self.index)
        nextAgentState:AgentState = nextState.getAgentState(self.index)

        ghosts = self.getGhostLocs(gameState)
        ghost_1_step = sum(nextAgentState.getPosition() in Actions.getLegalNeighbors(g,gameState.getWalls()) for g in ghosts)

        base_reward =  -50 + nextAgentState.numReturned + nextAgentState.numCarrying
        new_food_returned = nextAgentState.numReturned - currentAgentState.numReturned
        score = self.getScore(nextState)

        if ghost_1_step > 0:
            base_reward -= 5
        if score <0:
            base_reward += score
        if new_food_returned > 0:
            # return home with food get reward score
            base_reward += new_food_returned*10
        
        print("Agent ", self.index," reward ",base_reward)
        return base_reward
    
    def getDefensiveReward(self,gameState, nextState):
        print("Warnning: DefensiveReward not implemented yet, and learnning rate is 0 for defensive ",file=sys.stderr)
        return 0
    
    def getEscapeReward(self,gameState, nextState):
        print("Warnning: EscapeReward not implemented yet, and learnning rate is 0 for escape",file=sys.stderr)
        return 0



    #------------------------------- Feature Related Action Functions -------------------------------


    
    def getOffensiveFeatures(self, gameState: GameState, action):
        food = self.getFood(gameState) 
        currAgentState = gameState.getAgentState(self.index)

        walls = gameState.getWalls()
        ghosts = self.getGhostLocs(gameState)
        
        # Initialize features
        features = util.Counter()
        nextState = self.getSuccessor(gameState, action)

        # Successor Score
        features['successorScore'] = self.getScore(nextState)/(walls.width+walls.height) * 10

        # Bias
        features["bias"] = 1.0
        
        # Get the location of pacman after he takes the action
        next_x, next_y = nextState.getAgentPosition(self.index)

        # Number of Ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts) 
        
        
        dist_home =  self.getMazeDistance((next_x, next_y), gameState.getInitialAgentPosition(self.index))+1

        features["chance-return-food"] = (currAgentState.numCarrying)*(1 - dist_home/(walls.width+walls.height)) # The closer to home, the larger food carried, more chance return food
        
        # Closest food
        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features["closest-food"] = dist/(walls.width+walls.height)
        else:
            features["closest-food"] = 0

        return features

    def getOffensiveWeights(self):
        return MixedAgent.QLWeights["offensiveWeights"]
    


    def getEscapeFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemiesAround = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(enemiesAround) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemiesAround]
            features['enemyDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        features["distanceToHome"] = self.getMazeDistance(myPos,self.startPosition)

        return features

    def getEscapeWeights(self):
        return MixedAgent.QLWeights["escapeWeights"]
    


    def getDefensiveFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        team = [successor.getAgentState(i) for i in self.getTeam(successor)]
        team_dist = self.getMazeDistance(team[0].getPosition(), team[1].getPosition())
        features['teamDistance'] = team_dist

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getDefensiveWeights(self):
        return MixedAgent.QLWeights["defensiveWeights"]
    
    def closestFood(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None
    
    def stateClosestFood(self, gameState:GameState):
        pos = gameState.getAgentPosition(self.index)
        food = self.getFood(gameState)
        walls = gameState.getWalls()
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None
    
    def getSuccessor(self, gameState: GameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
    
    def getGhostLocs(self, gameState:GameState):
        ghosts = []
        opAgents = CaptureAgent.getOpponents(self, gameState)
        # Get ghost locations and states if observable
        if opAgents:
                for opponent in opAgents:
                        opPos = gameState.getAgentPosition(opponent)
                        opIsPacman = gameState.getAgentState(opponent).isPacman
                        if opPos and not opIsPacman: 
                                ghosts.append(opPos)
        return ghosts
    

