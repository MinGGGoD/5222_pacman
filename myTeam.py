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
from typing import List, Tuple

from numpy import true_divide
from numpy.ma import negative
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os
from capture import GameState, noisyDistance
from game import Directions, Actions, AgentState, Agent
from util import nearestPoint
import sys, os
import heapq

# the folder of current file.
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

from lib_piglet.utils.pddl_solver import pddl_solver
from lib_piglet.domains.pddl import pddl_state
from lib_piglet.utils.pddl_parser import Action

CLOSE_DISTANCE = 3
MEDIUM_DISTANCE = 15
LONG_DISTANCE = 25


#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed, first="MixedAgent", second="MixedAgent"):
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
    This is an agent that use pddl to guide the high level actions of Pacman
    """

    # Default weights for q learning, if no QLWeights.txt find, we use the following weights.
    # You should add your weights for new low level planner here as well.
    # weights are defined as class attribute here, so taht agents share same weights.
    QLWeights = {
        "offensiveWeights": {
            "bias": 1.0,
            "successorScore": 100.0,
            "chance-return-food": 50.0,
            # --- Cost/Penalty (negative values) ---
            "closest-food": -10.0,  # Distance to food
            "#-of-ghosts-1-step-away": -200.0,  # Ghosts nearby
            "crash-ghost": -1,  # Crashing into ghost
            "stop": -1.0,  # Stopping
            "reverse": -1.0,  # Reversing
        },
        "defensiveWeights": {
            "onDefense": 100.0,  # Staying in defense area
            "teamDistance": 2.0,  # Keep distance from teammate
            # --- Cost/Penalty (negative values) ---
            "numInvaders": -1000.0,  # Invaders are extremely bad
            "invaderDistance": -10.0,  # encourages getting closer
            "distanceToBorder": -5.0,  # encourages getting closer
            "stop": -1.0,
            "reverse": -1.0,
        },
        "escapeWeights": {
            "onDefense": 1000.0,  # Returning to home territory
            "enemyDistance": 30.0,  # encourages maximizing distance
            # --- Cost/Penalty (negative values) ---
            "distanceToHome": -100.0,  # encourages minimizing distance
            "crash-ghost": -1.0,
            "stop": -1.0,
        },
    }
    QLWeightsFile = BASE_FOLDER + "/QLWeightsMyTeam_zym.txt"

    # Also can use class variable to exchange information between agents.
    CURRENT_ACTION = {}

    # team-shared state
    teamRoles = {}  # {index: "attack"/"defend"}
    teamTargets = {}  # {index: (x, y) for current main target}
    lastUpdateTurn = -1

    def registerInitialState(self, gameState: GameState):
        self.pddl_solver = pddl_solver(BASE_FOLDER + "/myTeam.pddl")
        self.highLevelPlan: List[Tuple[Action, pddl_state]] = (
            None  # Plan is a list Action and pddl_state
        )
        self.currentNegativeGoalStates = []
        self.currentPositiveGoalStates = []
        self.currentActionIndex = (
            0  # index of action in self.highLevelPlan should be execute next
        )

        self.startPosition = gameState.getAgentPosition(
            self.index
        )  # the start location of the agent
        CaptureAgent.registerInitialState(self, gameState)

        self.lowLevelPlan: List[Tuple[str, Tuple]] = []
        self.lowLevelActionIndex = 0

        # REMEMBER TRUN TRAINNING TO FALSE when submit to contest server.
        self.trainning = False  # trainning mode to true will keep update weights and generate random movements by prob.
        self.epsilon = 0.1  # default exploration prob, change to take a random step
        self.alpha = 0.02  # default learning rate
        self.discountRate = (
            0.9  # default discount rate on successor state q value when update
        )

        # Use a dictionary to save information about current agent.
        MixedAgent.CURRENT_ACTION[self.index] = {}

        # --- Cache border coordinates ---
        self.borderCoordinates = []
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height

        # Determine the border x coordinate on our side of the territory
        border_x = 0
        if self.red:
            # Red team
            border_x = (width // 2) - 1
        else:
            # Blue team
            border_x = width // 2

        # Iterate through all y points at this x coordinate
        for y in range(height):
            # If this point is not a wall then it's a passable border point
            if not walls[border_x][y]:
                self.borderCoordinates.append((border_x, y))

        """
        Open weights file if it exists, otherwise start with empty weights.
        NEEDS TO BE CHANGED BEFORE SUBMISSION

        """
        if os.path.exists(MixedAgent.QLWeightsFile):
            with open(MixedAgent.QLWeightsFile, "r") as file:
                MixedAgent.QLWeights = eval(file.read())
            # print("Load QLWeights:",MixedAgent.QLWeights )

    def final(self, gameState: GameState):
        """
        This function write weights into files after the game is over.
        You may want to comment (disallow) this function when submit to contest server.
        """
        if self.trainning:
            # print("Write QLWeights:", MixedAgent.QLWeights)
            file = open(MixedAgent.QLWeightsFile, "w")
            file.write(str(MixedAgent.QLWeights))
            file.close()

    def updateTeamState(self, gameState: GameState):
        """
        Called by the agent with the smallest index in the team:
        - Decide who attacks and who defends (write to teamRoles)
        - If there are invaders, call assignInvaders to assign defense targets (write to teamTargets)
        """
        team = self.getTeam(gameState)
        score = self.getScore(gameState)
        timeLeft = gameState.data.timeleft

        # --- Role assignment: one attacker one defender / double attack ---
        attacker = min(team)
        defender = max(team)

        # Case 1: Clearly leading & not much time left -> 1 attacker 1 defender
        if score >= 5 and timeLeft < 300:
            MixedAgent.teamRoles[attacker] = "attack"
            MixedAgent.teamRoles[defender] = "defend"

        # Case 2: Tie/slightly behind & plenty of time left -> double attack
        elif score <= 0 and timeLeft > 300:
            for idx in team:
                MixedAgent.teamRoles[idx] = "attack"

        # Case 3: Severely behind & not much time left -> double attack all-in
        elif score < 0 and timeLeft < 150:
            for idx in team:
                MixedAgent.teamRoles[idx] = "attack"

        # Other cases: default 1 attacker 1 defender
        else:
            MixedAgent.teamRoles[attacker] = "attack"
            MixedAgent.teamRoles[defender] = "defend"

        # --- Defense target assignment: only when someone is defending & there are invaders ---
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [e for e in enemies if e.isPacman and e.getPosition() is not None]

        # Clear previous turn's assignments first to avoid dirty data
        MixedAgent.teamTargets.clear()

        if not invaders:
            return

        # Check which agents in the team are defense roles
        defenders = [idx for idx in team if MixedAgent.teamRoles.get(idx) == "defend"]

        if not defenders:
            # No pure defense roles, no assignment
            return

        # only assign to these defenders
        self.assignInvaders(gameState, defenders, invaders)

    def assignInvaders(self, gameState: GameState, defenders, invaders):
        """
        Based on defender and invader positions, assign an interception target position to each defender,
        write to MixedAgent.teamTargets.
        defenders: [agentIndex, ...]  agents currently marked as defense roles
        invaders:  [AgentState, ...]  enemies currently in our half
        """
        # Calculate maze distance from each defender to each invader
        dist = {}
        for d in defenders:
            dPos = gameState.getAgentState(d).getPosition()
            for j, inv in enumerate(invaders):
                dist[(d, j)] = self.getMazeDistance(dPos, inv.getPosition())

        # Simple case: only one invader -> assign to the "closest defender"
        if len(invaders) == 1:
            j = 0
            target_pos = invaders[0].getPosition()
            best_def = min(defenders, key=lambda d: dist[(d, j)])
            MixedAgent.teamTargets[best_def] = target_pos
            return

        # Common case: two defenders + two invaders, do a minimum total cost assignment
        if len(defenders) >= 2 and len(invaders) >= 2:
            d0, d1 = defenders[0], defenders[1]
            # Only consider first two invaders, sufficient
            i0, i1 = 0, 1
            d00 = dist[(d0, i0)] + dist[(d1, i1)]
            d01 = dist[(d0, i1)] + dist[(d1, i0)]
            if d00 <= d01:
                MixedAgent.teamTargets[d0] = invaders[i0].getPosition()
                MixedAgent.teamTargets[d1] = invaders[i1].getPosition()
            else:
                MixedAgent.teamTargets[d0] = invaders[i1].getPosition()
                MixedAgent.teamTargets[d1] = invaders[i0].getPosition()
            return

        # Other unusual number combinations (e.g., 1 defender vs multiple invaders), each defender chases closest invader
        for d in defenders:
            best_j = min(range(len(invaders)), key=lambda j: dist[(d, j)])
            MixedAgent.teamTargets[d] = invaders[best_j].getPosition()

    def chooseAction(self, gameState: GameState):
        """
        This is the action entry point for the agent.
        In the game, this function is called when its current agent's turn to move.

        We first pick a high-level action.
        Then generate low-level action ("North", "South", "East", "West", "Stop") to achieve the high-level action.
        """

        # ---------- Leader updates team state (roles + defense target assignment) ----------
        team = self.getTeam(gameState)
        leader = min(team)
        if self.index == leader:
            self.updateTeamState(gameState)

        # -------------High Level Plan Section-------------------
        # Get high level action from a pddl plan.

        # Collect objects and init states from gameState
        objects, initState = self.get_pddl_state(gameState)
        positiveGoal, negtiveGoal = self.getGoals(objects, initState, gameState)

        # Check if we can stick to current plan
        if not self.stateSatisfyCurrentPlan(initState, positiveGoal, negtiveGoal):
            # Cannot stick to current plan, prepare goals and replan
            print("Agnet:", self.index, "compute plan:")
            print(
                "\tOBJ:" + str(objects),
                "\tINIT:" + str(initState),
                "\tPOSITIVE_GOAL:" + str(positiveGoal),
                "\tNEGTIVE_GOAL:" + str(negtiveGoal),
                sep="\n",
            )
            self.highLevelPlan: List[Tuple[Action, pddl_state]] = self.getHighLevelPlan(
                objects, initState, positiveGoal, negtiveGoal
            )  # Plan is a list Action and pddl_state
            self.currentActionIndex = 0
            self.lowLevelPlan = []  # reset low level plan
            self.currentNegativeGoalStates = negtiveGoal
            self.currentPositiveGoalStates = positiveGoal
            print("\tPLAN:", self.highLevelPlan)
        if len(self.highLevelPlan) == 0:
            raise Exception(
                "Solver retuned empty plan, you need to think how you handle this situation or how you modify your model "
            )

        # Get next action from the plan
        highLevelAction = self.highLevelPlan[self.currentActionIndex][0].name
        MixedAgent.CURRENT_ACTION[self.index] = highLevelAction

        # -------------Low Level Plan Section-------------------
        # Get the low level plan using Q learning, and return a low level action at last.
        # A low level action is defined in Directions, whihc include {"North", "South", "East", "West", "Stop"}

        if not self.posSatisfyLowLevelPlan(gameState):
            # self.lowLevelPlan = self.getLowLevelPlanQL(gameState, highLevelAction) #Generate low level plan with q learning
            self.lowLevelPlan = self.getLowLevelPlanHS(
                gameState, highLevelAction
            )  # Generate low level plan with q learning
            # you can replace the getLowLevelPlanQL with getLowLevelPlanHS and implement heuristic search planner
            self.lowLevelActionIndex = 0
        lowLevelAction = self.lowLevelPlan[self.lowLevelActionIndex][0]
        self.lowLevelActionIndex += 1
        print("\tAgent:", self.index, lowLevelAction)
        return lowLevelAction

    # ------------------------------- PDDL and High-Level Action Functions -------------------------------

    def getHighLevelPlan(
        self, objects, initState, positiveGoal, negtiveGoal
    ) -> List[Tuple[Action, pddl_state]]:
        """
        This function prepare the pddl problem, solve it and return pddl plan
        """
        # Prepare pddl problem
        self.pddl_solver.parser_.reset_problem()
        self.pddl_solver.parser_.set_objects(objects)
        self.pddl_solver.parser_.set_state(initState)
        self.pddl_solver.parser_.set_negative_goals(negtiveGoal)
        self.pddl_solver.parser_.set_positive_goals(positiveGoal)

        # Solve the problem and return the plan
        return self.pddl_solver.solve()

    def get_pddl_state(self, gameState: GameState) -> Tuple[List[Tuple], List[Tuple]]:
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
        cloestFoodDist = self.closestFood(
            myPos, self.getFood(gameState), gameState.getWalls()
        )
        if cloestFoodDist != None and cloestFoodDist <= CLOSE_DISTANCE:
            states.append(("near_food", myObj))

        # Collect capsule states
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0:
            states.append(("capsule_available",))
        for cap in capsules:
            if self.getMazeDistance(cap, myPos) <= CLOSE_DISTANCE:
                states.append(("near_capsule", myObj))
                break

        # Collect winning states
        currentScore = gameState.data.score
        if gameState.isOnRedTeam(self.index):
            if currentScore > 0:
                states.append(("winning",))
            if currentScore > 3:
                states.append(("winning_gt3",))
            if currentScore > 5:
                states.append(("winning_gt5",))
            if currentScore > 10:
                states.append(("winning_gt10",))
            if currentScore > 20:
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
        agents: List[Tuple[int, AgentState]] = [
            (i, gameState.getAgentState(i)) for i in self.getTeam(gameState)
        ]
        walls = gameState.getWalls()
        border_x = (walls.width // 2) - 1 if self.red else walls.width // 2

        for agent_index, agent_state in agents:
            agent_object = "a{}".format(agent_index)
            agent_type = "current_agent" if agent_index == self.index else "ally"
            objects += [(agent_object, agent_type)]

            if (
                agent_index != self.index
                and self.getMazeDistance(
                    gameState.getAgentPosition(self.index),
                    gameState.getAgentPosition(agent_index),
                )
                <= CLOSE_DISTANCE
            ):
                states.append(("near_ally",))

            if agent_state.scaredTimer > 0:
                states.append(("is_scared", agent_object))

            if agent_state.numCarrying > 0:
                states.append(("food_in_backpack", agent_object))
                if agent_state.numCarrying >= 20:
                    states.append(("20_food_in_backpack", agent_object))
                if agent_state.numCarrying >= 10:
                    states.append(("10_food_in_backpack", agent_object))
                if agent_state.numCarrying >= 5:
                    states.append(("5_food_in_backpack", agent_object))
                if agent_state.numCarrying >= 3:
                    states.append(("3_food_in_backpack", agent_object))

            if agent_state.isPacman:
                states.append(("is_pacman", agent_object))

            # ========== Collaboration predicates: collect teammate's current actions ==========
            if agent_index != self.index:  # Only process teammates, exclude self
                ally_pos = agent_state.getPosition()
                if ally_pos is not None:
                    ally_x, ally_y = int(ally_pos[0]), int(ally_pos[1])

                    # 1. (eat_enemy ?a - ally) - teammate is eating enemy (scared ghost)
                    # Check: teammate is pacman, and nearby has scared ghost
                    if agent_state.isPacman:
                        enemies = [
                            gameState.getAgentState(i)
                            for i in self.getOpponents(gameState)
                        ]
                        scared_ghosts = [
                            e
                            for e in enemies
                            if not e.isPacman
                            and e.getPosition() is not None
                            and e.scaredTimer > 5
                        ]
                        for scared_ghost in scared_ghosts:
                            if (
                                self.getMazeDistance(
                                    ally_pos, scared_ghost.getPosition()
                                )
                                <= 3
                            ):
                                states.append(("eat_enemy", agent_object))
                                break

                    # 2. (go_home ?a - ally) - teammate is going home
                    # Check: teammate is pacman, carrying food, and current position is in enemy territory (moving towards our side)
                    if agent_state.isPacman and agent_state.numCarrying > 0:
                        # Check if in enemy territory (based on x coordinate)
                        is_in_enemy_land = (self.red and ally_x >= border_x + 1) or (
                            not self.red and ally_x <= border_x
                        )
                        if is_in_enemy_land:
                            states.append(("go_home", agent_object))

                    # 3. (go_enemy_land ?a - ally) - teammate is going to enemy territory
                    # Check: teammate is not pacman, and current position is in home territory (preparing to go to enemy)
                    if not agent_state.isPacman:
                        # Check if in home territory
                        is_in_home = (self.red and ally_x <= border_x) or (
                            not self.red and ally_x > border_x
                        )
                        if is_in_home:
                            states.append(("go_enemy_land", agent_object))

                    # 4. (eat_capsule ?a - ally) - teammate is eating capsule
                    # Check: teammate is pacman, and nearby has capsule
                    if agent_state.isPacman:
                        capsules = self.getCapsules(gameState)
                        for capsule in capsules:
                            if (
                                self.getMazeDistance(ally_pos, capsule)
                                <= CLOSE_DISTANCE
                            ):
                                states.append(("eat_capsule", agent_object))
                                break

                    # 5. (eat_food ?a - ally) - teammate is eating food
                    # Check: teammate is pacman, and nearby has food
                    if agent_state.isPacman:
                        food_list = self.getFood(gameState).asList()
                        for food in food_list:
                            if self.getMazeDistance(ally_pos, food) <= CLOSE_DISTANCE:
                                states.append(("eat_food", agent_object))
                                break

        # Collect enemy agents states
        enemies: List[Tuple[int, AgentState]] = [
            (i, gameState.getAgentState(i)) for i in self.getOpponents(gameState)
        ]
        noisyDistance = gameState.getAgentDistances()
        typeIndex = 1
        for enemy_index, enemy_state in enemies:
            enemy_position = enemy_state.getPosition()
            enemy_object = "e{}".format(enemy_index)
            objects += [(enemy_object, "enemy{}".format(typeIndex))]

            if enemy_state.scaredTimer > 0:
                states.append(("is_scared", enemy_object))

            if enemy_position != None:
                for agent_index, agent_state in agents:
                    if (
                        self.getMazeDistance(agent_state.getPosition(), enemy_position)
                        <= CLOSE_DISTANCE
                    ):
                        states.append(
                            ("enemy_around", enemy_object, "a{}".format(agent_index))
                        )
            else:
                if noisyDistance[enemy_index] >= LONG_DISTANCE:
                    states.append(
                        ("enemy_long_distance", enemy_object, "a{}".format(self.index))
                    )
                elif noisyDistance[enemy_index] >= MEDIUM_DISTANCE:
                    states.append(
                        (
                            "enemy_medium_distance",
                            enemy_object,
                            "a{}".format(self.index),
                        )
                    )
                else:
                    states.append(
                        ("enemy_short_distance", enemy_object, "a{}".format(self.index))
                    )

            if enemy_state.isPacman:
                states.append(("is_pacman", enemy_object))
            typeIndex += 1

        return objects, states

    def stateSatisfyCurrentPlan(
        self, init_state: List[Tuple], positiveGoal, negtiveGoal
    ):
        if self.highLevelPlan is None or len(self.highLevelPlan) == 0:
            # No plan, need a new plan
            self.currentNegativeGoalStates = negtiveGoal
            self.currentPositiveGoalStates = positiveGoal
            return False

        if (
            positiveGoal != self.currentPositiveGoalStates
            or negtiveGoal != self.currentNegativeGoalStates
        ):
            return False

        if self.pddl_solver.matchEffect(
            init_state, self.highLevelPlan[self.currentActionIndex][0]
        ):
            # The current state match the effect of current action, current action action done, move to next action
            if self.currentActionIndex < len(
                self.highLevelPlan
            ) - 1 and self.pddl_solver.satisfyPrecondition(
                init_state, self.highLevelPlan[self.currentActionIndex + 1][0]
            ):
                # Current action finished and next action is applicable
                self.currentActionIndex += 1
                self.lowLevelPlan = []  # reset low level plan
                return True
            else:
                # Current action finished, next action is not applicable or finish last action in the plan
                return False

        if self.pddl_solver.satisfyPrecondition(
            init_state, self.highLevelPlan[self.currentActionIndex][0]
        ):
            # Current action precondition satisfied, continue executing current action of the plan
            return True

        # Current action precondition not satisfied anymore, need new plan
        return False

    def getGoals(
        self, objects: List[Tuple], initState: List[Tuple], gameState: GameState
    ):
        # Check a list of goal functions from high priority to low priority if the goal is applicable
        # Return the pddl goal states for selected goal function
        myObj = "a{}".format(self.index)
        myCarrierGoal = False
        # Current agent
        cur_agent_state = gameState.getAgentState(self.index)
        carrying_count = cur_agent_state.numCarrying
        walls = gameState.getWalls()

        # Check if carrying enough food
        for fact in initState:
            if len(fact) == 2 and fact[0] == "3_food_in_backpack" and fact[1] == myObj:
                myCarrierGoal = True
                break

        # ==================== time awareness ====================
        timeRemaining = gameState.data.timeleft
        currentScore = self.getScore(gameState)

        # Get distance from current position to home (to judge if there's enough time)
        myPos = gameState.getAgentPosition(self.index)
        border_positions = (
            self.borderCoordinates if hasattr(self, "borderCoordinates") else []
        )
        if border_positions:
            dist_to_home = min(self.getMazeDistance(myPos, b) for b in border_positions)
        else:
            dist_to_home = walls.width // 2  # Estimated value

        # ==================== Priority 1: Time urgency strategy ====================
        if timeRemaining < 100:
            # Extremely urgent (<100 steps)
            if currentScore > 0:
                # Leading: return home immediately to secure victory
                print(
                    f"Agent {self.index}: time<100 and leading({currentScore}), return home immediately to secure victory"
                )
                # Check if current agent is pacman
                if cur_agent_state.isPacman:
                    # Return home
                    positiveGoal = []
                    negtiveGoal = [("is_pacman", myObj)]
                    return positiveGoal, negtiveGoal
                else:  # Patrol or defend
                    return self.goalDefWinning(objects, initState)
            elif currentScore < 0:
                # Behind: must take risk to get 1 point
                if carrying_count > 0:
                    print(
                        f"Agent {self.index}: time<100 and behind({currentScore}), carrying {carrying_count}, return home immediately"
                    )
                    positiveGoal = []
                    negtiveGoal = [("is_pacman", myObj)]
                    return positiveGoal, negtiveGoal
                else:
                    print(
                        f"Agent {self.index}: time<100 and behind({currentScore}), risk grabbing last food"
                    )
                    return self.goalScoringAggressive(objects, initState)
            else:
                # Tie: grab 1 point is enough
                if carrying_count > 0:
                    print(
                        f"Agent {self.index}: time<100 and tie, carrying {carrying_count}, return home"
                    )
                    positiveGoal = []
                    negtiveGoal = [("is_pacman", myObj)]
                    return positiveGoal, negtiveGoal
                else:
                    print(f"Agent {self.index}: time<100 and tie, grab 1 food")
                    return self.goalScoringAggressive(objects, initState)

        elif timeRemaining < 250:
            # Relatively urgent (100-250 steps)
            if currentScore > 0:
                # Leading: conservative, return home if carrying food
                if carrying_count > 0:
                    print(
                        f"Agent {self.index}: time<250 and leading({currentScore}), carrying {carrying_count}, return home"
                    )
                    positiveGoal = []
                    negtiveGoal = [("is_pacman", myObj)]
                    return positiveGoal, negtiveGoal
                else:
                    # Not carrying food, switch to defense
                    print(
                        f"Agent {self.index}: time<250 and leading({currentScore}), switch to defense"
                    )
                    return self.goalDefWinning(objects, initState)
            elif currentScore < 0:
                # Behind: need aggressive attack, but return home if carrying 2 or more
                max_home_dist = min(15, walls.width // 2)
                if carrying_count >= 2 or (
                    carrying_count > 0 and dist_to_home > max_home_dist
                ):
                    print(
                        f"Agent {self.index}: time<250 and behind({currentScore}), carrying {carrying_count} and distance {dist_to_home}, return home"
                    )
                    positiveGoal = []
                    negtiveGoal = [("is_pacman", myObj)]
                    return positiveGoal, negtiveGoal
                else:
                    print(
                        f"Agent {self.index}: time<250 and behind({currentScore}), continue attacking"
                    )
                    return self.goalScoring(objects, initState)
            else:
                # Tie: slightly aggressive
                if carrying_count >= 1:
                    print(
                        f"Agent {self.index}: time<250 and tie, carrying {carrying_count}, return home"
                    )
                    positiveGoal = []
                    negtiveGoal = [("is_pacman", myObj)]
                    return positiveGoal, negtiveGoal

        # ==================== Priority 2: Carrying food check ====================
        # When time is sufficient, return home with 3 food
        # When time is urgent (already handled), lower threshold
        if myCarrierGoal:
            print(f"Agent {self.index}: carrying >= 3 food, return home")
            positiveGoal = []
            negtiveGoal = [("is_pacman", myObj)]
            return positiveGoal, negtiveGoal

        # ==================== Priority 3: Defend when significantly leading ====================
        if ("winning_gt10",) in initState:
            print(f"Agent {self.index}: winning_gt10, go Patrol")
            return self.goalDefWinning(objects, initState)

        # ==================== Priority 4: Defend when no food available ====================
        if ("food_available",) not in initState:
            print(
                f"Agent {self.index}: no food_available, all agents return home to defend"
            )
            myAgentsIndices = self.getTeam(gameState)
            positiveGoal = []
            negtiveGoal = []
            for i in myAgentsIndices:
                negtiveGoal += [("is_pacman", "a{}".format(i))]
            return positiveGoal, negtiveGoal

        # ==================== Priority 5: Default attack ====================
        # Check if should be more aggressive (when behind)
        if currentScore < -3 and timeRemaining > 400:
            print(
                f"Agent {self.index}: behind {abs(currentScore)} points and time sufficient, aggressive attack"
            )
            return self.goalScoringAggressive(objects, initState)
        else:
            print(f"Agent {self.index}: normal attack mode")
            return self.goalScoring(objects, initState)

    def goalScoring(self, objects: List[Tuple], initState: List[Tuple]):
        # If we are not winning more than 5 points,
        # we invate enemy land and eat foods, and bring then back.
        current_agent_obj = "a{}".format(self.index)
        positiveGoal = [
            ("is_pacman", current_agent_obj),
            ("3_food_in_backpack", current_agent_obj),
        ]
        negtiveGoal = []  # no food avaliable means eat all the food

        for obj in objects:
            agent_obj = obj[0]
            agent_type = obj[1]

            if agent_type == "enemy1" or agent_type == "enemy2":
                negtiveGoal += [
                    ("is_pacman", agent_obj)
                ]  # no enemy should standing on our land.

        return positiveGoal, negtiveGoal

    def goalScoringAggressive(self, objects: List[Tuple], initState: List[Tuple]):
        """
        Aggressive attack
        """
        myObj = "a{}".format(self.index)

        # Only require carrying food, ignore defense and eliminating all food
        positiveGoal = [("is_pacman", myObj), ("3_food_in_backpack", myObj)]
        negtiveGoal = []

        return positiveGoal, negtiveGoal

    def goalDefWinning(self, objects: List[Tuple], initState: List[Tuple]):
        # If winning greater than 5 points,
        # this example want defend foods only, and let agents patrol on our ground.
        # The "defend_foods" pddl state is only reachable by the "patrol" action in pddl,
        # using it as goal, pddl will generate plan eliminate invading enemy and patrol on our ground.

        current_agent_obj = "a{}".format(self.index)

        # Check if there are enemy invaders (enemy is pacman)
        has_invader = False
        negtiveGoal = []

        # default eat enemies
        for obj in objects:
            agent_obj = obj[0]
            agent_type = obj[1]

            if agent_type == "enemy1" or agent_type == "enemy2":
                # Check if enemy is pacman (in initState)
                if ("is_pacman", agent_obj) in initState:
                    has_invader = True
                    # Set goal: require enemy is not pacman (achieved through defence action)
                    negtiveGoal.append(("is_pacman", agent_obj))

        if has_invader:
            # Has invaders: use defence action to eliminate invaders
            positiveGoal = []
            # negtiveGoal already set: require all invading enemies are not pacman
            print(
                f"Agent {self.index}: detected invaders, set defence goal: {negtiveGoal}"
            )
        else:
            # No invaders: use patrol action to patrol
            positiveGoal = [("defend_foods",)]
            negtiveGoal = []
            print(f"Agent {self.index}: no invaders, set patrol goal")

        return positiveGoal, negtiveGoal

    # ------------------------------- Heuristic search low level plan Functions -------------------------------
    def getLowLevelPlanHS(
        self, gameState: GameState, highLevelAction: str
    ) -> List[Tuple[str, Tuple]]:
        # This is a function for plan low level actions using heuristic search.
        # You need to implement this function if you want to solve low level actions using heuristic search.
        # Here, we list some function you might need, read the GameState and CaptureAgent code for more useful functions.
        # These functions also useful for collecting features for Q learnning low levels.

        walls = (
            gameState.getWalls()
        )  # a 2d array matrix of obstacles, map[x][y] = true means a obstacle(wall) on x,y, map[x][y] = false indicate a free location
        foods = self.getFood(
            gameState
        )  # a 2d array matrix of food,  foods[x][y] = true if there's a food.

        capsules = self.getCapsules(gameState)  # a list of capsules
        foodNeedDefend = self.getFoodYouAreDefending(
            gameState
        )  # return food will be eatan by enemy (food next to enemy)
        capsuleNeedDefend = self.getCapsulesYouAreDefending(
            gameState
        )  # return capsule will be eatan by enemy (capsule next to enemy)
        myPos = gameState.getAgentPosition(self.index)
        if highLevelAction == "attack":
            print(f"Agent {self.index} executing heuristic [Attack] strategy")
            return self._planAttack(gameState, myPos, walls)
        elif highLevelAction == "go_home":
            print(f"Agent {self.index} executing heuristic [Go Home] strategy")
            return self._planGoHome(gameState, myPos, walls)
        elif highLevelAction == "defence":
            print(f"Agent {self.index} executing heuristic [Defense] strategy")
            return self._planDefence(gameState, myPos, walls)
        elif highLevelAction == "patrol":
            print(f"Agent {self.index} executing heuristic [Patrol] strategy")
            return self._planPatrol(gameState, myPos, walls)
        else:  # default attack strategy
            return self._planAttack(gameState, myPos, walls)

    def posSatisfyLowLevelPlan(self, gameState: GameState):
        if (
            self.lowLevelPlan == None
            or len(self.lowLevelPlan) == 0
            or self.lowLevelActionIndex >= len(self.lowLevelPlan)
        ):
            return False
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(
            myPos, self.lowLevelPlan[self.lowLevelActionIndex][0]
        )
        if nextPos != self.lowLevelPlan[self.lowLevelActionIndex][1]:
            return False
        return True

    # ------------------------------- Heuristic search strategies Functions -------------------------------#
    def _planAttack(
        self, gameState: GameState, myPos: Tuple, walls
    ) -> List[Tuple[str, Tuple]]:
        """Attack strategy: find nearest food while avoiding enemies"""
        food_list = self.getFood(gameState).asList()
        ghosts = self.getGhostLocs(gameState)
        capsules = self.getCapsules(gameState)

        # Check if there are scared ghosts to chase
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        scared_ghosts = [
            e.getPosition()
            for e in enemies
            if not e.isPacman and e.getPosition() is not None and e.scaredTimer > 5
        ]

        if scared_ghosts:
            closest_scared = min(
                scared_ghosts, key=lambda g: self.getMazeDistance(myPos, g)
            )
            if self.getMazeDistance(myPos, closest_scared) <= 8:
                path = self._astar(
                    myPos, closest_scared, walls, ghosts, avoid_ghosts=False
                )
                return (
                    path
                    if path
                    else self._getEmergencyMove(gameState, myPos, walls, ghosts)
                )

        # Check if should eat capsule first
        if capsules and ghosts:
            closest_capsule = min(
                capsules, key=lambda c: self.getMazeDistance(myPos, c)
            )
            closest_ghost_dist = min(self.getMazeDistance(myPos, g) for g in ghosts)
            capsule_dist = self.getMazeDistance(myPos, closest_capsule)

            if closest_ghost_dist <= 5 and capsule_dist <= 4:
                path = self._astar(
                    myPos, closest_capsule, walls, ghosts, avoid_ghosts=True
                )
                return (
                    path
                    if path
                    else self._getEmergencyMove(gameState, myPos, walls, ghosts)
                )

        # Normal attack: find nearest food
        if not food_list:
            return self._planGoHome(gameState, myPos, walls)

        # Partition strategy: decide whether to partition based on food dispersion
        target_foods = self._shouldPartition(food_list, walls, gameState)

        # ========== Evaluate food safety ==========
        safe_foods = []
        for food in target_foods:
            if not self._isDeadEndTrap(food, walls, ghosts):
                safe_foods.append(food)

        # If all food is unsafe, choose the least dangerous
        if not safe_foods:
            safe_foods = target_foods

        # ========= Shallow/Deep division (based on dead-end depth) =========
        team = self.getTeam(gameState)
        is_shallow_runner = self.index == min(
            team
        )  # Smaller index handles shallow, larger index handles deep
        depth_limit = 5  # Threshold: <= 5 is considered "shallow", > 5 is considered "deep", you can adjust later

        # First divide safe_foods by dead-end depth
        if is_shallow_runner:
            # Shallow runner: prioritize shallow food, safe route + fast return home
            candidate_foods = [
                f for f in safe_foods if self._getDeadEndDepth(f, walls) <= depth_limit
            ]
            if not candidate_foods:
                # All are deep pits, then reluctantly choose from safe_foods
                candidate_foods = safe_foods
        else:
            # Deep assassin: prioritize deep food (usually more profitable with capsule / scared ghost)
            candidate_foods = [
                f for f in safe_foods if self._getDeadEndDepth(f, walls) > depth_limit
            ]
            if not candidate_foods:
                candidate_foods = safe_foods

        # Find optimal food
        best_food = None
        best_score = float("inf")

        for food in candidate_foods[:15]:  # Consider first 15
            food_dist = self.getMazeDistance(myPos, food)

            # Calculate danger score
            danger_score = 0
            if ghosts:
                min_ghost_dist = min(self.getMazeDistance(food, g) for g in ghosts)
                danger_score = max(0, 10 - min_ghost_dist)

            # Calculate dead-end depth penalty
            deadend_penalty = self._getDeadEndDepth(food, walls)

            # Comprehensive score
            total_score = food_dist + danger_score * 3 + deadend_penalty * 2

            if total_score < best_score:
                best_score = total_score
                best_food = food

        if best_food:
            path = self._astar(myPos, best_food, walls, ghosts, avoid_ghosts=True)
            return (
                path
                if path
                else self._getEmergencyMove(gameState, myPos, walls, ghosts)
            )

        return self._getEmergencyMove(gameState, myPos, walls, ghosts)

    def _planGoHome(
        self, gameState: GameState, myPos: Tuple, walls
    ) -> List[Tuple[str, Tuple]]:
        """Go home strategy: find nearest border line"""
        ghosts = self.getGhostLocs(gameState)
        border_positions = (
            self.borderCoordinates if hasattr(self, "borderCoordinates") else []
        )

        if not border_positions:
            mid_x = walls.width // 2
            if self.red:
                mid_x = mid_x - 1
            border_positions = [
                (mid_x, y) for y in range(walls.height) if not walls[mid_x][y]
            ]

        # Find safest border point
        best_border = None
        best_score = float("inf")

        for border in border_positions:
            border_dist = self.getMazeDistance(myPos, border)
            danger_score = 0
            if ghosts:
                min_ghost_dist = min(self.getMazeDistance(border, g) for g in ghosts)
                danger_score = max(0, 6 - min_ghost_dist)

            total_score = border_dist + danger_score * 3
            if total_score < best_score:
                best_score = total_score
                best_border = border

        if best_border:
            path = self._astar(
                myPos, best_border, walls, ghosts, avoid_ghosts=True, escape_mode=True
            )
            return (
                path
                if path
                else self._getEmergencyMove(gameState, myPos, walls, ghosts)
            )

        return self._getEmergencyMove(gameState, myPos, walls, ghosts)

    def _planDefence(
        self, gameState: GameState, myPos: Tuple, walls
    ) -> List[Tuple[str, Tuple]]:
        """Defense strategy: intercept invaders (supports teamTargets coordination)"""
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [e for e in enemies if e.isPacman and e.getPosition() is not None]

        # ---------- 1. If there is an assigned target, prioritize assigned_target ----------
        assigned_target = MixedAgent.teamTargets.get(self.index, None)

        if assigned_target is not None:
            invader_positions = {
                e.getPosition() for e in invaders if e.getPosition() is not None
            }
            # Target still corresponds to some invader, or at least invaders still exist
            if assigned_target in invader_positions or invaders:
                path = self._astar(
                    myPos, assigned_target, walls, [], avoid_ghosts=False
                )
                if path:
                    return path
            # Target unreachable / already invalid -> delete, fall back to default logic
            MixedAgent.teamTargets.pop(self.index, None)

        # ---------- 2. No valid assignment: use closest invader logic ----------
        if invaders:
            closest_invader = min(
                invaders, key=lambda e: self.getMazeDistance(myPos, e.getPosition())
            )
            target = closest_invader.getPosition()
            path = self._astar(myPos, target, walls, [], avoid_ghosts=False)
            if path:
                return path
            # Can't find path, at least make an emergency move
            return self._getEmergencyMove(gameState, myPos, walls, [])

        # ---------- 3. No invaders at all: clear own target & patrol ----------
        MixedAgent.teamTargets.pop(self.index, None)
        return self._planPatrol(gameState, myPos, walls)

    def _planPatrol(
        self, gameState: GameState, myPos: Tuple, walls
    ) -> List[Tuple[str, Tuple]]:
        """Patrol strategy: patrol at home territory border"""
        import random

        border_positions = (
            self.borderCoordinates if hasattr(self, "borderCoordinates") else []
        )

        if not border_positions:
            mid_x = walls.width // 2
            if self.red:
                mid_x = mid_x - 1
            border_positions = [
                (mid_x, y) for y in range(walls.height) if not walls[mid_x][y]
            ]

        # Choose patrol points with distance between 5-15
        patrol_targets = [
            b for b in border_positions if 5 <= self.getMazeDistance(myPos, b) <= 10
        ]

        if not patrol_targets:
            patrol_targets = border_positions

        target = (
            random.choice(patrol_targets)
            if patrol_targets
            else border_positions[len(border_positions) // 2]
        )

        path = self._astar(myPos, target, walls, [], avoid_ghosts=False)
        return path if path else self._getEmergencyMove(gameState, myPos, walls, [])

    def _astar(
        self,
        start: Tuple,
        goal: Tuple,
        walls,
        ghosts: List[Tuple],
        avoid_ghosts: bool = True,
        escape_mode: bool = False,
    ) -> List[Tuple[str, Tuple]]:
        """
        A* search algorithm with dead-end avoidance
        Return: [(action, position), ...] - path from start to goal
        """
        import heapq

        frontier = []
        heapq.heappush(frontier, (0, 0, start, []))
        visited = {start: 0}

        # Ghost danger zone
        ghost_danger_zone = set()
        if avoid_ghosts and ghosts:
            danger_radius = 3 if escape_mode else 2
            for ghost in ghosts:
                for dx in range(-danger_radius, danger_radius + 1):
                    for dy in range(-danger_radius, danger_radius + 1):
                        x, y = int(ghost[0]) + dx, int(ghost[1]) + dy
                        if 0 <= x < walls.width and 0 <= y < walls.height:
                            if not walls[x][y]:
                                ghost_danger_zone.add((x, y))

        # ========== Precompute dead-end zones ==========
        deadend_zones = set()
        if ghosts:
            # Only compute when there are ghosts
            for x in range(walls.width):
                for y in range(walls.height):
                    pos = (x, y)
                    if not walls[x][y]:
                        neighbors = Actions.getLegalNeighbors(pos, walls)
                        # Dead-end: only 1 neighbor
                        if len(neighbors) <= 1:
                            deadend_zones.add(pos)
                        # Narrow passage: 2 neighbors and nearby has ghost
                        elif len(neighbors) == 2 and ghosts:
                            min_ghost_dist = min(
                                self.getMazeDistance(pos, g) for g in ghosts
                            )
                            if min_ghost_dist <= 5:
                                deadend_zones.add(pos)

        max_iterations = 1000
        iterations = 0

        while frontier and iterations < max_iterations:
            iterations += 1
            _, g_score, current, path = heapq.heappop(frontier)

            if current == goal:
                return path

            for action in [
                Directions.NORTH,
                Directions.SOUTH,
                Directions.EAST,
                Directions.WEST,
            ]:
                next_pos = Actions.getSuccessor(current, action)
                x, y = int(next_pos[0]), int(next_pos[1])

                if walls[x][y]:
                    continue

                # Calculate move cost
                move_cost = 1

                # Ghost danger zone increases cost
                if next_pos in ghost_danger_zone:
                    move_cost += 50 if escape_mode else 1

                # ========== Dead-end penalty ==========
                if next_pos in deadend_zones:
                    # If goal is in dead-end and very close, acceptable
                    if next_pos == goal and self.getMazeDistance(current, goal) <= 2:
                        move_cost += 5  # Light penalty
                    else:
                        # Otherwise significantly increase cost
                        move_cost += 50

                # Check if entering dead-end (few neighbors)
                neighbors = Actions.getLegalNeighbors(next_pos, walls)
                if len(neighbors) <= 2 and ghosts:
                    # In narrow area with ghost, increase cost
                    min_ghost_dist = min(
                        self.getMazeDistance(next_pos, g) for g in ghosts
                    )
                    if min_ghost_dist <= 6:
                        move_cost += 30

                # Avoid staying in place
                if next_pos == current:
                    move_cost += 100

                new_g_score = g_score + move_cost

                if next_pos not in visited or new_g_score < visited[next_pos]:
                    visited[next_pos] = new_g_score
                    h_score = self.getMazeDistance(next_pos, goal)
                    f_score = new_g_score + h_score
                    new_path = path + [(action, next_pos)]
                    heapq.heappush(frontier, (f_score, new_g_score, next_pos, new_path))

        return None

    def _shouldPartition(
        self, food_list: List[Tuple], walls, gameState: GameState
    ) -> List[Tuple]:
        """
        Decide whether to partition based on food dispersion
        If food is very dispersed, partition; if food is concentrated, don't partition

        Args:
            food_list: list of food positions
            walls: map walls object
            gameState: game state object

        Returns:
            list of food this agent should focus on
        """
        if not food_list or len(food_list) <= 3:
            # Too little food, no need to partition
            return food_list

        # Calculate food distribution range on Y axis
        y_coords = [food[1] for food in food_list]
        min_y = min(y_coords)
        max_y = max(y_coords)
        y_range = max_y - min_y

        # Calculate dispersion: distribution range relative to map height
        # If food distribution range exceeds 40% of map height, consider food dispersed, should partition
        dispersion_threshold = 0.4
        y_dispersion = y_range / walls.height if walls.height > 0 else 0

        # If map is too small (height <= 10), don't partition
        if walls.height < 10:
            print("Map too small, no partition")
            return food_list

        # If food distribution is very concentrated (low dispersion), don't partition
        if y_dispersion < dispersion_threshold:
            print("Food distribution very concentrated, no partition")
            return food_list

        # Food distribution is dispersed, partition
        teamIndices = self.getTeam(gameState)
        upperAgent = teamIndices[1] if self.red else teamIndices[0]
        midY = walls.height // 2

        if self.index == upperAgent:
            # Upper half agent
            upper_food = [f for f in food_list if f[1] >= midY]
            target_foods = upper_food if upper_food else food_list
        else:
            # Lower half agent
            lower_food = [f for f in food_list if f[1] < midY]
            target_foods = lower_food if lower_food else food_list

        return target_foods

    def _getEmergencyMove(
        self, gameState: GameState, myPos: Tuple, walls, ghosts: List[Tuple]
    ) -> List[Tuple[str, Tuple]]:
        """Emergency plan: choose a legal move away from enemies"""
        legal_actions = gameState.getLegalActions(self.index)
        legal_actions = [a for a in legal_actions if a != Directions.STOP]

        if not legal_actions:
            return [(Directions.STOP, myPos)]

        best_action = None
        best_score = -float("inf")

        for action in legal_actions:
            next_pos = Actions.getSuccessor(myPos, action)

            if ghosts:
                min_ghost_dist = min(self.getMazeDistance(next_pos, g) for g in ghosts)
                score = min_ghost_dist
            else:
                score = 0

            if score > best_score:
                best_score = score
                best_action = action

        next_pos = Actions.getSuccessor(myPos, best_action)
        return [(best_action, next_pos)]

    # ==================== Dead-end detection functions ====================

    def _isDeadEndTrap(self, position: Tuple, walls, ghosts: List[Tuple]) -> bool:
        """
        Check if a position is a dead-end trap

        Criteria:
        1. Position itself is a dead-end or narrow area
        2. And there is a ghost on the escape path

        Returns: True = dangerous dead-end trap, False = safe
        """
        if not ghosts:
            return False

        # Check if it's a dead-end
        neighbors = Actions.getLegalNeighbors(position, walls)

        if len(neighbors) <= 1:
            # Complete dead-end
            return True

        if len(neighbors) == 2:
            # Narrow passage, need further check
            # Calculate distance to nearest ghost
            min_ghost_dist = min(self.getMazeDistance(position, g) for g in ghosts)

            if min_ghost_dist <= 4:
                # Ghost is very close and in narrow area, dangerous
                return True

            # Check if escape path is blocked
            escape_routes = self._getEscapeRoutes(position, walls, depth=5)
            if len(escape_routes) <= 1:
                # Only one escape route, easily blocked
                if min_ghost_dist <= 6:
                    return True

        return False

    def _getDeadEndDepth(self, position: Tuple, walls) -> int:
        """
        Calculate the dead-end depth of a position

        Returns: number of steps needed from this position to reach an open area
            0 = open area (3+ exits)
            1-5 = shallow dead-end
            6+ = deep dead-end (very dangerous)
        """
        visited = set()
        queue = [(position, 0)]
        visited.add(position)

        while queue:
            pos, depth = queue.pop(0)

            # Check if reached open area
            neighbors = Actions.getLegalNeighbors(pos, walls)
            if len(neighbors) >= 3:
                return depth

            # Limit search depth
            if depth >= 10:
                return 10

            # Continue searching
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return 10  # Extremely deep dead-end

    def _getEscapeRoutes(self, position: Tuple, walls, depth: int = 5) -> List[Tuple]:
        """
        Get all escape routes from a position

        Returns: all reachable open area positions at depth
        """
        escape_positions = set()
        visited = set()
        queue = [(position, 0)]
        visited.add(position)

        while queue:
            pos, d = queue.pop(0)

            if d >= depth:
                # Check if this position is an open area
                neighbors = Actions.getLegalNeighbors(pos, walls)
                if len(neighbors) >= 3:
                    escape_positions.add(pos)
                continue

            # Continue searching
            for neighbor in Actions.getLegalNeighbors(pos, walls):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))

        return list(escape_positions)

    # ------------------------------- Q-learning low level plan Functions -------------------------------

    """
    Iterate through all q-values that we get from all
    possible actions, and return the action associated
    with the highest q-value.
    """

    def getLowLevelPlanQL(
        self, gameState: GameState, highLevelAction: str
    ) -> List[Tuple[str, Tuple]]:
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
            # Temporarily disable stop when escaping home
            non_stop = [a for a in legalActions if a != Directions.STOP]
            if non_stop:
                legalActions = non_stop
            # The q learning process for escape actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getEscapeReward
            featureFunction = self.getEscapeFeatures
            weights = self.getEscapeWeights()
            learningRate = (
                self.alpha
            )  # learning rate set to 0 as reward function not implemented for this action, do not do q update,
        else:
            # Temporarily disable stop when defending/patrolling
            non_stop = [a for a in legalActions if a != Directions.STOP]
            if non_stop:
                legalActions = non_stop
            # The q learning process for defensive actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getDefensiveReward
            featureFunction = self.getDefensiveFeatures
            weights = self.getDefensiveWeights()
            learningRate = (
                self.alpha
            )  # learning rate set to 0 as reward function not implemented for this action, do not do q update

        if len(legalActions) != 0:
            prob = util.flipCoin(self.epsilon)  # get change of perform random movement
            if prob and self.trainning:
                action = random.choice(legalActions)
            else:
                for action in legalActions:
                    if self.trainning:
                        self.updateWeights(
                            gameState,
                            action,
                            rewardFunction,
                            featureFunction,
                            weights,
                            learningRate,
                        )
                    values.append(
                        (
                            self.getQValue(featureFunction(gameState, action), weights),
                            action,
                        )
                    )
                action = max(values)[1]
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos, action)
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

    def updateWeights(
        self, gameState, action, rewardFunction, featureFunction, weights, learningRate
    ):
        features = featureFunction(gameState, action)
        nextState = self.getSuccessor(gameState, action)

        reward = rewardFunction(gameState, nextState)
        for feature in features:
            # Handling no key error
            if feature not in weights:
                print("feature not in weights:", feature)
                weights[feature] = 0.0
            correction = (
                reward
                + self.discountRate * self.getValue(nextState, featureFunction, weights)
            ) - self.getQValue(features, weights)
            weights[feature] = (
                weights[feature] + learningRate * correction * features[feature]
            )

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
                qVals.append(self.getQValue(features, weights))
            return max(qVals)

    def getOffensiveReward(self, gameState: GameState, nextState: GameState):
        # Calculate the reward.
        currentAgentState: AgentState = gameState.getAgentState(self.index)
        nextAgentState: AgentState = nextState.getAgentState(self.index)

        ghosts = self.getGhostLocs(gameState)
        ghost_1_step = sum(
            nextAgentState.getPosition()
            in Actions.getLegalNeighbors(g, gameState.getWalls())
            for g in ghosts
        )
        food_list = self.getFood(gameState).asList()
        walls = gameState.getWalls()
        base_reward = -50 + nextAgentState.numReturned + nextAgentState.numCarrying
        new_food_returned = nextAgentState.numReturned - currentAgentState.numReturned
        score = self.getScore(nextState)

        if ghost_1_step > 0:
            base_reward -= 100  # 5 -> 100
        if score < 0:
            base_reward += score
        if new_food_returned > 0:
            # return home with food get reward score
            base_reward += new_food_returned * 30

        if nextAgentState.getPosition() in food_list:
            base_reward += 30  # Reward for eating food

        # Penalty for crashing into ghost
        if nextAgentState.getPosition() in ghosts:
            base_reward -= 150

        # Check if the *next* state's direction is STOP
        if nextAgentState.configuration.direction == Directions.STOP:
            base_reward -= 100  # Penalty for stopping
        if nextAgentState.configuration.direction == Directions.REVERSE:
            base_reward -= 30  # Penalty for oscillating

        print("Agent ", self.index, " reward ", base_reward)
        return base_reward

    def getDefensiveReward(self, gameState: GameState, nextState: GameState):
        """
        Calculate reward for 'defensive' (patrol) action.
        Goal: intercept and eat invaders.
        """

        # --- 1. Get current and next state information ---
        myCurrentState = gameState.getAgentState(self.index)
        myNextState = nextState.getAgentState(self.index)
        myCurrentPos = myCurrentState.getPosition()
        myNextPos = myNextState.getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invandersAround = [
            a for a in enemies if a.isPacman and a.getPosition() is not None
        ]
        invandersAroundPos = [a.getPosition() for a in invandersAround]
        # Set a base "action cost" to encourage agent to complete tasks faster
        base_reward = -50

        # --- Basic logic ---
        if len(invandersAround) > 0:  # Has invaders
            # Reward for getting closer to invaders each step
            for invader in invandersAround:
                current_dist_to_invader = self.getMazeDistance(
                    myCurrentPos, invader.getPosition()
                )
                next_dist_to_invader = self.getMazeDistance(
                    myNextPos, invader.getPosition()
                )
                if next_dist_to_invader < current_dist_to_invader:
                    base_reward += 5 * (current_dist_to_invader - next_dist_to_invader)
                else:
                    base_reward -= 5  # Penalty for moving away from invader

            # Reward for eating invader
            if myNextPos in invandersAroundPos:
                base_reward += 200

            # # Penalty for being eaten when in scared state
            # if myCurrentState.scaredTimer > 0 and myNextPos == self.startPosition:
            #     base_reward -= 200
        # else:
        #     # No invaders, patrol, try to get closer to border
        #     currentDistToBorder = self.getDistanceToBorder(myCurrentPos)
        #     nextDistToBorder = self.getDistanceToBorder(myNextPos)
        #     if nextDistToBorder < currentDistToBorder:
        #         base_reward += 5  # Closer to border, good
        #     elif nextDistToBorder > currentDistToBorder:
        #         base_reward -= 5  # Moving away from border, bad

        # General penalties
        if myNextState.configuration.direction == Directions.STOP:
            base_reward -= 100  # Penalty for stopping
        if myNextState.configuration.direction == Directions.REVERSE:
            base_reward -= 30  # Penalty for oscillating

        # print(f"Agent {self.index} (Defensive) reward: {reward}") # Uncomment for debugging during training
        return base_reward

    def getEscapeReward(self, gameState, nextState):
        currentAgentState: AgentState = gameState.getAgentState(self.index)
        nextAgentState: AgentState = nextState.getAgentState(self.index)

        ghosts = self.getGhostLocs(gameState)
        ghost_1_step = sum(
            nextAgentState.getPosition()
            in Actions.getLegalNeighbors(g, gameState.getWalls())
            for g in ghosts
        )
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        middle_x = width // 2
        middle_y = height // 2
        base_reward = -50
        # + nextAgentState.numReturned + nextAgentState.numCarrying
        new_food_returned = nextAgentState.numReturned - currentAgentState.numReturned

        if (
            ghost_1_step > 0
        ):  # When escaping must stay away from ghost, give slightly larger penalty
            base_reward -= 10

        # if new_food_returned > 0:  # This logic already exists in offensive
        #     # When escaping definitely carrying food, top priority is to bring food home to score, so returning home with food gives large reward
        #     base_reward += new_food_returned * 50

        # Get distance to map center point, reward for getting closer to home
        curr_dist_to_border = self.getDistanceToBorder(currentAgentState.getPosition())
        next_dist_to_border = self.getDistanceToBorder(nextAgentState.getPosition())
        if next_dist_to_border < curr_dist_to_border:
            base_reward += 5
        else:
            base_reward -= 5

        # Check if the *next* state's direction is STOP
        if nextAgentState.configuration.direction == Directions.STOP:
            base_reward -= 100  # Penalty for stopping
        if nextAgentState.configuration.direction == Directions.REVERSE:
            base_reward -= -50  # Penalty for oscillating

        print("Agent ", self.index, " reward ", base_reward)
        return base_reward

    # ------------------------------- Feature Related Action Functions -------------------------------

    def getOffensiveFeatures(self, gameState: GameState, action):
        food = self.getFood(gameState)
        currAgentState = gameState.getAgentState(self.index)
        myPos = currAgentState.getPosition()

        walls = gameState.getWalls()
        ghosts = self.getGhostLocs(gameState)

        # Initialize features
        features = util.Counter()
        nextState = self.getSuccessor(gameState, action)
        nextAgentState: AgentState = nextState.getAgentState(self.index)

        # Successor Score
        features["successorScore"] = self.getScore(nextState) / (
            walls.width + walls.height
        )

        # Bias
        features["bias"] = 1.0

        # Get the location of pacman after he takes the action
        next_x, next_y = nextState.getAgentPosition(self.index)

        # Number of Ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts
        )

        # Distance to home should be considered as home once past the border
        # Check if next position has returned to home territory (i.e., no longer Pacman)
        if not nextState.getAgentState(self.index).isPacman:
            dist_home = 0
        else:
            # Distance to nearest border position
            border_positions = (
                self.borderCoordinates if hasattr(self, "borderCoordinates") else []
            )
            if border_positions:
                dist_home = min(
                    self.getMazeDistance((next_x, next_y), border)
                    for border in border_positions
                )
            # else:
            #     dist_home = self.getMazeDistance((next_x, next_y), gameState.getInitialAgentPosition(self.index))+1

        features["chance-return-food"] = (currAgentState.numCarrying) * (
            1 - dist_home / (walls.width + walls.height)
        )  # The closer to home, the larger food carried, more chance return food

        teamIndices = self.getTeam(gameState)
        upperAgent = teamIndices[1] if self.red else teamIndices[0]
        midY = walls.height // 2
        food_list = self.getFood(gameState).asList()

        if self.index == upperAgent:
            # Upper half target: only consider upper half food
            upper_food = [p for p in food_list if int(p[1]) >= midY]
            if upper_food:
                dist = min(
                    self.getMazeDistance((next_x, next_y), p) for p in upper_food
                )
            else:
                # When upper half has no food, fall back to global nearest food
                dist = self.closestFood((next_x, next_y), food, walls)
        else:
            # Lower half target: only consider lower half food
            lower_food = [p for p in food_list if int(p[1]) < midY]
            if lower_food:
                dist = min(
                    self.getMazeDistance((next_x, next_y), p) for p in lower_food
                )
            else:
                # When lower half has no food, fall back to global nearest food
                dist = self.closestFood((next_x, next_y), food, walls)

        if dist is not None:
            features["closest-food"] = dist / (walls.width + walls.height)
        else:
            features["closest-food"] = 0

        # New feature for crashing into ghost, penalize if crash into ghost
        if "crash-ghost" not in features:
            features["crash-ghost"] = 0
        if nextState.getAgentPosition(self.index) in ghosts:
            features["crash-ghost"] = 1.0
        else:
            features["crash-ghost"] = 0

        # stop feature, penalize if stop
        if action == Directions.STOP:
            features["stop"] = 1
        else:
            features["stop"] = 0

        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]
        nbrs_here = Actions.getLegalNeighbors(myPos, gameState.getWalls())
        is_deadend_here = (
            len(nbrs_here) == 1
        )  # If only one neighbor, consider it a dead-end, allow reverse
        features["reverse"] = 1 if (action == rev and not is_deadend_here) else 0

        return features

    def getOffensiveWeights(self):
        return MixedAgent.QLWeights["offensiveWeights"]

    def getEscapeFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features["onDefense"] = 1
        if myState.isPacman:
            features["onDefense"] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemiesAround = [
            a for a in enemies if not a.isPacman and a.getPosition() != None
        ]
        if len(enemiesAround) > 0:
            dists = [
                self.getMazeDistance(myPos, a.getPosition()) for a in enemiesAround
            ]
            # features['enemyDistance'] = min(dists)
            # norm
            walls = gameState.getWalls()
            features["enemyDistance"] = min(dists) / (walls.width + walls.height)

        # Calculate distance from current position to nearest point in home territory
        walls = gameState.getWalls()
        minHomeDist = float("inf")
        myX, myY = int(myPos[0]), int(myPos[1])
        width = walls.width
        height = walls.height

        isRed = self.red
        # Find all points belonging to home territory
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    if isRed and x < (width // 2):
                        home = (x, y)
                    elif not isRed and x >= (width // 2):
                        home = (x, y)
                    else:
                        continue
                    dist = self.getMazeDistance(myPos, home)
                    if dist < minHomeDist:
                        minHomeDist = dist
        features["distanceToHome"] = (
            (minHomeDist / (walls.width + walls.height))
            if minHomeDist != float("inf")
            else 0
        )

        # Penalty for crashing into ghost
        if "crash-ghost" not in features:
            features["crash-ghost"] = 0.0
        ghosts = self.getGhostLocs(gameState)
        if myPos in ghosts:
            features["crash-ghost"] = 1.0
        else:
            features["crash-ghost"] = 0

        if action == Directions.STOP:
            features["stop"] = 1
        else:
            features["stop"] = 0

        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]
        nbrs_here = Actions.getLegalNeighbors(myPos, gameState.getWalls())
        is_deadend_here = (
            len(nbrs_here) == 1
        )  # If only one neighbor, consider it a dead-end, allow reverse
        features["reverse"] = 1 if (action == rev and not is_deadend_here) else 0

        return features

    def getEscapeWeights(self):
        return MixedAgent.QLWeights["escapeWeights"]

    def getDefensiveFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        walls = gameState.getWalls()

        # Computes whether we're on defense (1) or offense (0)
        features["onDefense"] = 1.0
        if myState.isPacman:
            features["onDefense"] = 0

        team = [successor.getAgentState(i) for i in self.getTeam(successor)]
        team_dist = self.getMazeDistance(team[0].getPosition(), team[1].getPosition())
        features["teamDistance"] = team_dist / (walls.width + walls.height)

        # --- Defense logic ---

        # 1. Get all enemy states
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

        # 2. Identify all invaders (whether visible or not)
        all_invaders = [a for a in enemies if a.isPacman]

        # 3. Identify visible invaders
        visible_invaders = [a for a in all_invaders if a.getPosition() != None]

        # 4. Identify invisible invaders (get their indices)
        # Fix: cannot use a.index (AgentState has no attribute 'index'), so map state back to index
        opponent_indices = self.getOpponents(successor)
        unseen_invader_indices = [
            opponent_indices[i]
            for i, a in enumerate(enemies)
            if a.isPacman
            and a.getPosition() is None  # Only invisible Pacman (invaders)
        ]

        # 5. Get noisy distances of all enemies
        noisy_distances = successor.getAgentDistances()

        features["numInvaders"] = (
            len(all_invaders) / 2.0
        )  # Normalized: total number of invaders

        min_dist_to_invader = float("inf")

        if len(visible_invaders) > 0:
            # [Case A] At least one visible invader: use precise distance
            dists = [
                self.getMazeDistance(myPos, a.getPosition()) for a in visible_invaders
            ]
            min_dist_to_invader = min(dists)

        elif len(unseen_invader_indices) > 0:
            # [Case B] Enemy not visible, but we know it exists: use noisy distance
            # This is just an estimate
            dists = [noisy_distances[i] for i in unseen_invader_indices]
            min_dist_to_invader = min(dists)

        if min_dist_to_invader != float("inf"):
            # [Activate "chase" mode]
            # As long as we know there are invaders (whether visible or not), activate invaderDistance
            features["invaderDistance"] = min_dist_to_invader / (
                walls.width + walls.height
            )
            features["distanceToBorder"] = 0.0  # Turn off "patrol border"
        else:
            # [Activate "patrol" mode]
            # Only patrol border when confirmed no invaders
            features["invaderDistance"] = 0.0
            distToBorder = self.getDistanceToBorder(myPos)
            features["distanceToBorder"] = distToBorder / (walls.width + walls.height)

        if action == Directions.STOP:
            features["stop"] = 1.0
        else:
            features["stop"] = 0.0
        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]
        nbrs_here = Actions.getLegalNeighbors(myPos, gameState.getWalls())
        is_deadend_here = (
            len(nbrs_here) == 1
        )  # If only one neighbor, consider it a dead-end, allow reverse
        features["reverse"] = 1 if (action == rev and not is_deadend_here) else 0
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
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no food found
        return None

    def stateClosestFood(self, gameState: GameState):
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
                fringe.append((nbr_x, nbr_y, dist + 1))
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

    def getGhostLocs(self, gameState: GameState):
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

    def getDistanceToBorder(self, pos):
        """
        Calculate the maze distance from the given position 'pos'
        to the closest point in 'self.borderCoordinates'
        """
        if not self.borderCoordinates:
            return 0

        minDist = float("inf")
        for borderPos in self.borderCoordinates:
            dist = self.getMazeDistance(pos, borderPos)
            if dist < minDist:
                minDist = dist
        return minDist
