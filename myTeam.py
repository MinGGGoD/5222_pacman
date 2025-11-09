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
import sys,os
import heapq

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
    This is an agent that use pddl to guide the high level actions of Pacman
    """
    # Default weights for q learning, if no QLWeights.txt find, we use the following weights.
    # You should add your weights for new low level planner here as well.
    # weights are defined as class attribute here, so taht agents share same weights.
    QLWeights = {
        'offensiveWeights': {
            'bias': 1.0,                      # 必须是 0 或小的正数，以鼓励行动。绝不能是 -478。
            'successorScore': 100.0,          # 得分是主要目标
            'chance-return-food': 50.0,       # 运送食物是好事
            
            # --- 成本/惩罚 (负数) ---
            'closest-food': -10.0,            # 距离是成本
            '#-of-ghosts-1-step-away': -200.0, # 幽灵在旁边非常糟糕
            'crash-ghost': -1,           # 撞鬼是极度糟糕的 (特征值为 1.0)
            'stop': -1.0,                   # 停止是糟糕的
            'reverse': -1.0,                  # 回头是低效的
        }, 
        'defensiveWeights': {
            'onDefense': 100.0,               # 待在防守区域是好的
            'teamDistance': 2.0,              # 与队友保持距离（散开）
            
            # --- 成本/惩罚 (负数) ---
            'numInvaders': -1000.0,           # 入侵者是极度糟糕的
            'invaderDistance': -10.0,         # “距离”是成本，权重为负（鼓励靠近）
            'distanceToBorder': -5.0,         # “距离”是成本，权重为负（鼓励靠近）
            'stop': -1.0, 
            'reverse': -1.0
        }, 
        'escapeWeights': {
            'onDefense': 1000.0,              # 回到己方领地是首要目标
            'enemyDistance': 30.0,            # [关键修复] “距离”是正权重（鼓励最大化距离）
            
            # --- 成本/惩罚 (负数) ---
            'distanceToHome': -100.0,         # “距离”是成本，权重为负（鼓励最小化距离）
            'crash-ghost': -1.0,           
            'stop': -1.0
        }
    }
    QLWeightsFile = BASE_FOLDER+'/QLWeightsMyTeam_zym.txt'

    # Also can use class variable to exchange information between agents.
    CURRENT_ACTION = {}


    def registerInitialState(self, gameState: GameState):
        self.pddl_solver = pddl_solver(BASE_FOLDER+'/myTeam.pddl')
        self.highLevelPlan: List[Tuple[Action,pddl_state]] = None # Plan is a list Action and pddl_state
        self.currentNegativeGoalStates = []
        self.currentPositiveGoalStates = []
        self.currentActionIndex = 0 # index of action in self.highLevelPlan should be execute next

        self.startPosition = gameState.getAgentPosition(self.index) # the start location of the agent
        CaptureAgent.registerInitialState(self, gameState)

        self.lowLevelPlan: List[Tuple[str,Tuple]] = []
        self.lowLevelActionIndex = 0

        # REMEMBER TRUN TRAINNING TO FALSE when submit to contest server.
        self.trainning = False # trainning mode to true will keep update weights and generate random movements by prob.
        self.epsilon = 0.1 #default exploration prob, change to take a random step
        self.alpha = 0.02 #default learning rate
        self.discountRate = 0.9 # default discount rate on successor state q value when update
        
        # Use a dictionary to save information about current agent.
        MixedAgent.CURRENT_ACTION[self.index]={}
        
        # --- 新增：缓存边境线坐标 ---
        self.borderCoordinates = []
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        
        # 确定我方领土一侧的边境线 x 坐标
        border_x = 0
        if self.red:
            # 红队 (左侧), 边境在 (width // 2) - 1
            border_x = (width // 2) - 1
        else:
            # 蓝队 (右侧), 边境在 width // 2
            border_x = width // 2
            
        # 遍历该 x 坐标上的所有 y 点
        for y in range(height):
            # 如果这个点不是墙 [cite: 261, 482]，它就是一个可通行的边境点
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
        
    
    def final(self, gameState : GameState):
        """
        This function write weights into files after the game is over. 
        You may want to comment (disallow) this function when submit to contest server.
        """
        if self.trainning:
            # print("Write QLWeights:", MixedAgent.QLWeights)
            file = open(MixedAgent.QLWeightsFile, 'w')
            file.write(str(MixedAgent.QLWeights))
            file.close()
    

    def chooseAction(self, gameState: GameState):
        """
        This is the action entry point for the agent.
        In the game, this function is called when its current agent's turn to move.

        We first pick a high-level action.
        Then generate low-level action ("North", "South", "East", "West", "Stop") to achieve the high-level action.
        """

        #-------------High Level Plan Section-------------------
        # Get high level action from a pddl plan.

        # Collect objects and init states from gameState
        objects, initState = self.get_pddl_state(gameState)
        positiveGoal, negtiveGoal = self.getGoals(objects,initState, gameState)

        # Check if we can stick to current plan 
        if not self.stateSatisfyCurrentPlan(initState, positiveGoal, negtiveGoal):
            # Cannot stick to current plan, prepare goals and replan
            print("Agnet:",self.index,"compute plan:")
            print("\tOBJ:"+str(objects),"\tINIT:"+str(initState), "\tPOSITIVE_GOAL:"+str(positiveGoal), "\tNEGTIVE_GOAL:"+str(negtiveGoal),sep="\n")
            self.highLevelPlan: List[Tuple[Action,pddl_state]] = self.getHighLevelPlan(objects, initState,positiveGoal, negtiveGoal) # Plan is a list Action and pddl_state
            self.currentActionIndex = 0
            self.lowLevelPlan = [] # reset low level plan
            self.currentNegativeGoalStates = negtiveGoal
            self.currentPositiveGoalStates = positiveGoal
            print("\tPLAN:",self.highLevelPlan)
        if len(self.highLevelPlan)==0:
            raise Exception("Solver retuned empty plan, you need to think how you handle this situation or how you modify your model ")
        
        # Get next action from the plan
        highLevelAction = self.highLevelPlan[self.currentActionIndex][0].name
        MixedAgent.CURRENT_ACTION[self.index] = highLevelAction

        #-------------Low Level Plan Section-------------------
        # Get the low level plan using Q learning, and return a low level action at last.
        # A low level action is defined in Directions, whihc include {"North", "South", "East", "West", "Stop"}

        if not self.posSatisfyLowLevelPlan(gameState):
            # self.lowLevelPlan = self.getLowLevelPlanQL(gameState, highLevelAction) #Generate low level plan with q learning
            self.lowLevelPlan = self.getLowLevelPlanHS(gameState, highLevelAction) #Generate low level plan with q learning
            # you can replace the getLowLevelPlanQL with getLowLevelPlanHS and implement heuristic search planner
            self.lowLevelActionIndex = 0
        lowLevelAction = self.lowLevelPlan[self.lowLevelActionIndex][0]
        self.lowLevelActionIndex+=1
        print("\tAgent:", self.index,lowLevelAction)
        return lowLevelAction

    #------------------------------- PDDL and High-Level Action Functions ------------------------------- 
    
    
    def getHighLevelPlan(self, objects, initState, positiveGoal, negtiveGoal) -> List[Tuple[Action,pddl_state]]:
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
        
        if self.pddl_solver.matchEffect(init_state, self.highLevelPlan[self.currentActionIndex][0] ):
            # The current state match the effect of current action, current action action done, move to next action
            if self.currentActionIndex < len(self.highLevelPlan) -1 and self.pddl_solver.satisfyPrecondition(init_state, self.highLevelPlan[self.currentActionIndex+1][0]):
                # Current action finished and next action is applicable
                self.currentActionIndex += 1
                self.lowLevelPlan = [] # reset low level plan
                return True
            else:
                # Current action finished, next action is not applicable or finish last action in the plan
                return False

        if self.pddl_solver.satisfyPrecondition(init_state, self.highLevelPlan[self.currentActionIndex][0]):
            # Current action precondition satisfied, continue executing current action of the plan
            return True
        
        # Current action precondition not satisfied anymore, need new plan
        return False
    
    def getGoals(self, objects: List[Tuple], initState: List[Tuple], gameState: GameState):
        # Check a list of goal functions from high priority to low priority if the goal is applicable
        # Return the pddl goal states for selected goal function
        myObj = "a{}".format(self.index)
        myCarrierGoal = False
        # 当前agent
        cur_agent_state = gameState.getAgentState(self.index)
        carrying_count = cur_agent_state.numCarrying
        
        # 检查是否携带了足够的食物
        for fact in initState:
            if len(fact) == 2 and fact[0] == "3_food_in_backpack" and fact[1] == myObj:
                myCarrierGoal = True
                break
        
        # ==================== 关键改进：时间感知 ====================
        timeRemaining = gameState.data.timeleft
        currentScore = self.getScore(gameState)
        
        # 获取当前位置到家的距离（用于判断是否来得及）
        myPos = gameState.getAgentPosition(self.index)
        border_positions = self.borderCoordinates if hasattr(self, 'borderCoordinates') else []
        if border_positions:
            dist_to_home = min(self.getMazeDistance(myPos, b) for b in border_positions)
        else:
            dist_to_home = 10  # 估计值
        
        # ==================== 优先级1: 时间紧迫策略 ====================
        if timeRemaining < 100:
            # 极度紧迫（<100步）
            if currentScore > 0:
                # 领先：立即回家，确保胜利
                print(f'Agent {self.index}: 时间<100且领先({currentScore}), 立即回家锁定胜利')
                # 判断当前agent是不是pacman
                if cur_agent_state.isPacman:
                    # 回家
                    positiveGoal = []
                    negtiveGoal = [("is_pacman", myObj)]
                    return positiveGoal, negtiveGoal
                else: # 巡逻
                    return self.goalDefWinning(objects, initState)
            elif currentScore < 0:
                # 落后：必须冒险拿1分
                if carrying_count > 0:
                    print(f'Agent {self.index}: 时间<100且落后({currentScore}), 携带{carrying_count}个，立即回家')
                    positiveGoal = []
                    negtiveGoal = [("is_pacman", myObj)]
                    return positiveGoal, negtiveGoal
                else:
                    print(f'Agent {self.index}: 时间<100且落后({currentScore}), 冒险抢最后1个食物')
                    return self.goalScoringAggressive(objects, initState)
            else:
                # 平局：抢1分即可
                if carrying_count > 0:
                    print(f'Agent {self.index}: 时间<100且平局, 携带{carrying_count}个，立即回家')
                    positiveGoal = []
                    negtiveGoal = [("is_pacman", myObj)]
                    return positiveGoal, negtiveGoal
                else:
                    print(f'Agent {self.index}: 时间<100且平局, 抢1个食物')
                    return self.goalScoringAggressive(objects, initState)
        
        elif timeRemaining < 250:
            # 比较紧迫（100-250步）
            if currentScore > 0:
                # 领先：保守，如果携带食物就回家
                if carrying_count > 0:
                    print(f'Agent {self.index}: 时间<250且领先({currentScore}), 携带{carrying_count}个回家')
                    positiveGoal = []
                    negtiveGoal = [("is_pacman", myObj)]
                    return positiveGoal, negtiveGoal
                else:
                    # 没携带食物，转入防守
                    print(f'Agent {self.index}: 时间<250且领先({currentScore}), 转入防守')
                    return self.goalDefWinning(objects, initState)
            elif currentScore < 0:
                # 落后：需要积极进攻，但携带2个以上就回家
                if carrying_count >= 2 or (carrying_count > 0 and dist_to_home > 15):
                    print(f'Agent {self.index}: 时间<250且落后({currentScore}), 携带{carrying_count}个且距离{dist_to_home}，回家')
                    positiveGoal = []
                    negtiveGoal = [("is_pacman", myObj)]
                    return positiveGoal, negtiveGoal
                else:
                    print(f'Agent {self.index}: 时间<250且落后({currentScore}), 继续进攻')
                    return self.goalScoring(objects, initState)
            else:
                # 平局：稍微激进
                if carrying_count >= 1:
                    print(f'Agent {self.index}: 时间<250且平局, 携带{carrying_count}个回家')
                    positiveGoal = []
                    negtiveGoal = [("is_pacman", myObj)]
                    return positiveGoal, negtiveGoal
        
        # ==================== 优先级2: 携带食物判断（降低阈值）====================
        # 时间充足时，携带3个回家
        # 时间紧迫时（已处理），降低阈值
        if myCarrierGoal:
            print(f'Agent {self.index}: 携带 >= 3 食物, 回家')
            positiveGoal = []
            negtiveGoal = [("is_pacman", myObj)]
            return positiveGoal, negtiveGoal
        
        # ==================== 优先级3: 大幅领先时防守 ====================
        if (("winning_gt10",) in initState):
            print(f'Agent {self.index}: winning_gt10, 去 Patrol')
            return self.goalDefWinning(objects, initState)
        
        # ==================== 优先级4: 没有食物时防守 ====================
        if ('food_available',) not in initState:
            print(f'Agent {self.index}: 没有 food_available, 所有agents都回家防守')
            myAgentsIndices = self.getTeam(gameState)
            positiveGoal = []
            negtiveGoal = []
            for i in myAgentsIndices:
                negtiveGoal += [("is_pacman", "a{}".format(i))]
            return positiveGoal, negtiveGoal
        
        # ==================== 优先级5: 默认进攻 ====================
        # 检查是否应该更激进（落后时）
        if currentScore < -3 and timeRemaining > 400:
            print(f'Agent {self.index}: 落后{abs(currentScore)}分且时间充足, 激进进攻')
            return self.goalScoringAggressive(objects, initState)
        else:
            print(f'Agent {self.index}: 正常进攻模式')
            return self.goalScoring(objects, initState)

    def goalScoring(self,objects: List[Tuple], initState: List[Tuple]):
        # If we are not winning more than 5 points,
        # we invate enemy land and eat foods, and bring then back.
        current_agent_obj = "a{}".format(self.index)
        positiveGoal = [('is_pacman', current_agent_obj), ('3_food_in_backpack', current_agent_obj)]
        negtiveGoal = [] # no food avaliable means eat all the food

        for obj in objects:
            agent_obj = obj[0]
            agent_type = obj[1]
            
            if agent_type == "enemy1" or agent_type == "enemy2":
                negtiveGoal += [("is_pacman", agent_obj)] # no enemy should standing on our land.
        
        return positiveGoal, negtiveGoal

    def goalScoringAggressive(self, objects: List[Tuple], initState: List[Tuple]):
        """
        激进进攻
        """
        myObj = "a{}".format(self.index)
        
        # 只要求携带食物，不管防守和消灭所有食物
        positiveGoal = [("is_pacman", myObj), ('3_food_in_backpack', myObj)]
        negtiveGoal = []
        
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

        walls = gameState.getWalls() # a 2d array matrix of obstacles, map[x][y] = true means a obstacle(wall) on x,y, map[x][y] = false indicate a free location
        foods = self.getFood(gameState) # a 2d array matrix of food,  foods[x][y] = true if there's a food.
        capsules = self.getCapsules(gameState) # a list of capsules
        foodNeedDefend = self.getFoodYouAreDefending(gameState) # return food will be eatan by enemy (food next to enemy)
        capsuleNeedDefend = self.getCapsulesYouAreDefending(gameState) # return capsule will be eatan by enemy (capsule next to enemy)
        myPos = gameState.getAgentPosition(self.index)
        if highLevelAction == 'attack':
            print(f"Agent {self.index} 执行 启发式【攻击】策略")
            return self._planAttack(gameState, myPos, walls)
        elif highLevelAction == 'go_home':
            print(f"Agent {self.index} 执行 启发式【回家】策略")
            return self._planGoHome(gameState, myPos, walls)
        elif highLevelAction == 'defence':
            print(f"Agent {self.index} 执行 启发式【防守】策略")
            return self._planDefence(gameState, myPos, walls)
        elif highLevelAction == 'patrol':
            print(f"Agent {self.index} 执行 启发式【巡逻】策略")
            return self._planPatrol(gameState, myPos, walls)
        else:
            return self._planAttack(gameState, myPos, walls)
        
    
    def posSatisfyLowLevelPlan(self,gameState: GameState):
        if self.lowLevelPlan == None or len(self.lowLevelPlan)==0 or self.lowLevelActionIndex >= len(self.lowLevelPlan):
            return False
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos,self.lowLevelPlan[self.lowLevelActionIndex][0])
        if nextPos != self.lowLevelPlan[self.lowLevelActionIndex][1]:
            return False
        return True
    
    #------------------------------- Heuristic search strategies Functions -------------------------------# ==================== Attack 策略 ====================
    def _planAttack(self, gameState: GameState, myPos: Tuple, walls) -> List[Tuple[str, Tuple]]:
        """进攻策略：找到最近的食物，同时避开敌人"""
        food_list = self.getFood(gameState).asList()
        ghosts = self.getGhostLocs(gameState)
        capsules = self.getCapsules(gameState)
        
        # 检查是否有scared ghost可以追杀
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        scared_ghosts = [
            e.getPosition() for e in enemies 
            if not e.isPacman and e.getPosition() is not None and e.scaredTimer > 5
        ]
        
        if scared_ghosts:
            closest_scared = min(scared_ghosts, key=lambda g: self.getMazeDistance(myPos, g))
            if self.getMazeDistance(myPos, closest_scared) <= 8:
                path = self._astar(myPos, closest_scared, walls, ghosts, avoid_ghosts=False)
                return path if path else self._getEmergencyMove(gameState, myPos, walls, ghosts)
        
        # 检查是否应该先吃胶囊
        if capsules and ghosts:
            closest_capsule = min(capsules, key=lambda c: self.getMazeDistance(myPos, c))
            closest_ghost_dist = min(self.getMazeDistance(myPos, g) for g in ghosts)
            capsule_dist = self.getMazeDistance(myPos, closest_capsule)
            
            if closest_ghost_dist <= 5 and capsule_dist <= 4:
                path = self._astar(myPos, closest_capsule, walls, ghosts, avoid_ghosts=True)
                return path if path else self._getEmergencyMove(gameState, myPos, walls, ghosts)
        
        # 正常进攻：找最近的食物
        if not food_list:
            return self._planGoHome(gameState, myPos, walls)
        
        # 分区策略
        teamIndices = self.getTeam(gameState)
        upperAgent = teamIndices[1] if self.red else teamIndices[0]
        midY = walls.height // 2
        
        if self.index == upperAgent:
            upper_food = [f for f in food_list if f[1] >= midY]
            target_foods = upper_food if upper_food else food_list
        else:
            lower_food = [f for f in food_list if f[1] < midY]
            target_foods = lower_food if lower_food else food_list
        
        # 找到最近的安全食物
        best_food = None
        best_score = float('inf')
        
        for food in target_foods[:10]:
            food_dist = self.getMazeDistance(myPos, food)
            danger_score = 0
            if ghosts:
                min_ghost_dist = min(self.getMazeDistance(food, g) for g in ghosts)
                danger_score = max(0, 8 - min_ghost_dist)
            
            total_score = food_dist + danger_score * 2
            if total_score < best_score:
                best_score = total_score
                best_food = food
        
        if best_food:
            path = self._astar(myPos, best_food, walls, ghosts, avoid_ghosts=True)
            return path if path else self._getEmergencyMove(gameState, myPos, walls, ghosts)
        
        return self._getEmergencyMove(gameState, myPos, walls, ghosts)


    def _planGoHome(self, gameState: GameState, myPos: Tuple, walls) -> List[Tuple[str, Tuple]]:
        """回家策略：找到最近的边境线"""
        ghosts = self.getGhostLocs(gameState)
        border_positions = self.borderCoordinates if hasattr(self, 'borderCoordinates') else []
        
        if not border_positions:
            mid_x = walls.width // 2
            if self.red:
                mid_x = mid_x - 1
            border_positions = [(mid_x, y) for y in range(walls.height) if not walls[mid_x][y]]
        
        # 找到最安全的边境点
        best_border = None
        best_score = float('inf')
        
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
            path = self._astar(myPos, best_border, walls, ghosts, avoid_ghosts=True, escape_mode=True)
            return path if path else self._getEmergencyMove(gameState, myPos, walls, ghosts)
        
        return self._getEmergencyMove(gameState, myPos, walls, ghosts)


    def _planDefence(self, gameState: GameState, myPos: Tuple, walls) -> List[Tuple[str, Tuple]]:
        """防守策略：拦截入侵者"""
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [e for e in enemies if e.isPacman and e.getPosition() is not None]
        
        if invaders:
            closest_invader = min(invaders, key=lambda e: self.getMazeDistance(myPos, e.getPosition()))
            target = closest_invader.getPosition()
            path = self._astar(myPos, target, walls, [], avoid_ghosts=False)
            return path if path else self._getEmergencyMove(gameState, myPos, walls, [])
        else:
            return self._planPatrol(gameState, myPos, walls)


    def _planPatrol(self, gameState: GameState, myPos: Tuple, walls) -> List[Tuple[str, Tuple]]:
        """巡逻策略：在己方领地边境巡逻"""
        import random
        
        border_positions = self.borderCoordinates if hasattr(self, 'borderCoordinates') else []
        
        if not border_positions:
            mid_x = walls.width // 2
            if self.red:
                mid_x = mid_x - 1
            border_positions = [(mid_x, y) for y in range(walls.height) if not walls[mid_x][y]]
        
        # 选择距离在5-15之间的巡逻点
        patrol_targets = [b for b in border_positions if 5 <= self.getMazeDistance(myPos, b) <= 15]
        
        if not patrol_targets:
            patrol_targets = border_positions
        
        target = random.choice(patrol_targets) if patrol_targets else border_positions[len(border_positions)//2]
        
        path = self._astar(myPos, target, walls, [], avoid_ghosts=False)
        return path if path else self._getEmergencyMove(gameState, myPos, walls, [])


    def _astar(self, start: Tuple, goal: Tuple, walls, ghosts: List[Tuple], 
            avoid_ghosts: bool = True, escape_mode: bool = False) -> List[Tuple[str, Tuple]]:
        """
        A*搜索算法
        返回: [(action, position), ...] - 从start到goal的路径
        """
        import heapq
        
        frontier = []
        heapq.heappush(frontier, (0, 0, start, []))
        visited = {start: 0}
        
        # Ghost危险区域
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
        
        max_iterations = 1000
        iterations = 0
        
        while frontier and iterations < max_iterations:
            iterations += 1
            _, g_score, current, path = heapq.heappop(frontier)
            
            if current == goal:
                return path
            
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                next_pos = Actions.getSuccessor(current, action)
                x, y = int(next_pos[0]), int(next_pos[1])
                
                if walls[x][y]:
                    continue
                
                move_cost = 1
                if next_pos in ghost_danger_zone:
                    move_cost += 50 if escape_mode else 10
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


    def _getEmergencyMove(self, gameState: GameState, myPos: Tuple, walls, ghosts: List[Tuple]) -> List[Tuple[str, Tuple]]:
        """应急方案：选择一个远离敌人的合法移动"""
        legal_actions = gameState.getLegalActions(self.index)
        legal_actions = [a for a in legal_actions if a != Directions.STOP]
        
        if not legal_actions:
            return [(Directions.STOP, myPos)]
        
        best_action = None
        best_score = -float('inf')
        
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
            # 逃回家时暂时禁用 stop TODO
            non_stop = [a for a in legalActions if a != Directions.STOP]
            if non_stop: legalActions = non_stop
            # The q learning process for escape actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getEscapeReward
            featureFunction = self.getEscapeFeatures
            weights = self.getEscapeWeights()
            learningRate = self.alpha # learning rate set to 0 as reward function not implemented for this action, do not do q update, 
        else:
            # 防守巡逻时暂时禁用 stop TODO
            non_stop = [a for a in legalActions if a != Directions.STOP]
            if non_stop: legalActions = non_stop
            # The q learning process for defensive actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getDefensiveReward
            featureFunction = self.getDefensiveFeatures
            weights = self.getDefensiveWeights()
            learningRate = self.alpha # learning rate set to 0 as reward function not implemented for this action, do not do q update 

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
            # 先临时处理下 TODO
            if feature not in weights:
                print('feature not in weights:', feature)
                weights[feature] = 0.0
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
        food_list = self.getFood(gameState).asList()
        walls = gameState.getWalls()
        base_reward =  -50 + nextAgentState.numReturned + nextAgentState.numCarrying
        new_food_returned = nextAgentState.numReturned - currentAgentState.numReturned
        score = self.getScore(nextState)

        if ghost_1_step > 0:
            base_reward -= 100 # 5 -> 100
        if score < 0:
            base_reward += score
        if new_food_returned > 0:
            # return home with food get reward score
            base_reward += new_food_returned*30
            
        if nextAgentState.getPosition() in food_list:
            base_reward += 30 # 吃食物给奖励
        
        # 装鬼惩罚
        if nextAgentState.getPosition() in ghosts:
            base_reward -= 150
            
        # 检查 *下一个* 状态的方向是否为 STOP
        if nextAgentState.configuration.direction == Directions.STOP:
            base_reward -= 100  # 停止不动惩罚
        if nextAgentState.configuration.direction == Directions.REVERSE:
            base_reward -= 30 # 来回摆动惩罚
        
        print("Agent ", self.index," reward ",base_reward)
        return base_reward
    
    def getDefensiveReward(self, gameState: GameState, nextState: GameState):
        """
        为 'defensive' (patrol) 动作计算奖励。
        目标: 拦截并吃掉入侵者。
        """
        
        # --- 1. 获取当前和下一步的状态信息 ---
        myCurrentState = gameState.getAgentState(self.index)
        myNextState = nextState.getAgentState(self.index)
        myCurrentPos = myCurrentState.getPosition()
        myNextPos = myNextState.getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invandersAround = [a for a in enemies if a.isPacman and a.getPosition() is not None]
        invandersAroundPos = [a.getPosition() for a in invandersAround]
        # 设置一个基础的“行动成本”，鼓励智能体更快地完成任务
        base_reward =  -50

        # --- 基础逻辑 ---
        if len(invandersAround) > 0: # 有入侵者
            # 每一步都在接近入侵者, 奖励
            for invader in invandersAround:
                current_dist_to_invader = self.getMazeDistance(myCurrentPos, invader.getPosition())
                next_dist_to_invader = self.getMazeDistance(myNextPos, invader.getPosition())
                if next_dist_to_invader < current_dist_to_invader:
                    base_reward += 5 * (current_dist_to_invader - next_dist_to_invader)
                else:
                    base_reward -= 5 # 远离入侵者惩罚
               
            # 吃掉入侵者奖励
            if myNextPos in invandersAroundPos:
                base_reward += 200
            
            # # 自己是scared状态, 被吃掉惩罚
            # if myCurrentState.scaredTimer > 0 and myNextPos == self.startPosition:
            #     base_reward -= 200
        # else: 
        #     # 没有入侵者，巡逻，尽量往边境线靠
        #     currentDistToBorder = self.getDistanceToBorder(myCurrentPos)
        #     nextDistToBorder = self.getDistanceToBorder(myNextPos)
        #     if nextDistToBorder < currentDistToBorder:
        #         base_reward += 5  # 离边境线更近了，很好
        #     elif nextDistToBorder > currentDistToBorder:
        #         base_reward -= 5  # 正在远离边境线，不好
                
        # 通用惩罚
        if myNextState.configuration.direction == Directions.STOP:
            base_reward -= 100  # 停止不动惩罚
        if myNextState.configuration.direction == Directions.REVERSE:
            base_reward -= 30 # 来回摆动惩罚

        # print(f"Agent {self.index} (Defensive) reward: {reward}") # 训练时可以取消注释来调试
        return base_reward
    
    def getEscapeReward(self,gameState, nextState):
        currentAgentState:AgentState = gameState.getAgentState(self.index)
        nextAgentState:AgentState = nextState.getAgentState(self.index)

        ghosts = self.getGhostLocs(gameState)
        ghost_1_step = sum(nextAgentState.getPosition() in Actions.getLegalNeighbors(g,gameState.getWalls()) for g in ghosts)
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        middle_x = width // 2
        middle_y = height // 2
        base_reward =  -50 
        # + nextAgentState.numReturned + nextAgentState.numCarrying
        new_food_returned = nextAgentState.numReturned - currentAgentState.numReturned

        if ghost_1_step > 0: # 逃跑时必须远离鬼，给稍大惩罚
            base_reward -= 10
        
        # if new_food_returned > 0:  # offensive里面已经有这个逻辑
        #     # 逃跑时身上肯定带了食物，重中之重是把食物送回家得分，所以回家带食物给大额奖励
        #     base_reward += new_food_returned * 50
            
        # 获取距离地图中心点的距离, 接近家给奖励
        curr_dist_to_border = self.getDistanceToBorder(currentAgentState.getPosition())
        next_dist_to_border = self.getDistanceToBorder(nextAgentState.getPosition())
        if next_dist_to_border < curr_dist_to_border:
            base_reward += 5
        else:
            base_reward -= 5
            
        # 检查 *下一个* 状态的方向是否为 STOP
        if nextAgentState.configuration.direction == Directions.STOP:
            base_reward -= 100  # 停止不动惩罚
        if nextAgentState.configuration.direction == Directions.REVERSE:
            base_reward -= -50 # 来回摆动惩罚

        
        print("Agent ", self.index," reward ",base_reward)
        return base_reward



    #------------------------------- Feature Related Action Functions -------------------------------


    
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
        features['successorScore'] = self.getScore(nextState)/(walls.width+walls.height)

        # Bias
        features["bias"] = 1.0
        
        # Get the location of pacman after he takes the action
        next_x, next_y = nextState.getAgentPosition(self.index)

        # Number of Ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts) 
        
        
        # 距离家的距离应该是过了边境线就算家
        # 检查下一个位置是否已经回到己方领地（即不再是Pacman身份）
        if not nextState.getAgentState(self.index).isPacman:
            dist_home = 0
        else:
            # 距离最近边境线位置的距离
            border_positions = self.borderCoordinates if hasattr(self, 'borderCoordinates') else []
            if border_positions:
                dist_home = min(self.getMazeDistance((next_x, next_y), border) for border in border_positions)
            # else:
            #     dist_home = self.getMazeDistance((next_x, next_y), gameState.getInitialAgentPosition(self.index))+1

        features["chance-return-food"] = (currAgentState.numCarrying)*(1 - dist_home/(walls.width+walls.height))# The closer to home, the larger food carried, more chance return food
                 
        teamIndices = self.getTeam(gameState)
        upperAgent = teamIndices[1] if self.red else teamIndices[0]
        midY = walls.height // 2
        food_list = self.getFood(gameState).asList()

        if self.index == upperAgent:
            # 上半区目标：只考虑上半边的食物
            upper_food = [p for p in food_list if int(p[1]) >= midY]
            if upper_food:
                dist = min(self.getMazeDistance((next_x, next_y), p) for p in upper_food)
            else:
                # 上半区没有食物时，退化为全局最近食物
                dist = self.closestFood((next_x, next_y), food, walls)
        else:
            # 下半区目标：只考虑下半边的食物
            lower_food = [p for p in food_list if int(p[1]) < midY]
            if lower_food:
                dist = min(self.getMazeDistance((next_x, next_y), p) for p in lower_food)
            else:
                # 下半区没有食物时，退化为全局最近食物
                dist = self.closestFood((next_x, next_y), food, walls)

        if dist is not None:
            features["closest-food"] = dist/(walls.width+walls.height)
        else:
            features["closest-food"] = 0
                
        # 新增撞上鬼的的feature, 如果撞上鬼，则惩罚
        if "crash-ghost" not in features:
            features["crash-ghost"] = 0
        if nextState.getAgentPosition(self.index) in ghosts:
            features["crash-ghost"] = 1.0
        else:
            features["crash-ghost"] = 0
        
        # stop的feature, 如果stop，则惩罚
        if action == Directions.STOP: 
            features['stop'] = 1
        else: 
            features['stop'] = 0

        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        nbrs_here = Actions.getLegalNeighbors(myPos, gameState.getWalls())
        is_deadend_here = (len(nbrs_here) == 1) # 如果只有一个邻居，则认为是死胡同, 允许回头
        features['reverse'] = 1 if (action == rev and not is_deadend_here) else 0
            
        # ============ 新增features ============
    
        # 1. 时间紧迫度 (归一化到 0-1)
        # timeRemaining = gameState.data.timeleft
        # if timeRemaining > 600:
        #     features['time-urgency'] = 0.0
        # elif timeRemaining > 300:
        #     features['time-urgency'] = 0.3
        # elif timeRemaining > 150:
        #     features['time-urgency'] = 0.6
        # else:
        #     features['time-urgency'] = 1.0
      
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
            # features['enemyDistance'] = min(dists)
            # norm
            walls = gameState.getWalls()
            features['enemyDistance'] = min(dists)/(walls.width + walls.height)
        
        # 计算当前位置到我方领地最近点的距离
        walls = gameState.getWalls()
        minHomeDist = float('inf')
        myX, myY = int(myPos[0]), int(myPos[1])
        width = walls.width
        height = walls.height

        isRed = self.red
        # 找所有属于本方领地的点（x小于宽度//2为红方，大于等于则为蓝方）
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
        features["distanceToHome"] = (minHomeDist /(walls.width + walls.height)) if minHomeDist != float('inf') else 0

        # 撞鬼惩罚
        if "crash-ghost" not in features:
            features["crash-ghost"] = 0.0
        ghosts = self.getGhostLocs(gameState)
        if myPos in ghosts:
            features["crash-ghost"] = 1.0
        else:
            features["crash-ghost"] = 0
        
        if action == Directions.STOP: 
            features['stop'] = 1
        else:
            features['stop'] = 0
        
        
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        nbrs_here = Actions.getLegalNeighbors(myPos, gameState.getWalls())
        is_deadend_here = (len(nbrs_here) == 1) # 如果只有一个邻居，则认为是死胡同, 允许回头
        features['reverse'] = 1 if (action == rev and not is_deadend_here) else 0
            
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
        features['onDefense'] = 1.0
        if myState.isPacman: features['onDefense'] = 0

        team = [successor.getAgentState(i) for i in self.getTeam(successor)]
        team_dist = self.getMazeDistance(team[0].getPosition(), team[1].getPosition())
        features['teamDistance'] = team_dist / (walls.width + walls.height)

        # --- 防守逻辑 ---
        
        # 1. 获取所有敌人的状态
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        
        # 2. 识别所有入侵者（无论可见与否）
        all_invaders = [a for a in enemies if a.isPacman]
        
        # 3. 识别可见的入侵者
        visible_invaders = [a for a in all_invaders if a.getPosition() != None]
        
        # 4. 识别不可见的入侵者 (获取他们的索引)
        # Fix: cannot use a.index (AgentState has no attribute 'index'), so map state back to index
        opponent_indices = self.getOpponents(successor)
        unseen_invader_indices = [
            opponent_indices[i] for i, a in enumerate(enemies)
            if a.isPacman and a.getPosition() is None   # 只要看不到的 Pacman（入侵者）
        ]
        
        # 5. 获取所有敌人的嘈杂距离
        noisy_distances = successor.getAgentDistances()

        features['numInvaders'] = len(all_invaders) / 2.0 # 归一化: 总入侵者数量
        
        min_dist_to_invader = float('inf')

        if len(visible_invaders) > 0:
            # [情况A] 至少有一个可见的入侵者：使用精确距离
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in visible_invaders]
            min_dist_to_invader = min(dists)
            
        elif len(unseen_invader_indices) > 0:
            # [情况B] 敌人不可见，但我们知道它在：使用嘈杂距离
            # 这只是一个估计值
            dists = [noisy_distances[i] for i in unseen_invader_indices]
            min_dist_to_invader = min(dists)
        
        
        if min_dist_to_invader != float('inf'):
            # [激活“追击”模式]
            # 只要我们知道有入侵者（无论是否可见），就激活 invaderDistance
            features['invaderDistance'] = min_dist_to_invader / (walls.width + walls.height)
            features['distanceToBorder'] = 0.0 # 关闭“巡逻边境”
        else:
            # [激活“巡逻”模式]
            # 确认没有入侵者，才去巡逻边境
            features['invaderDistance'] = 0.0
            distToBorder = self.getDistanceToBorder(myPos)
            features['distanceToBorder'] = distToBorder / (walls.width + walls.height)

        

        if action == Directions.STOP: 
            features['stop'] = 1.0
        else:
            features['stop'] = 0.0
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        nbrs_here = Actions.getLegalNeighbors(myPos, gameState.getWalls())
        is_deadend_here = (len(nbrs_here) == 1) # 如果只有一个邻居，则认为是死胡同, 允许回头
        features['reverse'] = 1 if (action == rev and not is_deadend_here) else 0
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
    
    def getDistanceToBorder(self, pos):
        """
        Calculate the maze distance from the given position 'pos' 
        to the closest point in 'self.borderCoordinates'
        """
        if not self.borderCoordinates:
            return 0 
            
        minDist = float('inf')
        for borderPos in self.borderCoordinates:
            dist = self.getMazeDistance(pos, borderPos)
            if dist < minDist:
                minDist = dist
        return minDist
    

