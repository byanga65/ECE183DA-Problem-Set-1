import numpy as np
import matplotlib.pyplot as plt
import random as rand
import time

# 0 is valid space, 1 is robot
# global variables: grid length and height, error probability
L = 5
H = 6
p_error = 0.01
discount = 0.9

def grid_create(l, h) :
    grid = np.zeros((l, h))
    return grid
# Debug Code, used to make sure the display isn't inverted ########################
# # def init_error() :
# #     p = np.random.random_sample()
# #     if p < p_error :
# #         return rand.randint(1, 4)
# #     else:
# #         return 0
# #
# #
# # def new_choice(choice):
# #     if choice == 1: return 'left'
# #     if choice == 2: return 'right'
# #     if choice == 3: return 'up'
# #     if choice == 4: return 'down'
# #     return 0
# #
# # def error_action(choice, robot) :
# #     move = new_choice(choice)
# #     return clean_action(move, robot)
# #
# # #Up , Down, Left, Right, None
# # def clean_action(param, robot) :
# #     new_x = robot[0]
# #     new_y = robot[1]
# #
# #     if param == 'left': new_x = new_x -1
# #     if param == 'right': new_x = new_x + 1
# #     if param == 'up': new_y = new_y+1
# #     if param == 'down': new_y = new_y-1
# #
# #     if new_x < 0 or new_x >= L: return robot
# #     if new_y < 0 or new_y >= H: return robot
# #
# #     new_robot = (new_x, new_y)
# #     return new_robot
#
# def action(param, robot) :
#     if param == 'none':  return robot
#
#     error_status = init_error()
#     if error_status > 0 :
#         return error_action(error_status, robot)
#     else:
#         return clean_action(param, robot)

# End Debug Code  #################################################

# 1A. Our state space size is L * H; in this case it is 30
# 1B. Our action space size is {Left, Right, Up, Down, None}, so it is 5

# 1C. No Obstacle Probability, given inputs state , action, next state. There is a 0.25* (error move probability chance) if the space is next to but a wrong action is chosen

def no_obs_prob(state, a, next_state):
    wrong_action_but_correct= 0.25 * p_error
    right_action_but_wrong = 0.75* p_error

    if a == 'none':
        if state == next_state: return 1
        else: return 0

    #left
    if next_state[0] == state[0] -1 and next_state[1] == state[1]:
        if a == 'left':
            return 1 - right_action_but_wrong
        else: return wrong_action_but_correct
    #right
    if next_state[0] == state[0] + 1 and next_state[1] == state[1]:
        if a == 'right':
            return 1 - right_action_but_wrong
        else: return wrong_action_but_correct
    #up
    if next_state[1] == state[1] + 1 and next_state[0] == state[0]:
        if a =='up':
            return 1 - right_action_but_wrong
        else:
            return wrong_action_but_correct
    #down
    if next_state[1] == state[1] - 1 and next_state[0] == state[0]:
        if a == 'down':
            return 1 - right_action_but_wrong
        else:
            return wrong_action_but_correct

    return 0

def setup(state_space) :
    state_space[4, :] = -100
    state_space[2, 0] = 10
    state_space[2, 2] = 5
    state_space[1, 3] = state_space[2, 3] = -1 * np.inf
    state_space[1, 1] = state_space[2, 1] = -1 * np.inf
    return state_space
def out_of_bounds_check(state):
    x = state[0]
    y = state[1]
    if x < 0 or y < 0 or x >= L or y >= H:
        return 1
    else: return 0

def obstacle(state):
    if state == (1, 3) or state == (2, 3) or state == (1, 1) or state == (2, 1) or out_of_bounds_check(state) == 1:
        return 1
    else: return 0

# 2a. Probability, including obstacles.
def obs_prob(state, a, next_state):
    if obstacle(next_state) == 1:
        return 0
    if obstacle(state) == 1:
        return 0

    return no_obs_prob(state, a, next_state)

# 2b. Rewards
def reward(state):
    if state[0] == 4: return -100
    if state == (2, 0): return 10
    if state == (2, 2): return 1
    if state == (1, 3) or state == (2, 3) or state == (1, 1) or state == (2, 1) :
        print('obstacle seen, this should never be reached')
        return -1* np.inf # some arbitrary value to never visit
    return 0

# Part 3a
def initial_policy() :
    policy_matrix = np.ones((L, H)) # Left is 1
    policy_matrix[1, 3] = policy_matrix[2, 3] = policy_matrix[1,1] = policy_matrix[2,1] = 5 # obstacle policies don't exist
    return policy_matrix

#Part 3b: Display a given policies and values.
def display_policy(policy_matrix) :
    state_space = grid_create(L, H)
    state_space = setup(state_space)
    plt.imshow(state_space.T, origin='lower', extent=[0, L, 0, H])
    value = plt.subplot()
    for i in range(L):
        for j in range(H):
            print_policy = 'empty'
            val = policy_matrix[i, j]
            if obstacle((i, j)) == 1:
                continue
            if val == 1 :
                print_policy = 'left'
            if val == 2 :
                print_policy = 'right'
            if val == 3 :
                print_policy = 'up'
            if val == 4 :
                print_policy = 'down'
            if val == 5 :
                print_policy = 'none'
            value.text(i+ 0.35, j+0.35, print_policy, color = 'r')
    return 0
def display_value(value_matrix) :
    state_space = grid_create(L, H)
    state_space = setup(state_space)
    plt.imshow(state_space.T, origin='lower', extent=[0, L, 0, H])
    value = plt.subplot()
    for i in range(value_matrix.shape[0]):
        for j in range(value_matrix.shape[1]):
            val = value_matrix[i, j]
            if obstacle((i, j)) == 1:
                continue
            value.text(i + 0.35, j + 0.35, '%.2f'%val, color='r')

# This helper function computes the sum over the possible next states, inside the policy evaluation step for a state
def policy_evaluate_helper(state, policy, previous_v_matrix)  :
    if obstacle(state) == 1:
        return 0

    x = state[0]
    y = state[1]
    left_state = (x-1, y)
    right_state = (x+1, y)
    up_state = (x, y+1)
    down_state = (x, y-1)

    # Default case: if movement into an obstacle, stay
    term_1 = term_2 = term_3 = term_4 = 0
    term_5 = obs_prob(state, policy, state) * (reward(state) + discount * previous_v_matrix[state])

    if obstacle(left_state) == 0:
        term_1 = obs_prob(state, policy, left_state) * (reward(left_state) + discount * previous_v_matrix[left_state])

    if obstacle(right_state) == 0:
        term_2 = obs_prob(state, policy, right_state) * (reward(right_state) + discount * previous_v_matrix[right_state])

    if obstacle(up_state) == 0:
        term_3 = obs_prob(state, policy, up_state) * (reward(up_state) + discount * previous_v_matrix[up_state])

    if obstacle(down_state) == 0:
        term_4 = obs_prob(state, policy, down_state) * (reward(down_state) + discount * previous_v_matrix[down_state])

    return term_1 + term_2 + term_3 + term_4 + term_5

# Using the above summation function, we compute the policy improvement for a state.
def policy_improve_state(state, value_matrix):
    val_left = policy_evaluate_helper(state, 'left', value_matrix)
    val_right = policy_evaluate_helper(state, 'right', value_matrix)
    val_up = policy_evaluate_helper(state, 'up', value_matrix)
    val_down= policy_evaluate_helper(state, 'down', value_matrix)
    val_none = policy_evaluate_helper(state, 'none', value_matrix)

    max_a = np.argmax([val_left, val_right, val_up, val_down, val_none]) + 1
    return max_a

# Part 3c: This function Compute policy evaluation for all states.
def one_policy_eval(value_matrix, policy_matrix):
    for l in range(0, L):
        for h in range(0, H):
            action = policy_matrix[l, h]
            policy = 'none'
            if action == 1:
                policy = 'left'
            if action == 2:
                policy = 'right'
            if action == 3:
                policy = 'up'
            if action == 4:
                policy = 'down'

            value_matrix[l, h] = policy_evaluate_helper((l, h), policy, value_matrix)
    return value_matrix

# Part 3d: This function compute policy improvement for all states
def one_policy_improve(value_matrix, policy_matrix):
    for l in range(0, L):
        for h in range(0, H):
            policy_matrix[l, h] = policy_improve_state((l, h), value_matrix)
    return policy_matrix

# Part 3e: Displaying the optimal policy and its runtime until convergence.
def policy_iteration():
 # plot the grid
 #    state_space = grid_create(L, H)
 #    state_space = setup(state_space)
 #    plt.imshow(state_space.T, origin='lower', extent=[0, L, 0, H])

# MDP Policy Iteration: Runs until convergence, with a value of 1e-10
    start_time = time.time()
    value = np.zeros((L, H))
    policy = initial_policy()
    i = 0

    while True:
        original_value = np.linalg.norm(value)
        value = one_policy_eval(value, policy)
        policy = one_policy_improve(value, policy)
        new_value = np.linalg.norm(value)
        i += 1
        if np.abs(new_value- original_value) < 1e-10 :
            print(i)
            print("Part 3f: --- %s seconds ---" % (time.time() - start_time))
            break

    display_policy(policy)
    return 0

# This helper function is for summing in the value iteration
def value_helper(state, value_matrix):
    val_left = policy_evaluate_helper(state, 'left', value_matrix)
    val_right = policy_evaluate_helper(state, 'right', value_matrix)
    val_up = policy_evaluate_helper(state, 'up', value_matrix)
    val_down= policy_evaluate_helper(state, 'down', value_matrix)
    val_none = policy_evaluate_helper(state, 'none', value_matrix)

    max_a = max([val_left, val_right, val_up, val_down, val_none])
    return max_a

def one_value_iteration(value_matrix):
    for l in range(L):
        for h in range(H):
            value_matrix[l, h] = value_helper( (l, h), value_matrix)
    return value_matrix

# Part 4a: Value Iteration
def value_iteration():
    value = np.zeros((L, H))
    # state_space = grid_create(L, H)
    # state_space = setup(state_space)
    # plt.imshow(state_space.T, origin='lower', extent=[0, L, 0, H])

    start_time = time.time()
    i = 0
    while True:
            original_value = np.linalg.norm(value)
            value = one_value_iteration(value)
            new_value = np.linalg.norm(value)
            i += 1
            if np.abs(new_value - original_value) < 1e-10:
                print(i)
                print("Part 4c: --- %s seconds ---" % (time.time() - start_time))
                break

    display_value(value)
    plt.show()
    return 0

policy_iteration()
plt.xlabel('Error probability : %.2f, Discount: %.2f '% (p_error, discount))
plt.show()
#value_iteration()




