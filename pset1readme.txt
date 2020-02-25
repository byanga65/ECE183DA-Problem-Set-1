Problem Set 1 Readme 

0a) Team Hermes: Bill, Aarranon, Harish

0b) Pieter Abbeel's slides and lecture videos

1a) Our state space is L* H - (number of obstacles). In this case, it is 26. 

1b) Our action space is {left, right, up, down, none}.

1c) The function no_obs_prob calculates the probability, assuming no obstacles. 

2a) The function obs_prob calculates the probability, using no_obs_prob.

2b) The function reward returns a reward for each state, and makes sure obstacles don't have a legit value. 

3a) The function initial_policy creates a 2D matrix, with all 1 values. Obstacles have the value 5 as a throwaway.

3b) The function display_policy shows the map, with an overlay of the policy action.

3c) The function one_policy_eval computes a single policy evaluation. It relies on the helper functions policy_evaluate_helper. 

3d) The function policy_improve_state computes a single policy improvement (the bellman backup). It relies on the helper function policy_evaluate_helper.

3e) The function policy_iteration iterates until it converges at an optimal policy, and displays it. The optimal policy is shown in "3e.png"

3f) The run time is 0.601 seconds, found using python. 

3g) The trajectory is in file "3g.png". The total discounted sum of rewards is 0.9^7 * 10 = 4.78.
	The expected discounted sum of rewards is (100-51.27) = 48.73, which is similar in magnitude if divided by 10. We are unsure where this came in.

4a) The function value_iteration does value_iteration until convergence, using the helper functions one_value_iteration and value_helper. Since it uses the same math,
	value_helper also uses policy_evaluate_helper. 

4b) The optimal policy is shown in image "4b.png" with the trajectory and computed results same as policy iteration above. 

4c) Using the same timing method, it takes 0.577 seconds.

5a) Various plots are in the folder (additional scenarios). 
	The original, with low error and discount 0.9 results in taking the right path, +10 reward.
	"Case1.png" : low error, discount 0.1 results in taking the left path, +1 reward. 
	"Case2.png" : mid error, discount 0.9 results in taking the left path, +10 reward.
	"Case3.png" : no error, discount 0.2 results in taking the right path, +1 reward.
	
	Increasing the error probability causes the path to avoid the -100 reward, and go left.  
	Making the discount factor closer to 0 causes the path to pursue the +1 reward instead.
	The right path is pursued only with 0 or very low error, else the left path is pursued. 
	

 
