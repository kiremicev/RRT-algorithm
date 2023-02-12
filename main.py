from rrt import *
import pickle
import time
from shapely.affinity import translate
import matplotlib.pyplot as plt

# Import necessary files
psi0 = 1
with open('obstacles.pkl', 'rb') as f:
	obstacles: List[BaseGeometry] = pickle.load(f)
with open('start.pkl', 'rb') as f:
	start_shape: BaseGeometry = pickle.load(f)
with open('goal.pkl', 'rb') as f:
	goal_shape: BaseGeometry = pickle.load(f)
with open('lanelet_network.pkl', 'rb') as f:
	lanelet_network: LaneletNetwork = pickle.load(f)

# Insert more obstacles and translate goal to another position
goal_shape = translate(goal_shape,5,-80)
obstacles.append(LineString(coordinates=[(-10,30),(-6,20)]))
obstacles.append(Point(5,30).buffer(2))
obstacles.append(LineString(coordinates=[(-2,0),(-12,14)]))

# Parameters
dist = 4 # distance between vertices
dist_max_goal = 100 # maximum distance between last vertex and goal
deltaAngle_max = 20 # maximum angle in between path vertices [deg]
deltaAngle_max_goal = 40 # maximum angle in between last vertex and goal [deg]
max_iter = 10000 # maximum iterations
max_failure = 20000 # maximum number of allowed consecutive failures
w_half = 1.6 # half width of car (for inflating the obstacles)
turning_start_distance = 2 # distance from node when the car is expected to start turning

start_time = time.time()

# RRT initialization
rrt = RRT(obstacles,
			start_shape,
			goal_shape,
			lanelet_network,
			psi0,
			dist,
			dist_max_goal,
			deltaAngle_max,
			deltaAngle_max_goal,
			max_iter,
			max_failure,
			w_half,
			turning_start_distance)

# Run algorithm with a given maximum number of tries
m_max = 3
m = 0
print('-------------------------')
print('  RRT algorithm started')
print('-------------------------')
while True:
	success, path = rrt.rrt()
	if success:
		break
	m += 1
	if not m < m_max:
		print('Maximum number of retries is reached. No path is found.')
		break
	print('-------------------------')
	print('  Retry RRT algorithm')
	print('-------------------------')

end_time = time.time()

print('Path:', path)
print('Time [s]:', (end_time-start_time))

# RRT plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis('equal')
rrt.plot_rrt_path(plot_whole_tree=False)
plt.show()