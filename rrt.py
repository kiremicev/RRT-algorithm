from random import randrange, randint, sample, shuffle
import math
from typing import Tuple, List, Dict, Any
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon, LineString, LinearRing, Point
from shapely.plotting import plot_line, plot_points, plot_polygon
import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork

# Ignorie unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

class RRT:

	def __init__(self, 
				# Parameters
				obstacles: List[BaseGeometry], # obstacles list (shapely)
				start_shape: BaseGeometry, # start shape (shapely)
				goal_shape: BaseGeometry, # goal shape (shapely)
				lanelet_network: LaneletNetwork, # lanelet network (commonroad.scenario.lanelet)
				psi0: float, # initial orientation
				dist = 4, # distance between vertices
				dist_max_goal = 100, # maximum distance between last vertex and goal
				deltaAngle_max = 20, # maximum angle in between path vertices [deg]
				deltaAngle_max_goal = 40, # maximum angle in between last vertex and goal [deg]
				max_iter = 10000, # maximum iterations
				max_failure = 20000, # maximum number of allowed consecutive failures
				w_half = 1.6, # half width of car (for inflating the obstacles)
				turning_start_distance = 6): # distance from node when the car is expected to start turning

		self.obstacles = obstacles
		self.start_shape = start_shape
		self.goal_shape = goal_shape
		self.lanelet_network = lanelet_network
		self.psi0 = psi0
		self.dist = dist
		self.dist_max_goal = dist_max_goal
		self.deltaAngle_max = deltaAngle_max
		self.deltaAngle_max_goal = deltaAngle_max_goal
		self.max_iter  = max_iter
		self.max_failure = max_failure
		self.w_half = w_half
		self.turning_start_distance = turning_start_distance

		# Define window size of the whole RRT problem
		self.x_range: float = None
		self.y_range: float = None
		self.window_size()

		# Container of points
		self.pointsContainer: List[Tuple[float,float]] = None

		# Parents of points
		self.parents: Dict[Tuple[float, float], Any] = None

		# Start and goal coordinates
		self.start = tuple(start_shape.centroid.coords)[0] # start
		self.start = (int(np.round(self.start[0])),int(np.round(self.start[1])))
		self.goal = tuple(goal_shape.centroid.coords)[0] # goal
		self.start = (int(np.round(self.start[0])),int(np.round(self.start[1])))

		# Goal shape bounds rounded
		self.goal_bounds = self.goal_shape.bounds
		goal_x_bounds = self.bounds_round_outward(self.goal_bounds[0],self.goal_bounds[2])
		goal_y_bounds = self.bounds_round_outward(self.goal_bounds[1],self.goal_bounds[3])

		# List of int goal points within goal
		self.x_goal_list = []
		self.y_goal_list = []
		for x in range(goal_x_bounds[0],goal_x_bounds[1]+1):
			for y in range(goal_y_bounds[0],goal_y_bounds[1]+1):
				if Point(x,y).intersects(goal_shape):
					self.x_goal_list.append(x)
					self.y_goal_list.append(y)

		# Success
		self.success = False

		# Path
		self.path: List[Tuple[float,float]] = None

	def update(self, 
				obstacles: List[BaseGeometry],
				start_shape: BaseGeometry, 
				goal_shape: BaseGeometry,
				psi0: float):
		self.obstacles = obstacles
		self.start_shape = start_shape
		self.goal_shape = goal_shape
		self.psi0 = psi0

	def bounds_round_inward(self, a: float, b: float):
		if not b > a:
			print('Error: Wrong order of a and b!')
			return None
		if a<0:
			a_rounded = int(a)
		if a>0:
			a_rounded = int(a+1)
		if b<0:
			b_rounded = int(b-1)
		if b>0:
			b_rounded = int(b)
		return (a_rounded,b_rounded)

	def bounds_round_outward(self, a: float, b: float):
		if not b > a:
			print('Error: Wrong order of a and b!')
			return None
		if a<0:
			a_rounded = int(a-1)
		if a>0:
			a_rounded = int(a)
		if b<0:
			b_rounded = int(b)
		if b>0:
			b_rounded = int(b+1)
		return (a_rounded,b_rounded)

	def window_size(self):
		x_min_bounds = []
		y_min_bounds = []
		x_max_bounds = []
		y_max_bounds = []

		geometric_objects: List[BaseGeometry] = []
		geometric_objects.extend(self.obstacles)
		geometric_objects.extend([self.start_shape,self.goal_shape])
		
		for geometric_object in geometric_objects:
			bounds = geometric_object.bounds
			x_min_bounds.append(bounds[0])
			y_min_bounds.append(bounds[1])
			x_max_bounds.append(bounds[2])
			y_max_bounds.append(bounds[3])

		self.x_range, self.y_range = ((int(min(x_min_bounds)), 
									int(max(x_max_bounds))),
									(int(min(y_min_bounds)),
									int(max(y_max_bounds))))

	def distance(self, p1, p2):
		return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

	def randomFeasibleAngleLengthPoint(self):
		while True:
			parent = self.pointsContainer[randrange(0,len(self.pointsContainer))]
			l_abs = self.dist
			if parent == self.start:
				alpha1 = randint(-self.deltaAngle_max,self.deltaAngle_max)/180*np.pi
				p1a = parent
				p1b = (int(p1a[0] + l_abs*np.cos(self.psi0+alpha1)), int(p1a[1] + l_abs*np.sin(self.psi0+alpha1)))
				sample = p1b
				if (self.x_range[0]<sample[0]<self.x_range[1] and 
					self.y_range[0]<sample[1]<self.y_range[1] and
					sample not in self.pointsContainer):
					return(parent,sample)
				continue
			l1 = (self.parents[parent],parent)
			dx1 = l1[1][0] - l1[0][0]
			dy1 = l1[1][1] - l1[0][1]
			alpha1 = np.arctan2(dy1,dx1)
			alpha2 = randint(-self.deltaAngle_max,self.deltaAngle_max)/180*np.pi
			p2a = l1[1]
			p2b = (int(p2a[0] + l_abs*np.cos(alpha1+alpha2)), int(p2a[1] + l_abs*np.sin(alpha1+alpha2)))
			sample = p2b
			if (self.x_range[0]<sample[0]<self.x_range[1] and 
				self.y_range[0]<sample[1]<self.y_range[1] and
				sample not in self.pointsContainer):
				return (parent,sample)
			continue

	def randomGoal(self):
		point_goal = (self.x_goal_list[randrange(0,len(self.x_goal_list))],
					  self.y_goal_list[randrange(0,len(self.y_goal_list))])
		return point_goal

	def deltaAngleLines(self,
						line1: Tuple[Tuple[float,float],Tuple[float,float]], 
						line2: Tuple[Tuple[float,float],Tuple[float,float]]):
		dx1 = line1[1][0] - line1[0][0]
		dx2 = line2[1][0] - line2[0][0]
		dy1 = line1[1][1] - line1[0][1]
		dy2 = line2[1][1] - line2[0][1]
		phi1 = np.arctan2(dy1,dx1)
		phi2 = np.arctan2(dy2,dx2)
		return (abs(phi1-phi2)*180/np.pi)%360

	def collision_line(self, line: Tuple[Tuple[float,float],Tuple[float,float]]):
		linestring = LineString(coordinates=[line[0],line[1]]).buffer(self.w_half)
		for obstacle in self.obstacles:
			if obstacle.intersects(linestring):
				return True
		return False

	def path_extractor(self):
		if not self.success:
			return None
		current = self.goal
		self.path = [current]
		while self.parents[current]:
			current = self.parents[current]
			self.path.insert(0,current)
		return self.path

	def rrt(self):
		
		# Initialization
		self.parents: Dict[Tuple[float, float], Any] = { self.start: None }
		self.pointsContainer = [self.start]
		current = self.start

		n = 0
		n_failure = 0

		while True:

			if n >= self.max_iter:
				print('Maximum number of iterations reached')
				print('Iterations:', n)
				print('Failures:', n_failure)
				self.success = False
				return self.success, self.path_extractor()

			if n_failure >= self.max_failure:
				print('Maximum number of failures reached')
				print('Iterations:', n)
				print('Failures:', n_failure)
				self.success = False
				return self.success, self.path_extractor()

			# Create random sample point with feasible angle and length w.r.t. a given parent
			parent, sample = self.randomFeasibleAngleLengthPoint()

			# Test if points are feasible w.r.t. obstacles
			if parent == sample:
				n_failure += 1
				continue
			if self.collision_line(line=(sample,parent)):
				n_failure += 1
				continue
			
			# Sample is fesible
			self.pointsContainer.append(sample)
			self.parents[sample] = parent
			current = sample
			n += 1

			self.goal = self.randomGoal()

			if (not self.collision_line(line=(current,self.goal)) and 
				self.distance(current, self.goal) < self.dist_max_goal):
				if self.parents[current]:
					line_parent = (self.parents[current],current)
					line_child = (current,self.goal)
					if self.deltaAngleLines(line1=line_parent,line2=line_child) < self.deltaAngle_max_goal:
						break
				else:
					break
		
		if self.goal not in self.parents:
			self.parents[self.goal] = current

		self.success = True
		print('RRT successfully finished. Path is found.')
		print('Iterations:', n)
		print('Failures:', n_failure)
		return self.success, self.path_extractor()

	def plot_rrt_path(self, plot_whole_tree: bool = False):
		
		plot_polygon(polygon=self.start_shape, add_points=False, color=(0,1,0))
		plot_polygon(polygon=self.goal_shape, add_points=False, color=(0,1,0))

		for obstacle in self.obstacles:
			if isinstance(obstacle,Polygon):
				plot_polygon(polygon=obstacle, add_points=False)
			if isinstance(obstacle,LinearRing) or isinstance(obstacle,LineString):
				plot_line(line=obstacle,add_points=False)
			if isinstance(obstacle,Point):
				plot_points(geom=obstacle)
		if self.path:
			if plot_whole_tree:
				parents_without_start = self.parents
				parents_without_start.pop(self.start)
				for child in parents_without_start:
					line = LineString(coordinates=[child,parents_without_start[child]])
					plot_line(line=line,add_points=False,linewidth=0.5)
			linestring_path = LineString(coordinates=self.path)
			plot_line(line=linestring_path, color=(1,0.5,0,0.5))
		else:
			if plot_whole_tree:
				parents_without_start = self.parents
				parents_without_start.pop(self.start)
				for child in parents_without_start:
					line = LineString(coordinates=[child,parents_without_start[child]])
					plot_line(line=line,add_points=False,linewidth=0.5)
			pass