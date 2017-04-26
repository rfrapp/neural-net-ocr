
import math, collections

class Blob(object):
	def __init__(self, x, y):
		self.minx = x
		self.maxx = x
		self.miny = y
		self.maxy = y
		self.cx = x
		self.cy = y

		# self.point_neighbors = collections.defaultdict(list)
		# self.point_neighbors[(x - 1, y)].append((x, y))
		# self.point_neighbors[(x, y - 1)].append((x, y))
		# self.point_neighbors[(x - 1, y - 1)].append((x, y))
		# self.point_neighbors[(x + 1, y)].append((x, y))
		# self.point_neighbors[(x, y + 1)].append((x, y))
		# self.point_neighbors[(x + 1, y + 1)].append((x, y))
		# self.point_neighbors[(x - 1, y + 1)].append((x, y))
		# self.point_neighbors[(x + 1, y - 1)].append((x, y))

		self.num_points = 0

	def add_point(self, x, y):
		# if len(self.point_neighbors[(x, y)]) >= 1:
		# 	self.point_neighbors[(x - 1, y)].append((x, y))
		# 	self.point_neighbors[(x, y - 1)].append((x, y))
		# 	self.point_neighbors[(x - 1, y - 1)].append((x, y))
		# 	self.point_neighbors[(x + 1, y)].append((x, y))
		# 	self.point_neighbors[(x, y + 1)].append((x, y))
		# 	self.point_neighbors[(x + 1, y + 1)].append((x, y))
		# 	self.point_neighbors[(x - 1, y + 1)].append((x, y))
		# 	self.point_neighbors[(x + 1, y - 1)].append((x, y))
		self.minx = min(x, self.minx)
		self.maxx = max(x, self.maxx)
		self.miny = min(y, self.miny)
		self.maxy = max(y, self.maxy)
		self.cx = (self.minx + self.maxx) / 2
		self.cy = (self.miny + self.maxy) / 2
		self.num_points += 1
		# 	return True
		# return False

	def dist_to(self, x, y):
		return math.sqrt((self.cx - x) ** 2 + (self.cy - y) ** 2)

	@property
	def rect(self):
		return ((self.minx, self.miny), (self.maxx, self.miny),
				(self.minx, self.maxy), (self.maxx, self.maxy))

	@property
	def h(self):
		return self.maxy - self.miny

	@property
	def w(self):
		return self.maxx - self.minx

	@property
	def aspect(self):
		return self.h / self.w if self.h < self.w else self.w / self.h

	@property
	def area(self):
		return self.w * self.h
