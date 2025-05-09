import math

# Helper functions
def hit_wall(x,y):
    return x>=1.85 or x<=-1.85 or y>=1.85 or y<= -1.85 or (x<=0.15 and x>=-0.15 and y<=0)

def dist(x1,y1, x2, y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)



# Single destination 1.5, 1.5
def single_destination(x,y):
    if hit_wall(x,y):
        return -50
    if dist(x,y,1.5,1.5)<=0.1:
        return 200
    return 0

# Two destinations 1.5, 1.5 and -1.5, -1.5
def two_destinations(x,y):
    if hit_wall(x,y):
        return -50
    if dist(x,y,1.5,1.5)<=0.1 or dist(x,y,-1.5,-1.5)<=0.1:
        return 100

# Move up. Reward is y if not hit_wall, linearly increasing with y.
def move_up(x,y):
    if hit_wall(x,y):
        return -50
    return y+1.85

# Move right. The reward logic is similar to move_up, but more complicated for the car since there is a wall in this direction.
def move_right(x,y):
    if hit_wall(x,y):
        return -50
    return x+1.85

# Travel in a dangerous zone within (0.1,0.3] near the wall
def stronger_storm_better_price(x,y):
    if hit_wall(x,y):
        return -50
    if x>=1.65 or x<=-1.65 or y>=1.65 or y<= -1.65 or (x<=0.35 and x>=-0.35 and y<=0.2):
        return 50

# Rewards the car when the back is pointing towards the closest surrounding wall of the maze (one of walls 1-4). Closer the car is to the wall higher the reward
def butt_scraper(x,y,theta):
    if hit_wall(x,y):
        return -50
    # Right
    if x>0 and abs(y)<x:
        return math.isclose(abs(theta),math.pi)*x
    # Left
    if x<0 and abs(y)<-x:
        return math.isclose(theta,0)*-x
    # Up
    if y>0 and abs(x)<y:
        return math.isclose(theta,-math.pi/2)*y
    
    return math.isclose(theta,math.pi/2)*-y