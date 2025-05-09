import math

def reward(x,y):
    if x>=1.85 or x<=-1.85 or y>=1.85 or y<= -1.85:
        return -50
    if x<=0.15 and x>=-0.15:
        if y<=0:
            return -50
    if dist(x,y)<=0.1:
        return 200
    return 0
    
def dist(x,y):
    return math.sqrt((x-1.5)**2+(y-1.5)**2)