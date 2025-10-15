import math as m
import numpy as np


L1=10
L2=10

distance=5
#Height correction
corr=3
y=2
height=y-corr

r=(distance**2+height**2)**(1/2)
alpha=(180/np.pi)*(np.acos((L1**2+L2**2-r**2)/(2*L1*L2)))

#Angle of second motor from parallel to L1
phi=180-alpha

beta=(180/np.pi)*(np.atan(height/distance))
gamma=(180/np.pi)*(np.asin(L2*(np.sin(alpha))/r))



theta=beta+gamma

clawAngle=180-phi+theta

print(theta)
print(phi)
print(clawAngle)