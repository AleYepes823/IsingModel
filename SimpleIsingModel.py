from typing import NewType
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

cmap = colors.ListedColormap(['black','white'])

def GenerateInitialState(N,M):
    """this function generates the initial
    spin state for a ising model system

    Args:
        N (int): number of vertical sites (number of rows)
        M (int): number of horizontal sites (number of columns)
    """
    State = 2*np.random.randint(2, size=(N,M))-1
    return State

def Evolve(State,J,beta,a,b,number):
    N,M = np.shape(State)
    NewState = State
    Energy = 0
    for pos in [(-1,0),(0,-1),(1,0),(0,1)]:
        if not ((a+pos[0] ==-1 or a+pos[0] ==N) or (b+pos[1] ==-1 or b+pos[1] ==M)):
            numbersum = NumberedState[a+pos[0],b+pos[1]]
            Energy -= J[number,numbersum]*NewState[a,b]*NewState[a+pos[0],b+pos[1]]/np.sqrt(M*N)
    print(Energy)
    if Energy < 0:
        NewState[a,b] *= -1
    if Energy >= 0 and np.random.rand() < np.exp(-Energy*beta):
        NewState[a,b] *= -1
    return NewState




fig, ax = plt.subplots(1)
N = 30; M = 30
State0 = GenerateInitialState(N,M)
J = np.ones(shape = (N*M,N*M))*2
J -= np.diag(np.diag(J))
ax.imshow(State0,cmap = cmap)
plt.draw()
plt.pause(0.01)
NumberedState = np.arange(N*M).reshape(N,M)
Iteracion = 50000
State = State0.copy()
for i in range(Iteracion):
    particle = np.random.randint(M*N)
    a,b = np.where(NumberedState == particle)
    ax.clear()
    fig.suptitle(f'Time = {i}')
    State = Evolve(State,J,0,a,b,particle)
    ax.imshow(State,cmap = cmap)
    plt.draw()
    plt.pause(0.01)
plt.close()