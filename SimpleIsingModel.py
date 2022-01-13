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

def Evolve(State,G,beta,h,a,b,number):
    N,M = np.shape(State)
    NewState = State
    Energy1 = 0
    Energy2 = 0
    for pos in [(-1,0),(0,-1),(1,0),(0,1)]:
        if not ((a+pos[0] ==-1 or a+pos[0] ==N) or (b+pos[1] ==-1 or b+pos[1] ==M)):
            numbersum = NumberedState[a+pos[0],b+pos[1]]
            Energy1 += beta*G[number,numbersum]*NewState[a,b]*NewState[a+pos[0],b+pos[1]]/np.sqrt(M*N)
            Energy2 += beta*G[number,numbersum]*(-NewState[a,b])*NewState[a+pos[0],b+pos[1]]/np.sqrt(M*N)
    Energy1 += h*NewState[a,b]
    Energy2 += h*(-NewState[a,b])
    if Energy2 - Energy1 < 0:
        NewState[a,b] *= -1
        print(1)
    elif Energy2 - Energy1 >= 0 and np.random.rand() < np.exp(-(Energy2 - Energy1)*beta):
        NewState[a,b] *= -1
        print(2)
    else:
        print(Energy2 - Energy1)
        print('No paso')
    return NewState

N = 30; M = 30
State0 = GenerateInitialState(N,M)
G = np.random.normal(size = int((N*M)**2.0))
count, bins, ignored = plt.hist(G, 30, density=True)
plt.plot(bins, 1/(1 * np.sqrt(2 * np.pi)) *np.exp( - (bins - 0)**2 / (2 * 1**2) ),linewidth=2, color='r')
plt.show()
G = np.reshape(G,(N*M,N*M))
G += G.T; G /= 2; G -= np.diag(np.diag(G))
fig, ax = plt.subplots(1)
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
    State = Evolve(State,G,3,0,a,b,particle)
    ax.imshow(State,cmap = cmap)
    plt.draw()
    plt.pause(0.01)
plt.close()