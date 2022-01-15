from typing import NewType
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from os.path import exists
import pandas as pd

cmap = colors.ListedColormap(['black','white'])

def GenerateInitialState(N,M):
    """this function generates the initial
    spin state for a ising model system

    Args:
        N (int): number of vertical sites (number of rows)
        M (int): number of horizontal sites (number of columns)
    """
    State = 2*np.random.randint(2, size=(N,M))-1
    np.save('SavedObjects/InitialState.npy',State)
    return State

def Evolve(State,G,beta,h,a,b,number):
    """This function computes the posibility of
    generating a new state in the ising latice
    base on the Glauber dynamic

    Args:
        State (np array): current state of the latice
        G (np array): interaction parameter
        beta (float): 1/(KbT)
        h (float): magnetic field parameter
        a (int): x position of particle under consideration
        b (int): y position of particle under consideration
        number (int): numbered particle under consideration

    Returns:
        NewState (np array): new state of the latice
    """
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
    elif Energy2 - Energy1 >= 0 and np.random.rand() < np.exp(-(Energy2 - Energy1)*beta):
        NewState[a,b] *= -1
    else:
        pass
    return NewState

N = 30; M = 30
if exists('SavedObjects/InitialState.npy'):
    State0 = np.load('SavedObjects/InitialState.npy')
else:
    State0 = GenerateInitialState(N,M)

if exists('SavedObjects/InteractionTerm.npy'):
    G = np.load('SavedObjects/InteractionTerm.npy')
else:
    G = np.random.normal(size = int((N*M)**2.0))
    count, bins, ignored = plt.hist(G, 30, density=True)
    G = np.reshape(G,(N*M,N*M))
    G += G.T; G /= 2; G -= np.diag(np.diag(G))
    np.save('SavedObjects/InteractionTerm.npy',G)

NumberedState = np.arange(N*M).reshape(N,M)
Iteracion = 100000
H = np.linspace(-50,50,201)
Beta = np.linspace(0,400,801)
Results = {}
Cases = 1
for beta in Beta:
    for h in H:
        print(f'beta={beta}, h={h}')
        State = State0.copy()
        for i in range(Iteracion):
            particle = np.random.randint(M*N)
            a,b = np.where(NumberedState == particle)
            State = Evolve(State,G,3,0,a,b,particle)
        if np.abs(State.sum())>=300:
            Results[f'Caso {Cases}'] = {'beta':beta,'h':h,'Suma':State.sum(),'Cumulos':True}
        else:
            Results[f'Caso {Cases}'] = {'beta':beta,'h':h,'Suma':State.sum(),'Cumulos':False}

        fig, ax = plt.subplots(1)
        ax.imshow(State,cmap = cmap)
        plt.savefig(f'BetaImages/Caso{Cases}.png',bbox_inches = 'tight')
        plt.draw()
        plt.pause(5)
        plt.close()
        np.save(f'SavedObjects/CriticalBeta/FinalStateCase{Cases}.npy',State)
        Cases += 1
Results = pd.DataFrame(Results).T
Results.to_excel('Results/CriticalBeta.xlsx',engine = 'openpyxl')