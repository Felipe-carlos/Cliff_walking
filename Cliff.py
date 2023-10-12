# Título: Trabalho 2 - Cliff Walking
# Autor: Felipe C. dos Santos
#
import numpy as np
import matplotlib . pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

n = 4   #dimensão do gridworld
m = 12  #dimensão do gridworld
inicio = (n-1,0) #inicio do gridwolrd
final_state = (n-1,m-1) # estados finais com reward 1
cliff = [(n-1, x) for x in range(1, m-1)] #definição da posição do penhasco
gama = 1 #undiscouted
epsilon = 0.1 #e-greedy policy
alpha = 0.5 #atualização do q-learning
episodes = 501

def init_states(n,m):
    states = []
    for i in range(n):
        for j in range(m):
            states.append((i, j))
    return states
def reward(s): #implementação da função reward, recebe s que é a posição que o agente esta
    if s == final_state:
        R = 0
    elif s in cliff:
        R=-100
    else:
        R=-1
    return R

def take_action(s,a):     #recebe a posição de 2 no gridwolrd e a string da ação e eretorna a posição futura após aquela ação
    x = s[0]
    y = s[1]
    fall = False

    if (x,y) in cliff:
        x_new = inicio[0]
        y_new = inicio[1]
        fall = True
    else:
        if a == 'up':
            x_new = max(0, x - 1)
            y_new = y
        if a == 'down':
            x_new = min(n - 1, x + 1)
            y_new = y
        if a == 'left':
            x_new = x
            y_new = max(0, y - 1)
        if a == 'right':
            x_new = x
            y_new = min(m - 1, y + 1)

    return (x_new ,y_new),fall

def init_qsa(states):   #inicia a função q com zeros para todas as açoes e posições
    qsa = dict()
    for s in states:
        qsa[s] = {'up':0,
                  'down': 0,
                  'left': 0,
                  'right': 0,
                  }
    return qsa

def egreedy_policy(q,s,episilon):   #recebe o dicionario q, a tuple s e o float episilon e retorna a chave de uma ação
    if np.random.random() <episilon:
        return np.random.choice(list(q[s].keys()))
    else:
        return max(q[s],key=q[s].get)

def q_learning(qsa,episodes):
    history = []                #salva todas as posiçoes que o agente passou
    caminho_anim = dict()       #salva o caminho e as recompensas em determinados episósios
    caminho =[(inicio,0,1)]       #vai acumulando posição, recompensa em determidados episódios

    for k in range(episodes):
        s = inicio                                   #coloca o agente no inicio
        history.append(s)

        while s != final_state:                             # repete até chegar ao fim da grid
            action = egreedy_policy(qsa,s,epsilon)          #escolhe a ação usando egreedy
            next_state,falling = take_action(s,action)      #diz a proxima ação e secair no penhasco falling = true
            R_next = reward(next_state)              #recompensa da próxima transição
            best_next_action = max(qsa[next_state],key=qsa[next_state].get)     #descobre a melhor ação, independente do egreedy


            if k in [1 ,50,500 ]:                   #salva posição e recompensa para gerar a animação dos episódios 1 e 50
                caminho.append((next_state,R_next,k))
                if next_state == final_state:
                    caminho_anim[f'caminho_k={k}'] = caminho
                    caminho=[(inicio,0,k)]

            qsa[s][action] += alpha*(R_next + gama*qsa[next_state][best_next_action] - qsa[s][action]) #atualiza a função Q usando Q-learning
            history.append(next_state)
            history.append(falling)
            s = next_state      #atualiza a posição do agente

    return qsa, history,caminho_anim

def sarsa(qsa,episodes):
    history = []                #salva todas as posiçoes que o agente passou
    caminho_anim = dict()       #salva o caminho e as recompensas em determinados episósios
    caminho =[(inicio,0,1)]       #vai acumulando posição, recompensa em determidados episódios


    for k in range(episodes):
        s = inicio                                   #coloca o agente no inicio
        history.append(s)
        action = egreedy_policy(qsa,s,epsilon)

        while s != final_state:                             # repete até chegar ao fim da grid
            next_state,falling = take_action(s,action)      #diz a proxima ação e secair no penhasco falling = true
            R_next = reward(next_state)              #recompensa da próxima transição
            sarsa_action = egreedy_policy(qsa,next_state,epsilon)

            if k in [1 ,50, 500 ]:                   #salva posição e recompensa para gerar a animação dos episódios 1 50 e 500
                caminho.append((next_state,R_next,k))
                if next_state == final_state:
                    caminho_anim[f'caminho_k={k}'] = caminho
                    caminho=[(inicio,0,k)]

            qsa[s][action] += alpha*(R_next + gama*qsa[next_state][sarsa_action] - qsa[s][action]) #atualiza a função Q usando sarsa
            history.append(next_state)
            history.append(falling)
            s = next_state      #atualiza a posição do agente
            action = sarsa_action

    return qsa, history,caminho_anim
def plot_heatmap(historioco): # plota heatmap da quantidade de vezes o agente esteve em cada ponto
    matrix = np.zeros((n, m))
    count = 0
    for s in historioco:
        if s == True:       #conta o numero de quedas
            count+=1
        else:               #aumenta o numero de vezes que o agente passa por s
            matrix[s] += 1

    plt.imshow(matrix, cmap='YlOrRd')
    plt.colorbar()
    plt.title('Posição do agente ao longo dos episódios', fontsize=15)
    plt.text(1, 5, f'Total de {count} quedas e {episodes} episodios',fontsize=12 )
    #plt.savefig(f'imagens/gama={gama}_epis={epsilon}_alfa={alpha}.png')
    print('quantidade de vezes que o agente passa em cada estado:\n',matrix)

def plot_position(info):  #recebe as informações em forma de tuplas ((s,R),k) o par (s,R) em que s é uma tupla com a posição e R a recompensa
    k = info[2]
    s = info[0]
    R = info[1]
    ax.clear()
    mundo=np.zeros((n,m))
    for i in cliff:
        mundo[i] = -5       #valores para obter as cores desejadas
    mundo[s] = -1           #valores para obter as cores desejadas
    plt.imshow(mundo,cmap='inferno')
    plt.title(f"Posição do agente ao longo do episódio {k}")
    plt.text(s[1], s[0], str(R), ha='center', va='center', color='w')   #Printa a recompensa de cada local
    plt.text((m-2)/2, n-1, 'Cliff', ha='center', va='center', color='w',fontsize= 16)   #eescreve cliff

def animation(frames,k,fig):              #cria um gif com a evolução do resultado no tempo

    anim = FuncAnimation(fig,plot_position, frames=frames)
    plt.show()
    #anim.save(f'Animações/caminho_k={k}_gama={gama}_epis={epsilon}_alfa={alpha}.gif', dpi=300, writer=PillowWriter(fps=8))

def animation_comparacao(frames,k,fig):              #cria um gif com a evolução do resultado no tempo para comparação q-learing e sarsa

    anim = FuncAnimation(fig,comparacao, frames=frames)
    anim.save(f'Animações/Compcaminho_k={k}_gama={gama}_epis={epsilon}_alfa={alpha}.gif', dpi=300, writer=PillowWriter(fps=10))
def comparacao(info_q_and_sarsa):   #plota posição para comparação q-learing e sarsa
    k = info_q_and_sarsa[0][2]
    s_q = info_q_and_sarsa[0][0]
    s_sarsa = info_q_and_sarsa[1][0]
    ax.clear()
    mundo = np.zeros((n, m))
    for i in cliff:
        mundo[i] = -5       #valores para obter as cores desejadas
    if s_q == s_sarsa:
        mundo[s_q]=-3
        plt.text(s_q[1], s_q[0], 'Q&S', ha='center', va='center', color='w')  # Printa a recompensa de cada local
    else:
        plt.text(s_q[1], s_q[0], 'Q', ha='center', va='center', color='w')  # Printa a recompensa de cada local
        plt.text(s_sarsa[1], s_sarsa[0], 'S', ha='center', va='center', color='w')  # Printa a recompensa de cada local
    mundo[s_q] = -1
    mundo[s_sarsa] = -2
    plt.imshow(mundo, cmap='inferno')
    plt.title(f"Posição do agente ao longo do episódio {k}")
    plt.text((m - 2) / 2, n - 1, 'Cliff', ha='center', va='center', color='w', fontsize=16)  # eescreve cliff

def faz_frames(dic_caminhos,dic_caminhos_sarsa,k): #faz os frames para comparação q-learing e sarsa pq podem tem tamanhos diferentes de caminhos
    frame = []
    if len(dic_caminhos[f'caminho_k={k}']) > len(dic_caminhos_sarsa[f'caminho_k={k}']):
        for i in range(len(dic_caminhos_sarsa[f'caminho_k={k}'])):  # o menor
            frame.append((dic_caminhos[f'caminho_k={k}'][i], dic_caminhos_sarsa[f'caminho_k={k}'][i]))
        for i in range(len(dic_caminhos_sarsa[f'caminho_k={k}']), len(dic_caminhos[f'caminho_k={k}'])):
            frame.append(((dic_caminhos)[f'caminho_k={k}'][i],
                          dic_caminhos_sarsa[f'caminho_k={k}'][len(dic_caminhos_sarsa[f'caminho_k={k}']) - 1]))

    elif len(dic_caminhos[f'caminho_k={k}']) < len(dic_caminhos_sarsa[f'caminho_k={k}']):
        for i in range(len(dic_caminhos[f'caminho_k={k}'])):  # o menor
            frame.append((dic_caminhos[f'caminho_k={k}'][i], dic_caminhos_sarsa[f'caminho_k={k}'][i]))
        for i in range(len(dic_caminhos[f'caminho_k={k}']), len(dic_caminhos_sarsa[f'caminho_k={k}'])):
            frame.append(((dic_caminhos)[f'caminho_k={k}'][len(dic_caminhos[f'caminho_k={k}']) - 1],
                          dic_caminhos_sarsa[f'caminho_k={k}'][i]))

    else:
        for i in range(len(dic_caminhos[f'caminho_k={k}'])):
            frame.append((dic_caminhos[f'caminho_k={k}'][i], dic_caminhos_sarsa[f'caminho_k={k}'][i]))

    return frame

#--- inicialização e q-learnig
states = init_states(n,m)
qsa_q = init_qsa(states)
final_q,histo_q,dic_caminhos = q_learning(qsa_q,episodes)

#--- inicialização e q-learnig
qsa_sarsa = init_qsa(states)
final_sarsa,histo_sarsa,dic_caminhos_sarsa = sarsa(qsa_sarsa,episodes)


#-- plota o heatmap
plot_heatmap(histo_q)


#--------- animação do episódio k
fig, ax = plt.subplots()
k = 50
animation(dic_caminhos[f'caminho_k={k}'],k,fig)

#--------- animação de comparação sarsa e q-learning ao longo dos ep
'''for k in [1 ,50, 500 ]:
    fig, ax = plt.subplots()
    frame = faz_frames(dic_caminhos,dic_caminhos_sarsa,k)
    animation_comparacao(frame, k, fig)'''

