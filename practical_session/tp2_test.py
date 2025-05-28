import numpy as np

# Définir la matrice de récompense R
R = np.array([[1, 10], [0, -15]])  # R: [0->0, 0->1, 1->0, 1->1]

# Définir les valeurs de gamma
gammas = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

# Définir une politique arbitraire (par exemple, politique qui choisit l'action 0 partout)
policy = np.array([0, 0])

# Définition de eps
eps = 1e-2

for gamma in gammas:
    V = np.array([[0, 0], [0, 0]])
    
    # Calculer la matrice de transition P en incluant les transitions que vous avez décrites
    P = np.array([[1, 1], [1, 1]])

    count = 0
    count2 = 0
    while True:
        count += 1
        
        # Calculer la matrice de transition pondérée par gamma
        P_gamma = gamma * P
        
        # Calculer la matrice inverse (I - gamma*P)^(-1)
        I = np.identity(len(R))
        inv_matrix = np.linalg.inv(I - P_gamma)
        
        # Calculer la valeur V en utilisant l'équation de Bellman V = (I - gamma*P)^(-1) * R
        V_new = np.dot(inv_matrix, R)

        print(np.max(np.abs(V_new - V)))

        # Vérifier la convergence de la valeur
        if np.max(np.abs(V_new - V)) < eps:
            break

        else:
            count2 += 1

        
        V = V_new.copy()

        # Mettre à jour la politique en choisissant l'action qui maximise la valeur de Bellman
        policy[0] = np.argmax(R[0] + gamma * np.dot(P[0], V))
        policy[1] = np.argmax(R[1] + gamma * np.dot(P[1], V))

        # if policy[0] == 0:
        #     P[0][0] = 1
        #     P[0][1] = 0
        # else:
        #     P[0][0] = 0
        #     P[0][1] = 1
        
        # if policy[1] == 0:
        #     P[1][0] = 1
        #     P[1][1] = 0
        # else:
        #     P[1][0] = 0
        #     P[1][1] = 1

        if count2 > 10:
            P = np.array([[0, 1], [1, 0]])


    
    print(f"Gamma = {gamma}:")
    print("Politique optimale :", policy)
    print("Nombre d'itérations :", count)
    print()
