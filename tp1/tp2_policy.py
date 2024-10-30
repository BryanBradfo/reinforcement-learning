import numpy as np

# Définir la matrice de récompense R
R = np.array([[1, 10], [0, -15]])  # R: [0->0, 0->1, 1->0, 1->1]

# Définir les valeurs de gamma
gammas = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

# Définir une politique arbitraire (par exemple, politique qui choisit l'action 0 partout)
policy = np.array([0, 0])

# Définition de eps
eps = 1e-2

max_iter = 1e3

for gamma in gammas:
    V = np.array([[0, 0], [0, 0]])
    
    count = 0
    while True:
        count += 1
        
        # Calculer la valeur V en utilisant l'équation de Bellman V = R + gamma * P * V
        V_new = np.abs(R + gamma * np.dot(np.array([[1, 1], [1, 1]]), V))
        
        # print(np.abs(V_new - V_old))

        # Vérifier la convergence de la valeur
        if np.max(np.abs(V_new - V)) < eps or count > max_iter:
            break

        V = V_new.copy()

        # print(gamma * np.dot(np.array([1, 1]), V))

        # Mettre à jour la politique en choisissant l'action qui maximise la valeur de Bellman
        policy[0] = np.argmax(R[0] + gamma * np.dot(np.array([1, 1]), V))
        policy[1] = np.argmax(R[1] + gamma * np.dot(np.array([1, 1]), V))
    
    print(f"Gamma = {gamma}:")
    print("Politique optimale :", policy)
    print("Nombre d'itérations :", count)
    print()
