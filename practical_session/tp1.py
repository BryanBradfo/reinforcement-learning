import matplotlib.pyplot as plt 
import numpy as np

def print_full(matrix):
    print("[", end="")
    for i in range (len(matrix)):
        if i == len(matrix) - 1:
            print(matrix[i], end="]")
        else:
            print(matrix[i], end=",\n")
    print()


sieges_nb = 20
temps_nb = 50

act_list = np.zeros((temps_nb + 1, sieges_nb + 1))
val_list = np.zeros((temps_nb + 1, sieges_nb + 1))

def Value(temps, sieges):
    rs_a1 = 0.5
    rs_a2 = 0.8

    for s in range(sieges):
        for t in range(temps):
            res1 = rs_a1 + 0.9 * val_list[t - 1][s] + 0.1 * val_list[t - 1][s - 1]
            res2 = rs_a2 + 0.2 * val_list[t - 1][s] + 0.8 * val_list[t - 1][s - 1]

            if res1 > res2:
                act_list[t][s] = 1
                val_list[t][s] = res1
            else:
                act_list[t][s] = 2
                val_list[t][s] = res2

Value(temps_nb, sieges_nb)
print("Action à prendre pour chaque état")
# make act_list a list of int
act_list = act_list.astype(int)
print(act_list)
# print(len(act_list))
# print_full(act_list)

# Enregistrer la matrice dans un fichier texte
# np.savetxt("action_matrix.txt", act_list)

print("Valeur de la fonction de valeur pour chaque état")
# keep only the two first decimals of the float
val_list = np.around(val_list, decimals=2)
print_full(val_list)
# np.savetxt("value_function_matrix.txt", val_list)


# Heatmap de l'action à prendre pour chaque état
plt.imshow(act_list.T, cmap="Blues")
plt.xlabel("Temps avant le décollage (t)")
plt.ylabel("Nombre de tickets d'avion (S)")
plt.title("Action à prendre pour chaque état")
plt.colorbar()
plt.show()

