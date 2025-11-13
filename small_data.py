import sys
import matplotlib
# Forcer matplotlib en mode non interactif (évite que plt.show() bloque)
matplotlib.use('Agg')

# Forcer stdout en UTF-8 pour éviter UnicodeEncodeError lors de redirection
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Données
points = np.array([
    [0, 3],  # X1
    [0, 0],  # X2
    [2, 0],  # X3
    [6, 0],  # X4
    [5, 2]   # X5
])

print("=" * 80)
print("MEAN-SHIFT MANUEL SUR 5 POINTS")
print("=" * 80)

print("\n📍 Données initiales :")
print("-" * 80)
for i, p in enumerate(points, 1):
    print(f"X{i} = ({p[0]}, {p[1]})")

# ============================================================
# ÉTAPE 1 : CALCUL DU BANDWIDTH AVEC LA RÈGLE DE SCOTT
# ============================================================
print("\n" + "=" * 80)
print("ÉTAPE 1 : CALCUL DU BANDWIDTH (Règle de Scott)")
print("=" * 80)

n = len(points)
d = points.shape[1]

# Calculer l'écart-type pour chaque dimension
sigma_x = np.std(points[:, 0], ddof=0)  # ddof=0 pour population
sigma_y = np.std(points[:, 1], ddof=0)
sigma = (sigma_x + sigma_y) / 2

# Formule de Scott
h = n ** (-1 / (d + 4)) * sigma

print(f"\nn (nombre de points)     = {n}")
print(f"d (dimensions)           = {d}")
print(f"\nCalcul des écarts-types :")
print(f"  σ_x = sqrt(Σ(xi - μx)² / n)")

# Détail du calcul de σ_x
mean_x = np.mean(points[:, 0])
mean_y = np.mean(points[:, 1])
print(f"\n  Moyenne X (μx) = ({' + '.join(map(str, points[:, 0]))}) / {n} = {mean_x:.2f}")
print(f"  Moyenne Y (μy) = ({' + '.join(map(str, points[:, 1]))}) / {n} = {mean_y:.2f}")

print(f"\n  Pour X :")
for i, p in enumerate(points, 1):
    print(f"    X{i}: ({p[0]} - {mean_x:.2f})² = {(p[0] - mean_x)**2:.2f}")
sum_x = np.sum((points[:, 0] - mean_x)**2)
print(f"  Somme = {sum_x:.2f}")
print(f"  σ_x = sqrt({sum_x:.2f} / {n}) = {sigma_x:.4f}")

print(f"\n  Pour Y :")
for i, p in enumerate(points, 1):
    print(f"    X{i}: ({p[1]} - {mean_y:.2f})² = {(p[1] - mean_y)**2:.2f}")
sum_y = np.sum((points[:, 1] - mean_y)**2)
print(f"  Somme = {sum_y:.2f}")
print(f"  σ_y = sqrt({sum_y:.2f} / {n}) = {sigma_y:.4f}")

print(f"\nσ (écart-type moyen)     = ({sigma_x:.4f} + {sigma_y:.4f}) / 2 = {sigma:.4f}")

print(f"\nFormule de Scott : h = n^(-1/(d+4)) × σ")
print(f"                 : h = {n}^(-1/({d}+4)) × {sigma:.4f}")
print(f"                 : h = {n}^(-1/{d+4}) × {sigma:.4f}")
print(f"                 : h = {n**(-1/(d+4)):.4f} × {sigma:.4f}")
print(f"                 : h = {h:.4f}")

print(f"\n✅ BANDWIDTH h = {h:.4f}")

# ============================================================
# ÉTAPE 2 : MEAN-SHIFT ITÉRATIF
# ============================================================
print("\n" + "=" * 80)
print("ÉTAPE 2 : MEAN-SHIFT ITÉRATIF")
print("=" * 80)

def gaussian_kernel(distance, bandwidth):
    """Noyau Gaussien"""
    return np.exp(-0.5 * (distance / bandwidth) ** 2)

def mean_shift_step(point, all_points, bandwidth):
    """Une itération de mean-shift pour un point"""
    # Calculer les distances
    distances = np.sqrt(np.sum((all_points - point) ** 2, axis=1))
    
    # Calculer les poids avec le noyau Gaussien
    weights = gaussian_kernel(distances, bandwidth)
    
    # Points dans le rayon h (pour affichage)
    in_window = distances <= bandwidth
    
    # Nouveau centre = moyenne pondérée
    numerator = np.sum(weights[:, np.newaxis] * all_points, axis=0)
    denominator = np.sum(weights)
    new_point = numerator / denominator
    
    return new_point, weights, distances, in_window

# Appliquer mean-shift sur chaque point
print(f"\nNoyau utilisé : Gaussien K(d) = exp(-0.5 × (d/h)²)")
print(f"Rayon de convergence : h = {h:.4f}\n")

trajectories = {i: [points[i].copy()] for i in range(n)}
max_iterations = 20
convergence_threshold = 0.01

for point_idx in range(n):
    print("=" * 80)
    print(f"POINT X{point_idx + 1} = ({points[point_idx][0]}, {points[point_idx][1]})")
    print("=" * 80)
    
    current_point = points[point_idx].copy()
    
    for iteration in range(max_iterations):
        print(f"\n--- Itération {iteration + 1} ---")
        print(f"Position actuelle : ({current_point[0]:.4f}, {current_point[1]:.4f})")
        
        # Calculer le prochain point
        new_point, weights, distances, in_window = mean_shift_step(current_point, points, h)
        
        # Afficher les détails
        print(f"\nDistances et poids :")
        print(f"{'Point':<8} {'Distance':<12} {'Poids':<12} {'Dans fenêtre'}")
        print("-" * 50)
        for i in range(n):
            print(f"X{i+1:<7} {distances[i]:<12.4f} {weights[i]:<12.4f} {'✓' if in_window[i] else '✗'}")
        
        # Calcul du nouveau centre
        print(f"\nCalcul du nouveau centre :")
        print(f"  Numérateur   = Σ(poids × point)")
        for i in range(n):
            print(f"    + {weights[i]:.4f} × ({points[i][0]}, {points[i][1]}) = ({weights[i]*points[i][0]:.4f}, {weights[i]*points[i][1]:.4f})")
        numerator = np.sum(weights[:, np.newaxis] * points, axis=0)
        denominator = np.sum(weights)
        print(f"  Somme        = ({numerator[0]:.4f}, {numerator[1]:.4f})")
        print(f"  Dénominateur = Σ(poids) = {denominator:.4f}")
        print(f"  Nouveau centre = ({new_point[0]:.4f}, {new_point[1]:.4f})")
        
        # Vérifier la convergence
        shift_distance = np.linalg.norm(new_point - current_point)
        print(f"\nDéplacement : {shift_distance:.4f}")
        
        trajectories[point_idx].append(new_point.copy())
        
        if shift_distance < convergence_threshold:
            print(f"✅ CONVERGENCE atteinte (déplacement < {convergence_threshold})")
            print(f"   Centre final : ({new_point[0]:.4f}, {new_point[1]:.4f})")
            break
        
        current_point = new_point.copy()
    else:
        print(f"⚠️  Max itérations atteint")
        print(f"   Position finale : ({current_point[0]:.4f}, {current_point[1]:.4f})")

# ============================================================
# ÉTAPE 3 : IDENTIFICATION DES CLUSTERS
# ============================================================
print("\n" + "=" * 80)
print("ÉTAPE 3 : IDENTIFICATION DES CLUSTERS")
print("=" * 80)

# Points finaux
final_points = np.array([trajectories[i][-1] for i in range(n)])
print(f"\nPositions finales après convergence :")
for i in range(n):
    fp = final_points[i]
    print(f"X{i+1} → ({fp[0]:.4f}, {fp[1]:.4f})")

# Regrouper les points similaires
cluster_threshold = h * 0.5  # Points à moins de h/2 sont dans le même cluster
clusters = []
assigned = [False] * n

for i in range(n):
    if assigned[i]:
        continue
    
    # Nouveau cluster
    cluster = [i]
    assigned[i] = True
    
    for j in range(i + 1, n):
        if not assigned[j]:
            distance = np.linalg.norm(final_points[i] - final_points[j])
            if distance < cluster_threshold:
                cluster.append(j)
                assigned[j] = True
    
    clusters.append(cluster)

print(f"\n✅ Nombre de clusters détectés : {len(clusters)}")
print(f"   (Seuil de regroupement : {cluster_threshold:.4f})\n")

for cluster_id, cluster_members in enumerate(clusters, 1):
    members_str = ', '.join([f"X{i+1}" for i in cluster_members])
    center = np.mean([final_points[i] for i in cluster_members], axis=0)
    print(f"Cluster {cluster_id} : {members_str}")
    print(f"  Centre : ({center[0]:.4f}, {center[1]:.4f})")

# ============================================================
# VISUALISATION
# ============================================================
print("\n" + "=" * 80)
print("VISUALISATION")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Graphique 1 : Points initiaux avec rayon h
ax = axes[0]
ax.scatter(points[:, 0], points[:, 1], s=200, c='blue', marker='o', 
           edgecolors='black', linewidths=2, zorder=3)
for i, p in enumerate(points):
    ax.annotate(f'X{i+1}', (p[0], p[1]), fontsize=12, fontweight='bold',
                ha='center', va='center', color='white')
    circle = Circle((p[0], p[1]), h, fill=False, edgecolor='gray', 
                    linestyle='--', linewidth=1.5, alpha=0.5)
    ax.add_patch(circle)

ax.set_title(f'Points initiaux\n(Bandwidth h = {h:.4f})', fontsize=14, fontweight='bold')
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 4)

# Graphique 2 : Trajectoires
ax = axes[1]
for i in range(n):
    traj = np.array(trajectories[i])
    ax.plot(traj[:, 0], traj[:, 1], 'o-', color=colors[i], linewidth=2,
            markersize=6, label=f'X{i+1}', alpha=0.7)
    # Point de départ
    ax.scatter(traj[0, 0], traj[0, 1], s=200, c=colors[i], marker='o',
               edgecolors='black', linewidths=2, zorder=3)
    # Point d'arrivée
    ax.scatter(traj[-1, 0], traj[-1, 1], s=300, c=colors[i], marker='*',
               edgecolors='black', linewidths=2, zorder=4)

ax.set_title('Trajectoires Mean-Shift', fontsize=14, fontweight='bold')
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 4)

# Graphique 3 : Clusters finaux
ax = axes[2]
cluster_colors = ['red', 'blue', 'green', 'orange']
for cluster_id, cluster_members in enumerate(clusters):
    cluster_points = points[cluster_members]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               s=200, c=cluster_colors[cluster_id], marker='o',
               edgecolors='black', linewidths=2, label=f'Cluster {cluster_id+1}')
    
    # Ajouter les labels
    for member_idx in cluster_members:
        p = points[member_idx]
        ax.annotate(f'X{member_idx+1}', (p[0], p[1]), fontsize=12, 
                    fontweight='bold', ha='center', va='center', color='white')
    
    # Centre du cluster
    center = np.mean([final_points[i] for i in cluster_members], axis=0)
    ax.scatter(center[0], center[1], s=400, c=cluster_colors[cluster_id],
               marker='X', edgecolors='white', linewidths=3, zorder=5)

ax.set_title(f'Résultat final\n({len(clusters)} clusters)', fontsize=14, fontweight='bold')
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 4)

plt.tight_layout()
plt.savefig('meanshift_manual_result.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphique sauvegardé : meanshift_manual_result.png")
plt.show()

print("\n" + "=" * 80)
print("FIN DE L'ANALYSE")
print("=" * 80)