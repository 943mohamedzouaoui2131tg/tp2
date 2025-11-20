import sys
import matplotlib
matplotlib.use('Agg')

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
    # Cercle supérieur
    [0, 2],      # X1
    [0.5, 2.5],  # X2
    [1, 2],      # X3
    [0.5, 1.5],  # X4
    # Zone de jonction (va provoquer la fusion)
    [0.5, 1],    # X5 - Point clé
    # Cercle inférieur
    [0, 0],      # X6
    [0.5, -0.5], # X7
    [1, 0],      # X8
    [0.5, 0.5]   # X9
])

print("=" * 80)
print("MEAN-SHIFT AVEC OPTIMISATION DE FUSION DES CLUSTERS")
print("=" * 80)

print("\n📊 Données initiales :")
print("-" * 80)
for i, p in enumerate(points, 1):
    print(f"X{i} = ({p[0]}, {p[1]})")

# ÉTAPE 1 : CALCUL DU BANDWIDTH
print("\n" + "=" * 80)
print("ÉTAPE 1 : CALCUL DU BANDWIDTH (Règle de Scott)")
print("=" * 80)

n = len(points)
d = points.shape[1]

sigma_x = np.std(points[:, 0], ddof=0)
sigma_y = np.std(points[:, 1], ddof=0)
sigma = (sigma_x + sigma_y) / 2

h = n ** (-1 / (d + 4)) * sigma

mean_x = np.mean(points[:, 0])
mean_y = np.mean(points[:, 1])

print(f"\nn = {n}, d = {d}")
print(f"σ_x = {sigma_x:.4f}, σ_y = {sigma_y:.4f}")
print(f"σ = {sigma:.4f}")
print(f"✅ BANDWIDTH h = {h:.4f}")

# ÉTAPE 2 : MEAN-SHIFT ITÉRATIF
print("\n" + "=" * 80)
print("ÉTAPE 2 : MEAN-SHIFT ITÉRATIF")
print("=" * 80)

def gaussian_kernel(distance, bandwidth):
    return np.exp(-0.5 * (distance / bandwidth) ** 2)

def mean_shift_step(point, all_points, bandwidth):
    distances = np.sqrt(np.sum((all_points - point) ** 2, axis=1))
    weights = gaussian_kernel(distances, bandwidth)
    in_window = distances <= bandwidth
    
    numerator = np.sum(weights[:, np.newaxis] * all_points, axis=0)
    denominator = np.sum(weights)
    new_point = numerator / denominator
    
    return new_point, weights, distances, in_window

trajectories = {i: [points[i].copy()] for i in range(n)}
max_iterations = 20
convergence_threshold = 0.01

for point_idx in range(n):
    print(f"\n🔹 Point X{point_idx + 1}")
    current_point = points[point_idx].copy()
    
    for iteration in range(max_iterations):
        new_point, weights, distances, in_window = mean_shift_step(current_point, points, h)
        shift_distance = np.linalg.norm(new_point - current_point)
        
        trajectories[point_idx].append(new_point.copy())
        
        if shift_distance < convergence_threshold:
            print(f"   Convergé en {iteration + 1} itérations → ({new_point[0]:.4f}, {new_point[1]:.4f})")
            break
        
        current_point = new_point.copy()

# ÉTAPE 3 : IDENTIFICATION INITIALE DES CLUSTERS
print("\n" + "=" * 80)
print("ÉTAPE 3 : IDENTIFICATION INITIALE DES CLUSTERS")
print("=" * 80)

final_points = np.array([trajectories[i][-1] for i in range(n)])
cluster_threshold = h * 0.5

clusters = []
assigned = [False] * n

for i in range(n):
    if assigned[i]:
        continue
    
    cluster = [i]
    assigned[i] = True
    
    for j in range(i + 1, n):
        if not assigned[j]:
            distance = np.linalg.norm(final_points[i] - final_points[j])
            if distance < cluster_threshold:
                cluster.append(j)
                assigned[j] = True
    
    clusters.append(cluster)

print(f"\n✅ Clusters initiaux détectés : {len(clusters)}")
for cluster_id, cluster_members in enumerate(clusters, 1):
    members_str = ', '.join([f"X{i+1}" for i in cluster_members])
    center = np.mean([final_points[i] for i in cluster_members], axis=0)
    print(f"Cluster {cluster_id} : {members_str}")
    print(f"  Centre : ({center[0]:.4f}, {center[1]:.4f})")

# ============================================================
# ÉTAPE 4 : OPTIMISATION - FUSION DES CLUSTERS QUI SE CHEVAUCHENT
# ============================================================
print("\n" + "=" * 80)
print("ÉTAPE 4 : OPTIMISATION - FUSION DES CLUSTERS")
print("=" * 80)

def calculate_cluster_center(cluster_members, final_points):
    """Calcule le centre d'un cluster"""
    return np.mean([final_points[i] for i in cluster_members], axis=0)

def calculate_cluster_density(cluster_members, all_points, bandwidth):
    """
    Calcule la densité d'un cluster basée sur le noyau gaussien.
    Densité = somme des poids de tous les points dans le cluster
    """
    center = calculate_cluster_center(cluster_members, all_points)
    distances = np.sqrt(np.sum((all_points[cluster_members] - center) ** 2, axis=1))
    weights = gaussian_kernel(distances, bandwidth)
    density = np.sum(weights)
    return density

def circles_intersect(center1, center2, radius):
    """Vérifie si deux cercles se chevauchent"""
    distance = np.linalg.norm(center1 - center2)
    return distance < 2 * radius

def merge_clusters_weighted(cluster1, cluster2, all_points, bandwidth):
    """
    Fusionne deux clusters en utilisant une moyenne pondérée par la densité.
    Justification : Les points du cluster le plus dense ont plus d'influence
    sur le centre final, ce qui préserve mieux la structure des données.
    """
    density1 = calculate_cluster_density(cluster1, all_points, bandwidth)
    density2 = calculate_cluster_density(cluster2, all_points, bandwidth)
    
    merged_cluster = cluster1 + cluster2
    
    return merged_cluster, density1, density2

print(f"\nVérification des intersections de fenêtres (rayon h = {h:.4f}):")
print("-" * 80)

# Créer une copie des clusters pour la fusion
optimized_clusters = [cluster.copy() for cluster in clusters]
merge_occurred = True
iteration = 0

while merge_occurred and iteration < 10:  # Maximum 10 itérations pour éviter boucle infinie
    merge_occurred = False
    iteration += 1
    
    print(f"\n🔄 Itération de fusion {iteration}")
    
    i = 0
    while i < len(optimized_clusters):
        j = i + 1
        while j < len(optimized_clusters):
            cluster1 = optimized_clusters[i]
            cluster2 = optimized_clusters[j]
            
            center1 = calculate_cluster_center(cluster1, final_points)
            center2 = calculate_cluster_center(cluster2, final_points)
            
            # Vérifier si les fenêtres se chevauchent
            if circles_intersect(center1, center2, h):
                print(f"\n⚠️  Intersection détectée entre Cluster {i+1} et Cluster {j+1}")
                print(f"   Distance entre centres : {np.linalg.norm(center1 - center2):.4f} < 2h = {2*h:.4f}")
                
                # Calculer les densités
                density1 = calculate_cluster_density(cluster1, final_points, h)
                density2 = calculate_cluster_density(cluster2, final_points, h)
                
                print(f"   Densité Cluster {i+1} : {density1:.4f} ({len(cluster1)} points)")
                print(f"   Densité Cluster {j+1} : {density2:.4f} ({len(cluster2)} points)")
                
                # Fusionner en utilisant la moyenne pondérée
                merged, d1, d2 = merge_clusters_weighted(cluster1, cluster2, final_points, h)
                
                # Calcul du poids relatif pour le centre fusionné
                total_density = d1 + d2
                weight1 = d1 / total_density
                weight2 = d2 / total_density
                
                new_center = weight1 * center1 + weight2 * center2
                
                print(f"   ✅ FUSION effectuée (moyenne pondérée par densité)")
                print(f"   Poids Cluster {i+1} : {weight1:.4f}")
                print(f"   Poids Cluster {j+1} : {weight2:.4f}")
                print(f"   Nouveau centre : ({new_center[0]:.4f}, {new_center[1]:.4f})")
                print(f"   Nouveau cluster : {len(merged)} points")
                
                # Remplacer cluster1 par le cluster fusionné et supprimer cluster2
                optimized_clusters[i] = merged
                del optimized_clusters[j]
                merge_occurred = True
                break
            
            j += 1
        
        if merge_occurred:
            break
        i += 1
    
    if not merge_occurred:
        print(f"   ✅ Aucune fusion nécessaire")

print("\n" + "=" * 80)
print("RÉSULTAT FINAL APRÈS OPTIMISATION")
print("=" * 80)

print(f"\n✅ Nombre de clusters finaux : {len(optimized_clusters)}")
for cluster_id, cluster_members in enumerate(optimized_clusters, 1):
    members_str = ', '.join([f"X{i+1}" for i in cluster_members])
    center = calculate_cluster_center(cluster_members, final_points)
    density = calculate_cluster_density(cluster_members, final_points, h)
    print(f"\nCluster {cluster_id} : {members_str}")
    print(f"  Centre : ({center[0]:.4f}, {center[1]:.4f})")
    print(f"  Densité : {density:.4f}")
    print(f"  Nombre de points : {len(cluster_members)}")

# ============================================================
# VISUALISATION COMPARATIVE
# ============================================================
print("\n" + "=" * 80)
print("VISUALISATION")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 16))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']

# Graphique 1 : Points initiaux avec rayon h
ax = axes[0, 0]
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
ax.set_xlim(-4, 7)
ax.set_ylim(-1, 4)

# Graphique 2 : Trajectoires
ax = axes[0, 1]
for i in range(n):
    traj = np.array(trajectories[i])
    ax.plot(traj[:, 0], traj[:, 1], 'o-', color=colors[i % len(colors)], linewidth=2,
            markersize=6, label=f'X{i+1}', alpha=0.7)
    ax.scatter(traj[0, 0], traj[0, 1], s=200, c=colors[i % len(colors)], marker='o',
               edgecolors='black', linewidths=2, zorder=3)
    ax.scatter(traj[-1, 0], traj[-1, 1], s=300, c=colors[i % len(colors)], marker='*',
               edgecolors='black', linewidths=2, zorder=4)

ax.set_title('Trajectoires de convergence', fontsize=14, fontweight='bold')
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.set_xlim(-4, 7)
ax.set_ylim(-1, 4)

# Graphique 3 : Clusters AVANT optimisation
ax = axes[1, 0]
for cluster_id, cluster_members in enumerate(clusters):
    cluster_points = points[cluster_members]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               s=200, c=colors[cluster_id % len(colors)], marker='o',
               edgecolors='black', linewidths=2, label=f'Cluster {cluster_id+1}')
    
    for member_idx in cluster_members:
        p = points[member_idx]
        ax.annotate(f'X{member_idx+1}', (p[0], p[1]), fontsize=12, 
                    fontweight='bold', ha='center', va='center', color='white')
    
    center = calculate_cluster_center(cluster_members, final_points)
    ax.scatter(center[0], center[1], s=400, c=colors[cluster_id % len(colors)],
               marker='X', edgecolors='white', linewidths=3, zorder=5)
    
    # Dessiner la fenêtre autour du centre
    circle = Circle((center[0], center[1]), h, fill=False, 
                    edgecolor=colors[cluster_id % len(colors)], 
                    linestyle='--', linewidth=2, alpha=0.7)
    ax.add_patch(circle)

ax.set_title(f'AVANT optimisation\n({len(clusters)} clusters)', fontsize=14, fontweight='bold')
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.set_xlim(-4, 7)
ax.set_ylim(-1, 4)

# Graphique 4 : Clusters APRÈS optimisation
ax = axes[1, 1]
for cluster_id, cluster_members in enumerate(optimized_clusters):
    cluster_points = points[cluster_members]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               s=200, c=colors[cluster_id % len(colors)], marker='o',
               edgecolors='black', linewidths=2, label=f'Cluster {cluster_id+1}')
    
    for member_idx in cluster_members:
        p = points[member_idx]
        ax.annotate(f'X{member_idx+1}', (p[0], p[1]), fontsize=12, 
                    fontweight='bold', ha='center', va='center', color='white')
    
    center = calculate_cluster_center(cluster_members, final_points)
    ax.scatter(center[0], center[1], s=400, c=colors[cluster_id % len(colors)],
               marker='X', edgecolors='white', linewidths=3, zorder=5)
    
    # Dessiner la fenêtre autour du centre
    circle = Circle((center[0], center[1]), h, fill=False, 
                    edgecolor=colors[cluster_id % len(colors)], 
                    linestyle='--', linewidth=2, alpha=0.7)
    ax.add_patch(circle)

ax.set_title(f'APRÈS optimisation\n({len(optimized_clusters)} clusters)', 
             fontsize=14, fontweight='bold', color='green')
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.set_xlim(-4, 7)
ax.set_ylim(-1, 4)

plt.tight_layout()
plt.savefig('meanshift_optimized_result.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphique sauvegardé : meanshift_optimized_result.png")

print("\n" + "=" * 80)
print("FIN DE L'ANALYSE OPTIMISÉE")
print("=" * 80)