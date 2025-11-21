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
# Données
points = np.array([
    [-3.50855218,  8.35083161],
       [ 9.25992542, -2.19438068],
       [ 1.97823271,  4.15085122],
       [-1.01491687,  7.0781822 ],
       [-0.87088204,  3.61786079],
       [ 0.4046787 ,  3.99816714],
       [10.56525084, -0.86951871],
       [ 2.1631428 ,  1.23935462],
       [-2.06872548,  2.58020122],
       [-1.34869521,  7.25730854],
       [ 2.39187323, -0.03143862],
       [ 9.96559706, -2.28108287],
       [ 2.02201716,  5.3258102 ],
       [ 2.24128444,  5.91529313],
       [ 9.1626643 , -2.25130938],
       [ 9.11119974, -2.66679171],
       [ 1.37208315,  2.99798019],
       [ 5.062052  ,  0.21722842],
       [ 0.41426969,  8.16703934],
       [ 2.54536972,  0.72060972],
       [ 2.27913241,  5.55274228],
       [ 0.08500006,  3.09677208],
       [ 0.2151235 ,  4.63265445],
       [ 1.99726216,  2.35400497],
       [ 1.71104085,  3.67294706],
       [ 1.46488484,  0.7704743 ],
       [ 7.70664609, -2.73744753],
       [ 2.57723499,  1.68153933],
       [-2.20605771,  7.21438393],
       [-2.73735579,  2.49856036],
       [-1.2889681 ,  2.83445443],
       [-1.28410151,  7.5916059 ],
       [ 1.05811263,  2.54974167],
       [ 7.12082963,  1.04806065],
       [-1.84725101,  8.63723163],
       [-1.85991951,  7.14441598],
       [-1.3619021 ,  8.75154225],
       [10.06532633, -2.04219939],
       [ 6.53284862,  0.58239714],
       [ 2.37522922, -0.03682701],
       [ 0.63356967,  5.34286564],
       [-0.75325194,  3.18751774],
       [-1.98833627,  3.2715248 ],
       [ 1.27936863,  1.8471275 ],
       [10.43310728, -2.91720143],
       [-2.33945726,  2.62379775],
       [ 1.1270823 ,  3.96227353],
       [ 1.51479835,  1.47263196],
       [ 8.76439936, -3.27968206],
       [-2.32968333,  2.56934002],
       [ 8.86554777, -1.80386507],
       [ 5.27473074,  0.14079952],
       [ 1.62315214,  4.40721109],
       [ 0.4370963 ,  3.99545734],
       [ 1.10797539,  4.62522547],
       [ 2.86043169,  0.76590508],
       [-1.36381302,  8.80181767],
       [ 5.00906188,  0.84549302],
       [-1.58778745,  8.15000752],
       [ 6.12256927,  2.29578542],
       [-0.61314569,  6.8248568 ],
       [ 1.94476066,  3.25418538],
       [ 1.40083886,  1.35602547],
       [-1.78956504,  1.49327903],
       [-0.85852525,  2.52142603],
       [ 9.02519345, -2.978438  ],
       [-0.99545728,  1.55613325],
       [ 5.39460958, -0.25410699],
       [ 3.68976824,  2.15610123],
       [ 4.93806657,  1.27510906],
       [ 6.79380572,  0.9994992 ],
       [-0.66191033,  8.94202931],
       [ 2.06419254,  2.41565358],
       [ 0.2360925 ,  3.81176511],
       [ 8.12629121,  0.51506243],
       [ 9.75545195, -2.52044346],
       [-1.82779884,  1.74917366],
       [ 3.88917373,  0.79166015],
       [-0.01410729,  5.06948984],
       [ 1.63193994,  2.53776591],
       [-2.23200775,  7.5700116 ],
       [-2.89409361,  7.99592182],
       [ 6.49019582,  1.4529214 ],
       [ 6.22816154,  0.81512979],
       [ 7.31254254,  1.42313363],
       [ 8.97283822, -3.70517268],
       [-0.08860598,  4.96465413],
       [ 1.70369139,  0.26232707],
       [ 4.08094058,  1.70047122],
       [ 1.27918152,  4.90437452],
       [-1.34221533,  8.70260682],
       [-1.08399887,  2.42846249],
       [-0.59099074,  6.82916147],
       [-1.84070858,  2.71440996],
       [ 3.64269264,  1.66780162],
       [-0.34002279,  7.45008208],
       [ 9.82007251, -3.66619952],
       [-1.19377126,  4.85936313],
       [-1.51250902,  7.91825016],
       [ 6.56770737, -1.67937783],
       [10.6741699 , -2.19157603],
       [ 1.20508448, -0.41539177],
       [ 2.86227521,  0.97208222],
       [-2.46562985,  2.96222258],
       [-2.53743438,  2.48708836],
       [-2.03462267,  2.71564879],
       [-1.08211885,  2.77206788],
       [-0.82857526,  7.73677167],
       [ 3.65594561, -0.24793154],
       [10.74606822, -1.74934368],
       [ 9.58845646, -2.21041709],
       [ 0.54208571,  3.30024997],
       [ 7.4928426 , -2.226006  ],
       [-1.62304001,  7.59715784],
       [-0.39554309,  4.12295809],
       [ 0.64704228,  4.04682999],
       [ 5.76145961,  0.74304739],
       [-1.40624382,  2.64645311],
       [-1.55824529,  7.96865829],
       [-0.8029484 ,  7.91061879],
       [ 9.01050793, -2.52961629],
       [-0.40939852,  4.69715224],
       [ 1.97158288,  0.33370712],
       [ 9.76499848, -2.03396334],
       [-2.03377095,  2.87320006],
       [-1.16518955,  8.33097064],
       [ 1.03280972,  4.56088844],
       [ 1.685387  ,  2.46953783],
       [10.00249055, -1.36123304],
       [-1.39387346,  3.11523614],
       [-0.98279222,  1.54346962],
       [ 1.32322569,  2.52121887],
       [10.07484438, -4.65989522],
       [ 8.78933996, -2.10175092],
       [-2.13672114,  8.86493351],
       [ 0.95231518,  4.66786942],
       [ 1.42222592,  0.1951909 ],
       [ 2.78357354,  0.34429191],
       [-1.92096059,  3.32714151],
       [ 0.22165219,  2.62011033],
       [ 0.71007342,  6.93444657],
       [ 5.60018081, -0.02537038],
       [ 8.71880565, -2.77418054],
       [-0.65854859,  7.69982259],
       [ 2.83799752,  1.16858216],
       [ 1.31826047,  3.72169875],
       [ 0.78769857,  1.27114661],
       [ 5.01807204, -0.09625136],
       [ 1.66806489,  0.91252095],
       [ 8.62373287, -3.55329238],
       [ 3.01290806, -0.02027812],
       [ 1.15330381,  5.13393051],
       [-1.30063822,  7.74425053],
       [ 0.71145043,  4.35152787],
       [ 7.82986526, -2.00198122],
       [ 1.35355383,  4.58741051],
       [-2.3588036 ,  9.24487109],
       [ 5.49855613, -0.40760904],
       [ 4.95905577,  0.28044654],
       [-2.74076277,  8.21875431],
       [ 9.31532594, -1.91140836],
       [ 4.96916517,  0.51172185],
       [ 6.59547199,  1.74661785],
       [ 1.01364733,  1.61537219],
       [ 2.19693999,  1.43744038],
       [ 4.83221652,  1.68411226],
       [ 8.43557108, -3.58222938],
       [ 2.62671806,  1.24400622],
       [ 9.0456817 , -3.28997926],
       [-2.83658148,  2.97165496],
       [-1.80280307,  7.50314965],
       [ 6.22330403, -0.24202182],
       [ 1.17616111,  1.477019  ],
       [ 9.11595959, -3.01767084],
       [10.51984712, -1.60650047],
       [-1.00486364,  8.35297727],
       [ 1.07991917,  0.52551662],
       [-2.07397966,  1.02798958],
       [ 0.20506647,  4.34794091],
       [-0.68183016,  7.4438482 ],
       [-0.90906638,  5.47875617],
       [ 6.51498084,  0.64441954],
       [-2.87240165,  3.43670473],
       [ 1.48208476,  0.92471963],
       [ 4.12055513,  0.11936196],
       [ 9.30609118, -3.73937746],
       [ 6.76974245,  1.68121707],
       [ 6.57895861,  0.48005705],
       [ 1.59599706,  4.4134228 ],
       [-1.83649815,  9.14138051],
       [-1.83291774,  6.80349422],
       [ 6.76639558,  0.37881496],
       [ 1.37236931,  0.31154635],
       [10.06333455, -1.83627872],
       [ 5.43768205, -0.6264718 ],
       [ 0.31184654,  4.05393102],
       [ 7.35602562, -3.20030044],
       [-1.6398525 ,  2.66476001],
       [-0.54742218,  3.83555806],
       [ 0.93674622,  0.50531675],
       [ 6.95576711, -0.1721576 ],
       [ 2.30876997,  2.02509167],
       [ 6.00390593,  0.53488539],
       [ 9.92491891, -2.02832073],
       [-1.7821302 ,  7.42658696],
       [-1.16458409,  3.49325409],
       [ 9.09738709, -1.58301755],
       [ 5.4359662 , -1.2933267 ],
       [ 6.57311973,  0.81323952],
       [-1.54137137,  3.24016124],
       [ 9.51163066, -3.17532469],
       [ 1.01516482,  4.14468105],
       [-2.21826363,  1.67142158],
       [ 0.39350859,  2.88196368],
       [ 9.98864178, -2.54320942],
       [-1.82966504,  9.24607819],
       [ 3.06509284,  1.16706488],
       [-0.47405958,  5.96194641],
       [ 8.04370072, -1.65091637],
       [-0.93909607,  3.50841903],
       [-2.97433757,  9.59027895],
       [ 2.24623729,  4.1294028 ],
       [-1.38227274,  9.75367499],
       [-2.14366178,  1.7420229 ],
       [ 9.86694379, -2.32796437],
       [-1.56029441,  1.92500279],
       [ 1.24237762,  3.57780595],
       [ 5.16926598,  0.48406749],
       [-1.57540361,  7.91565887],
       [ 0.54306572,  3.93142417],
       [ 1.82746466,  1.5797516 ],
       [ 0.06613035,  5.2000713 ],
       [-2.15553256,  4.22944467],
       [-2.62583239,  3.1448755 ],
       [ 2.90556151,  3.0675765 ],
       [ 0.97535527,  1.72165086],
       [-0.59219336,  3.47485643],
       [ 6.15982333,  0.54879246],
       [ 0.28509574,  2.83644712],
       [ 5.46118967,  0.73203606],
       [ 5.53916771,  0.08382019],
       [-0.25659215,  1.15839499],
       [-2.38237548,  6.69075014],
       [ 5.7956524 ,  0.7653299 ],
       [-1.54758122,  2.29055599],
       [ 4.44676141, -0.75470417],
       [ 0.68054475,  4.43668395],
       [ 5.40710657,  2.62848402],
       [-2.50965908,  3.58190065],
       [ 7.76999753, -2.0290101 ]  # X5
])

print("=" * 80)
print("L'ALGORITHME AVEC OPTIMISATION DE FUSION DES CLUSTERS")
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

                # Calculer les densités
                density1 = calculate_cluster_density(cluster1, final_points, h)
                density2 = calculate_cluster_density(cluster2, final_points, h)
                

                # Fusionner en utilisant la moyenne pondérée
                merged, d1, d2 = merge_clusters_weighted(cluster1, cluster2, final_points, h)
                
                # Calcul du poids relatif pour le centre fusionné
                total_density = d1 + d2
                weight1 = d1 / total_density
                weight2 = d2 / total_density
                
                new_center = weight1 * center1 + weight2 * center2
                

                
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
ax.set_xlim(-4, 15)
ax.set_ylim(-4, 11)

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

ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.set_xlim(-4, 15)
ax.set_ylim(-4,11)

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
ax.set_xlim(-4, 15)
ax.set_ylim(-4, 11)

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
ax.set_xlim(-4, 15)
ax.set_ylim(-4, 11)

plt.tight_layout()
plt.savefig('result_two_optimized_result.png', dpi=300, bbox_inches='tight')
print("\n✅ Graphique sauvegardé : meanshift_optimized_result.png")

print("\n" + "=" * 80)
print("FIN DE L'ANALYSE OPTIMISÉE")
print("=" * 80)