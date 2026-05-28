import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ==========================================
# 1. Φόρτωση Εικόνων (UAV Ground Target)
# ==========================================
print("Φόρτωση εικόνων...")
img_target = mpimg.imread('img_desired.jpeg')
img_init = mpimg.imread('img_initial.jpeg')

# ==========================================
# 2. Επιλογή Σημείων (Desired Position)
# ==========================================
plt.figure('Επιλογή Στόχου (Desired)', figsize=(10, 8))
plt.imshow(img_target)
plt.title('Κάνε κλικ σε 4 γωνίες (TL, TR, BR, BL) και μετά κλείσε το παράθυρο')
points_star = plt.ginput(4, timeout=0)
plt.close()

s_star_pix = np.zeros((8, 1))
for i in range(4):
    s_star_pix[2*i, 0] = points_star[i][0]
    s_star_pix[2*i+1, 0] = points_star[i][1]

# ==========================================
# 3. Επιλογή Σημείων (Initial Position)
# ==========================================
plt.figure('Επιλογή Αρχικής Θέσης (Initial)', figsize=(10, 8))
plt.imshow(img_init)
plt.title('Κάνε κλικ στις ΙΔΙΕΣ 4 γωνίες με την ίδια σειρά')
points_init = plt.ginput(4, timeout=0)
plt.close()

s_init_pix = np.zeros((8, 1))
for i in range(4):
    s_init_pix[2*i, 0] = points_init[i][0]
    s_init_pix[2*i+1, 0] = points_init[i][1]

# ==========================================
# 4. Παράμετροι Κάμερας Drone
# ==========================================
print("Αρχικοποίηση παραμέτρων κάμερας...")
f = 3.5e-3
pw = 1e-6
ph = 1e-6
fx = f / pw
fy = f / ph
cu = 3060 / 2
cv = 4080 / 2

# ==========================================
# 5. IBVS Control Loop Setup
# ==========================================
print("Εκκίνηση Visual Servoing Control Loop...")
lam = 0.5
dt = 0.05
iterations = 250
Z_est = 0.1  # Εκτίμηση βάθους (10cm)

s = np.copy(s_init_pix)
s_star = np.copy(s_star_pix)

error_history = np.zeros(iterations)
v_linear = np.zeros((3, iterations))
v_angular = np.zeros((3, iterations))
s_history = np.zeros((8, iterations))

# ==========================================
# 6. Κύριος Βρόχος Ελέγχου (Control Loop)
# ==========================================
for k in range(iterations):
    L = np.empty((0, 6))
    for i in range(4):
        u = s[2*i, 0]
        v = s[2*i+1, 0]

        # Κανονικοποιημένες Συντεταγμένες
        x = (u - cu) / fx
        y = (v - cv) / fy

        # Πίνακας Αλληλεπίδρασης (Interaction Matrix)
        L_norm = np.array([
            [-1/Z_est, 0, x/Z_est, x*y, -(1+x**2), y],
            [0, -1/Z_est, y/Z_est, 1+y**2, -x*y, -x]
        ])

        # Μετατροπή σε pixel velocity interaction matrix
        Li = np.array([[fx, 0], [0, fy]]) @ L_norm
        L = np.vstack((L, Li))

    # Σφάλμα Χαρακτηριστικών
    e = s - s_star
    error_history[k] = np.linalg.norm(e)

    # Νόμος Ελέγχου (Ταχύτητα Κάμερας με Ψευδοαντίστροφο)
    vc = -lam * np.linalg.pinv(L) @ e

    # Αποθήκευση Δεδομένων
    v_linear[:, k] = vc[0:3, 0]
    v_angular[:, k] = vc[3:6, 0]
    s_history[:, k] = s[:, 0]

    # Δυναμική Χαρακτηριστικών (Update)
    s_dot = L @ vc
    s = s + dt * s_dot

# ==========================================
# 7. Γραφήματα (Plots)
# ==========================================
print("Δημιουργία γραφημάτων...")

# Γράφημα 1: Σφάλμα
plt.figure('Feature Error', figsize=(8, 5))
plt.plot(range(iterations), error_history, linewidth=2)
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Error Norm (pixels)')
plt.title('IBVS Feature Error over Time')

# Γράφημα 2: Ταχύτητες
plt.figure('Camera Velocities', figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(v_linear[0, :], label='v_x', linewidth=2)
plt.plot(v_linear[1, :], label='v_y', linewidth=2)
plt.plot(v_linear[2, :], label='v_z', linewidth=2)
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Velocity (m/s)')
plt.title('UAV Linear Velocities')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(v_angular[0, :], label='w_x', linewidth=2)
plt.plot(v_angular[1, :], label='w_y', linewidth=2)
plt.plot(v_angular[2, :], label='w_z', linewidth=2)
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('UAV Angular Velocities')
plt.legend()

# Γράφημα 3: Τροχιές (Trajectories)
plt.figure('Feature Trajectories', figsize=(10, 8))
plt.imshow(img_init)
colors = ['b', 'c', 'm', 'y']

for i in range(4):
    # Desired (Πράσινα Αστέρια)
    plt.plot(s_star_pix[2*i, 0], s_star_pix[2*i+1, 0], 'g*', markersize=12, label='Desired' if i==0 else "")
    # Initial (Κόκκινοι Κύκλοι)
    plt.plot(s_init_pix[2*i, 0], s_init_pix[2*i+1, 0], 'ro', markersize=8, label='Initial' if i==0 else "")
    # Τροχιά (Γραμμές)
    plt.plot(s_history[2*i, :], s_history[2*i+1, :], color=colors[i], linewidth=2, label=f'Point {i+1}')

plt.title('2D UAV Camera Feature Trajectories')
plt.legend(loc='upper right')
plt.show()