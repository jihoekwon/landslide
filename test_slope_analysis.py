"""
DEM과 경사각 분포 비교 시각화
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
d = np.load('guryoung_dem_10m/simulation_results.npz')
terrain = d['terrain']
cell_size = float(d['cell_size'])

print(f"DEM shape: {terrain.shape}")
print(f"Cell size: {cell_size} m")
print(f"Elevation range: {terrain.min():.1f} ~ {terrain.max():.1f} m")

# 경사 계산 (np.gradient 사용)
grad_y, grad_x = np.gradient(terrain, cell_size)
slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
slope_deg = np.degrees(slope_rad)

print(f"Slope range: {slope_deg.min():.1f} ~ {slope_deg.max():.1f} degrees")
print(f"Mean slope: {slope_deg.mean():.1f} degrees")

# 2패널 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 좌측: DEM
im1 = axes[0].imshow(terrain, cmap='terrain', origin='lower',
                      extent=[0, terrain.shape[1]*cell_size, 0, terrain.shape[0]*cell_size])
axes[0].set_title(f'DEM 고도 분포\n({terrain.min():.0f} ~ {terrain.max():.0f} m)', fontsize=14)
axes[0].set_xlabel('X (m)')
axes[0].set_ylabel('Y (m)')
cbar1 = plt.colorbar(im1, ax=axes[0], label='고도 (m)')

# 우측: 경사각
im2 = axes[1].imshow(slope_deg, cmap='YlOrRd', origin='lower',
                      extent=[0, terrain.shape[1]*cell_size, 0, terrain.shape[0]*cell_size])
axes[1].set_title(f'경사각 분포 (np.gradient 기반)\n({slope_deg.min():.1f} ~ {slope_deg.max():.1f}°)', fontsize=14)
axes[1].set_xlabel('X (m)')
axes[1].set_ylabel('Y (m)')
cbar2 = plt.colorbar(im2, ax=axes[1], label='경사각 (°)')

# 마찰각 기준선 표시
phi_bed = float(d['phi_bed'])
axes[1].contour(slope_deg, levels=[phi_bed], colors='blue', linewidths=2,
                extent=[0, terrain.shape[1]*cell_size, 0, terrain.shape[0]*cell_size])
axes[1].text(100, 100, f'파란선: 마찰각 {phi_bed}°', color='blue', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('guryoung_dem_10m/dem_slope_comparison.png', dpi=150)
print("\nSaved: guryoung_dem_10m/dem_slope_comparison.png")
plt.show()
