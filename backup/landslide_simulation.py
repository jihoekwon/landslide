"""
Landslide Simulation using Cellular Automata
=============================================
This module simulates landslide dynamics on a Digital Elevation Model (DEM).

The simulation uses a simplified cellular automata approach where:
- Each cell has an elevation and material thickness
- Material flows to lower neighboring cells when slope exceeds critical angle
- Flow rate depends on slope steepness and material properties

Author: Claude Code Demo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LightSource
import os


class DEMGenerator:
    """Generate synthetic Digital Elevation Model for testing."""

    @staticmethod
    def create_mountain_terrain(size=(100, 100), seed=42):
        """
        Create a synthetic mountain terrain with valleys and ridges.

        Args:
            size: Tuple of (rows, cols) for the DEM grid
            seed: Random seed for reproducibility

        Returns:
            2D numpy array representing elevation in meters
        """
        np.random.seed(seed)
        rows, cols = size

        # Create base terrain using multiple frequency components
        x = np.linspace(0, 4*np.pi, cols)
        y = np.linspace(0, 4*np.pi, rows)
        X, Y = np.meshgrid(x, y)

        # Mountain ridge running diagonally
        terrain = 500 + 300 * np.sin(0.5*X + 0.3*Y) * np.cos(0.2*X - 0.4*Y)

        # Add a prominent peak
        center_x, center_y = cols // 3, rows // 3
        for i in range(rows):
            for j in range(cols):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                terrain[i, j] += 400 * np.exp(-dist**2 / (2 * (size[0]/4)**2))

        # Add smaller scale roughness
        terrain += 50 * np.random.randn(rows, cols)

        # Smooth the terrain
        from scipy.ndimage import gaussian_filter
        terrain = gaussian_filter(terrain, sigma=2)

        # Add a steep slope area (potential landslide source)
        slope_center_x, slope_center_y = int(cols * 0.35), int(rows * 0.35)
        for i in range(rows):
            for j in range(cols):
                dist = np.sqrt((i - slope_center_y)**2 + (j - slope_center_x)**2)
                if dist < size[0] / 6:
                    terrain[i, j] += 200 * (1 - dist / (size[0]/6))

        return terrain

    @staticmethod
    def save_dem(terrain, filename):
        """Save DEM to a numpy binary file."""
        np.save(filename, terrain)
        print(f"DEM saved to {filename}")

    @staticmethod
    def load_dem(filename):
        """Load DEM from a numpy binary file."""
        terrain = np.load(filename)
        print(f"DEM loaded from {filename}: shape {terrain.shape}")
        return terrain


class LandslideSimulator:
    """
    Cellular Automata based landslide simulator.

    Uses a simplified physics model where:
    - Material accumulates on terrain
    - When slope exceeds critical angle, material flows downhill
    - Flow volume depends on excess slope and material properties
    """

    def __init__(self, dem, cell_size=10.0):
        """
        Initialize the landslide simulator.

        Args:
            dem: 2D numpy array of terrain elevation
            cell_size: Size of each cell in meters
        """
        self.dem = dem.copy().astype(float)
        self.cell_size = cell_size
        self.rows, self.cols = dem.shape

        # Material thickness on each cell (meters)
        self.material = np.zeros_like(dem, dtype=float)

        # Track total surface (DEM + material)
        self.surface = self.dem.copy()

        # Simulation parameters
        self.critical_slope = 0.7  # tan(35 degrees) - typical angle of repose
        self.flow_rate = 0.3  # Fraction of excess material that flows per step
        self.friction = 0.1  # Energy loss per step

        # Velocity field for momentum
        self.velocity = np.zeros_like(dem, dtype=float)

        # History for animation
        self.history = []

    def initialize_landslide(self, center, radius, thickness):
        """
        Initialize a landslide source area.

        Args:
            center: (row, col) tuple for landslide center
            radius: Radius of initial failure zone in cells
            thickness: Initial material thickness in meters
        """
        row_c, col_c = center
        for i in range(self.rows):
            for j in range(self.cols):
                dist = np.sqrt((i - row_c)**2 + (j - col_c)**2)
                if dist < radius:
                    # Gaussian distribution of material
                    self.material[i, j] = thickness * np.exp(-dist**2 / (2*(radius/2)**2))

        self.surface = self.dem + self.material
        self._save_state()

    def _get_neighbors(self, i, j):
        """Get valid neighbor indices (8-connectivity)."""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    # Distance factor (diagonal = sqrt(2))
                    dist = np.sqrt(di**2 + dj**2) * self.cell_size
                    neighbors.append((ni, nj, dist))
        return neighbors

    def _calculate_slope(self, i, j, ni, nj, dist):
        """Calculate slope between two cells."""
        elevation_diff = self.surface[i, j] - self.surface[ni, nj]
        return elevation_diff / dist

    def step(self):
        """
        Perform one simulation step.

        Returns:
            total_flow: Total material flow in this step
        """
        new_material = self.material.copy()
        new_velocity = self.velocity.copy()
        total_flow = 0

        # Process each cell with material
        for i in range(self.rows):
            for j in range(self.cols):
                if self.material[i, j] < 0.01:  # Skip cells with negligible material
                    continue

                neighbors = self._get_neighbors(i, j)

                # Find downslope neighbors
                outflows = []
                for ni, nj, dist in neighbors:
                    slope = self._calculate_slope(i, j, ni, nj, dist)

                    if slope > self.critical_slope:
                        # Excess slope drives flow
                        excess = slope - self.critical_slope
                        outflows.append((ni, nj, excess, dist))

                if not outflows:
                    # No flow possible, material stops
                    new_velocity[i, j] *= (1 - self.friction)
                    continue

                # Calculate total excess for normalization
                total_excess = sum(o[2] for o in outflows)

                # Available material for flow (limited by flow rate and current material)
                available = min(self.material[i, j],
                              self.material[i, j] * self.flow_rate * (1 + self.velocity[i, j]))

                # Distribute material to downslope neighbors
                for ni, nj, excess, dist in outflows:
                    fraction = excess / total_excess
                    flow_amount = available * fraction

                    new_material[i, j] -= flow_amount
                    new_material[ni, nj] += flow_amount

                    # Transfer momentum
                    new_velocity[ni, nj] = max(new_velocity[ni, nj],
                                               self.velocity[i, j] * (1 - self.friction) +
                                               0.1 * excess)

                    total_flow += flow_amount

        # Update state
        self.material = np.maximum(new_material, 0)
        self.velocity = new_velocity * (1 - self.friction)
        self.surface = self.dem + self.material

        return total_flow

    def _save_state(self):
        """Save current state to history."""
        self.history.append({
            'surface': self.surface.copy(),
            'material': self.material.copy(),
            'velocity': self.velocity.copy()
        })

    def run(self, max_steps=200, min_flow=0.1, save_interval=1):
        """
        Run the simulation until equilibrium or max steps.

        Args:
            max_steps: Maximum number of steps
            min_flow: Stop when total flow is below this threshold
            save_interval: Save state every N steps

        Returns:
            Number of steps executed
        """
        print(f"Starting landslide simulation...")
        print(f"Initial material volume: {self.material.sum():.1f} cubic meters")

        for step_num in range(max_steps):
            total_flow = self.step()

            if step_num % save_interval == 0:
                self._save_state()

            if step_num % 20 == 0:
                print(f"Step {step_num}: flow = {total_flow:.2f}, "
                      f"max velocity = {self.velocity.max():.3f}")

            if total_flow < min_flow and step_num > 10:
                print(f"Equilibrium reached at step {step_num}")
                break

        print(f"Simulation complete. {len(self.history)} frames saved.")
        return step_num + 1


class LandslideVisualizer:
    """Visualize landslide simulation results with matplotlib animation."""

    def __init__(self, simulator):
        """
        Initialize visualizer.

        Args:
            simulator: LandslideSimulator instance with history
        """
        self.simulator = simulator
        self.history = simulator.history
        self.dem = simulator.dem

    def create_animation(self, output_file='landslide_animation.gif',
                        fps=10, dpi=100):
        """
        Create animated visualization of the landslide.

        Args:
            output_file: Output filename (gif or mp4)
            fps: Frames per second
            dpi: Resolution
        """
        fig = plt.figure(figsize=(14, 6))

        # Create subplots
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        # Light source for terrain shading
        ls = LightSource(azdeg=315, altdeg=45)

        # Get data ranges for consistent coloring
        all_surfaces = [h['surface'] for h in self.history]
        vmin = min(s.min() for s in all_surfaces)
        vmax = max(s.max() for s in all_surfaces)

        all_materials = [h['material'] for h in self.history]
        mat_max = max(m.max() for m in all_materials)

        # Create meshgrid for 3D plot
        x = np.arange(self.simulator.cols) * self.simulator.cell_size
        y = np.arange(self.simulator.rows) * self.simulator.cell_size
        X, Y = np.meshgrid(x, y)

        def init():
            ax1.clear()
            ax2.clear()
            return []

        def update(frame):
            ax1.clear()
            ax2.clear()

            state = self.history[frame]
            surface = state['surface']
            material = state['material']

            # 3D surface plot
            # Color based on material thickness
            colors = plt.cm.terrain(np.interp(surface, [vmin, vmax], [0, 1]))

            # Highlight areas with material
            material_mask = material > 0.1
            colors[material_mask] = plt.cm.Reds(
                np.interp(material[material_mask], [0, mat_max], [0.3, 1])
            )

            ax1.plot_surface(X, Y, surface, facecolors=colors,
                           shade=True, antialiased=True,
                           rstride=2, cstride=2)

            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Elevation (m)')
            ax1.set_title(f'Landslide Simulation - Frame {frame}/{len(self.history)-1}')

            # Set consistent view
            ax1.set_zlim(vmin - 50, vmax + 50)
            ax1.view_init(elev=35, azim=225)

            # 2D material thickness map
            im = ax2.imshow(material, cmap='hot', origin='lower',
                          extent=[0, self.simulator.cols * self.simulator.cell_size,
                                 0, self.simulator.rows * self.simulator.cell_size],
                          vmin=0, vmax=mat_max)

            # Add contours of terrain
            ax2.contour(X, Y, self.dem, levels=15, colors='gray',
                       alpha=0.5, linewidths=0.5)

            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title(f'Material Thickness (m)\nTotal: {material.sum():.0f} m³')

            if frame == 0:
                plt.colorbar(im, ax=ax2, label='Thickness (m)')

            plt.tight_layout()
            return []

        print(f"Creating animation with {len(self.history)} frames...")
        anim = animation.FuncAnimation(fig, update, init_func=init,
                                       frames=len(self.history),
                                       interval=1000/fps, blit=False)

        # Save animation
        print(f"Saving animation to {output_file}...")
        if output_file.endswith('.gif'):
            anim.save(output_file, writer='pillow', fps=fps, dpi=dpi)
        else:
            anim.save(output_file, writer='ffmpeg', fps=fps, dpi=dpi)

        print(f"Animation saved to {output_file}")
        plt.close()

        return output_file

    def plot_final_state(self, output_file='landslide_final.png'):
        """Plot the final state of the simulation."""
        fig = plt.figure(figsize=(16, 10))

        # Get final state
        final = self.history[-1]
        initial = self.history[0]

        # 1. 3D view of final terrain
        ax1 = fig.add_subplot(221, projection='3d')
        x = np.arange(self.simulator.cols) * self.simulator.cell_size
        y = np.arange(self.simulator.rows) * self.simulator.cell_size
        X, Y = np.meshgrid(x, y)

        surf = ax1.plot_surface(X, Y, final['surface'], cmap='terrain',
                               antialiased=True, rstride=2, cstride=2)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Elevation (m)')
        ax1.set_title('Final Terrain Surface')
        ax1.view_init(elev=35, azim=225)

        # 2. Material distribution comparison
        ax2 = fig.add_subplot(222)
        diff = final['material'] - initial['material']
        im = ax2.imshow(diff, cmap='RdBu_r', origin='lower',
                       extent=[0, self.simulator.cols * self.simulator.cell_size,
                              0, self.simulator.rows * self.simulator.cell_size])
        ax2.contour(X, Y, self.dem, levels=15, colors='gray',
                   alpha=0.5, linewidths=0.5)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Material Change (Initial to Final)\nBlue: Erosion, Red: Deposition')
        plt.colorbar(im, ax=ax2, label='Change (m)')

        # 3. Initial material
        ax3 = fig.add_subplot(223)
        im3 = ax3.imshow(initial['material'], cmap='hot', origin='lower',
                        extent=[0, self.simulator.cols * self.simulator.cell_size,
                               0, self.simulator.rows * self.simulator.cell_size])
        ax3.contour(X, Y, self.dem, levels=15, colors='gray',
                   alpha=0.5, linewidths=0.5)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_title(f'Initial Material Distribution\nTotal: {initial["material"].sum():.0f} m³')
        plt.colorbar(im3, ax=ax3, label='Thickness (m)')

        # 4. Final material
        ax4 = fig.add_subplot(224)
        im4 = ax4.imshow(final['material'], cmap='hot', origin='lower',
                        extent=[0, self.simulator.cols * self.simulator.cell_size,
                               0, self.simulator.rows * self.simulator.cell_size])
        ax4.contour(X, Y, self.dem, levels=15, colors='gray',
                   alpha=0.5, linewidths=0.5)
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_title(f'Final Material Distribution\nTotal: {final["material"].sum():.0f} m³')
        plt.colorbar(im4, ax=ax4, label='Thickness (m)')

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Final state plot saved to {output_file}")
        plt.close()

        return output_file


def main():
    """Run a demonstration landslide simulation."""
    print("=" * 60)
    print("LANDSLIDE SIMULATION DEMONSTRATION")
    print("=" * 60)

    # Step 1: Generate synthetic DEM
    print("\n[1/4] Generating synthetic terrain DEM...")
    dem_generator = DEMGenerator()
    terrain = dem_generator.create_mountain_terrain(size=(80, 100), seed=42)

    # Save DEM file
    dem_file = 'terrain_dem.npy'
    dem_generator.save_dem(terrain, dem_file)

    print(f"Terrain shape: {terrain.shape}")
    print(f"Elevation range: {terrain.min():.1f} - {terrain.max():.1f} m")

    # Step 2: Initialize simulator
    print("\n[2/4] Initializing landslide simulator...")
    simulator = LandslideSimulator(terrain, cell_size=10.0)

    # Initialize landslide at a steep slope area
    # Find the steepest area (near the peak we created)
    landslide_center = (28, 35)  # Near the artificial peak
    landslide_radius = 12  # cells
    landslide_thickness = 15.0  # meters of material

    simulator.initialize_landslide(landslide_center, landslide_radius, landslide_thickness)
    print(f"Landslide initialized at {landslide_center}")
    print(f"Initial failure zone radius: {landslide_radius} cells ({landslide_radius * simulator.cell_size} m)")
    print(f"Initial material thickness: {landslide_thickness} m")

    # Step 3: Run simulation
    print("\n[3/4] Running simulation...")
    steps = simulator.run(max_steps=150, min_flow=0.5, save_interval=2)
    print(f"Simulation completed in {steps} steps")

    # Step 4: Visualize results
    print("\n[4/4] Creating visualizations...")
    visualizer = LandslideVisualizer(simulator)

    # Create animation
    animation_file = visualizer.create_animation('landslide_animation.gif', fps=8, dpi=80)

    # Create final state plot
    final_plot = visualizer.plot_final_state('landslide_final.png')

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Output files:")
    print(f"  - DEM data: {dem_file}")
    print(f"  - Animation: {animation_file}")
    print(f"  - Final state: {final_plot}")
    print("\nOpen the animation file to see the landslide progression!")


if __name__ == '__main__':
    main()
