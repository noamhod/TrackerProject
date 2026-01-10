import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator

def draw_detector_final_v9():
    # --- GEOMETRY SETUP ---
    # Plot Limits
    Y_PLOT_MIN = -10.0
    Y_PLOT_MAX = 190.0
    X_PLOT_MIN = -60.0 
    X_PLOT_MAX = 220.0
    
    # Interface at Z=0
    Z_JOINT = 0.0
    
    # Heights
    # Positron chamber sits on top of pipe/step at 51.65
    Y_PIPE_TOP_LEFT = 51.65     
    Y_PIPE_TOP_RIGHT = 51.65    
    
    # Window (Blue)
    GAP_WINDOW = 1.85
    Y_WINDOW_BOTTOM = Y_PIPE_TOP_LEFT + GAP_WINDOW # 53.5 mm
    HEIGHT_WINDOW = 119.8
    Y_WINDOW_TOP = Y_WINDOW_BOTTOM + HEIGHT_WINDOW
    WIDTH_WINDOW = 1.0
    
    # Chips (Green)
    NUM_CHIPS = 5
    CHIP_PITCH = 20.0
    CHIP_Z_WIDTH = 1.0 
    
    Y_CHIP_BOTTOM = 75.05 
    HEIGHT_CHIP = 29.94
    Y_CHIP_TOP = Y_CHIP_BOTTOM + HEIGHT_CHIP
    
    # Detector Box (Solid Gray)
    GAP_PIPE_TO_BOX = 1.525
    Y_BOX_BOTTOM = Y_PIPE_TOP_RIGHT + GAP_PIPE_TO_BOX 
    
    # Box Symmetry
    Y_CHIP_CENTER_VERT = Y_CHIP_BOTTOM + (HEIGHT_CHIP / 2)
    box_half_height = Y_CHIP_CENTER_VERT - Y_BOX_BOTTOM
    Y_BOX_TOP = Y_CHIP_CENTER_VERT + box_half_height
    
    # Z Locations
    # DIST_CHAMBER_TO_CHIP1 = 114.3
    DIST_CHAMBER_TO_CHIP1 = 114.3+10.5
    DIST_BOX_TO_CHIP1 = 10.5
    
    Z_CHIP1_START = Z_JOINT + DIST_CHAMBER_TO_CHIP1
    Z_BOX_FRONT = Z_CHIP1_START - DIST_BOX_TO_CHIP1
    
    # Chip Positions
    chips_z_starts = [Z_CHIP1_START + i * CHIP_PITCH for i in range(NUM_CHIPS)]
    
    # Box Z Length
    z_chips_min = chips_z_starts[0]
    z_chips_max = chips_z_starts[-1] + CHIP_Z_WIDTH
    z_chips_center_horz = (z_chips_min + z_chips_max) / 2
    
    dist_center_to_front = z_chips_center_horz - Z_BOX_FRONT
    Z_BOX_BACK = z_chips_center_horz + dist_center_to_front

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(10,7)) # Adjusted figsize to better fit aspect ratio
    
    # 1. Left Beam Pipe (Light Gray, No Frame)
    rect_bp_left = patches.Rectangle((X_PLOT_MIN, Y_PLOT_MIN), Z_JOINT - X_PLOT_MIN, Y_PIPE_TOP_LEFT - Y_PLOT_MIN, 
                                     linewidth=0, facecolor='#d3d3d3')
    ax.add_patch(rect_bp_left)

    # 2. Positron Vacuum Chamber (Darker Gray, No Frame)
    rect_vac = patches.Rectangle((X_PLOT_MIN, Y_PIPE_TOP_LEFT), Z_JOINT - X_PLOT_MIN, Y_PLOT_MAX - Y_PIPE_TOP_LEFT, 
                                 linewidth=0, facecolor='#a9a9a9', label='Positron vac. chamber')
    ax.add_patch(rect_vac)
    
    # 3. Right Beam Pipe (Light Gray, No Frame)
    rect_bp_right = patches.Rectangle((Z_JOINT, Y_PLOT_MIN), X_PLOT_MAX - Z_JOINT, Y_PIPE_TOP_RIGHT - Y_PLOT_MIN, 
                                      linewidth=0, facecolor='#d3d3d3', label='Beam Pipe')
    ax.add_patch(rect_bp_right)
    
    # 4. Detector Box (Solid Aluminum Gray, No Frame)
    rect_box = patches.Rectangle((Z_BOX_FRONT, Y_BOX_BOTTOM), Z_BOX_BACK - Z_BOX_FRONT, Y_BOX_TOP - Y_BOX_BOTTOM, 
                                 linewidth=0, facecolor='#bdbdbd', alpha=1.0, label='Detector Box (Al)')
    ax.add_patch(rect_box)

    # 5. Chips (Thin Green Lines)
    for z in chips_z_starts:
        rect_chip = patches.Rectangle((z, Y_CHIP_BOTTOM), CHIP_Z_WIDTH, HEIGHT_CHIP, 
                                      linewidth=0, facecolor='limegreen', label='Chips' if z == chips_z_starts[0] else "")
        ax.add_patch(rect_chip)

    # 6. Kapton Windows (Orange Lines)
    kapton_y_min = Y_CHIP_BOTTOM - 10.0
    kapton_y_max = Y_CHIP_TOP + 10.0
    ax.plot([Z_BOX_FRONT, Z_BOX_FRONT], [kapton_y_min, kapton_y_max], 
            color='darkorange', linewidth=3, solid_capstyle='butt', label='Kapton Window')
    ax.plot([Z_BOX_BACK, Z_BOX_BACK], [kapton_y_min, kapton_y_max], 
            color='darkorange', linewidth=3, solid_capstyle='butt')

    # 7. Vacuum Exit Window (Blue)
    rect_win = patches.Rectangle((-WIDTH_WINDOW/2, Y_WINDOW_BOTTOM), WIDTH_WINDOW, HEIGHT_WINDOW, 
                                 linewidth=0, facecolor='blue', alpha=1.0, label='Exit Window')
    ax.add_patch(rect_win)
    
    # 8. Beam Axis
    ax.axhline(y=0, color='black', linestyle='-.', linewidth=1, label="Beam Axis")

    # --- ANNOTATIONS ---
    
    def add_dim(x1, y1, x2, y2, text, color='black', text_offset=(0, 0), 
                text_on_arrow=False, ha='center', va='center', rotation=0, 
                bbox_style=dict(boxstyle='square,pad=0.2', fc='white', ec='none', alpha=0.9)):
        ax.annotate('', xy=(x1, y1), xytext=(x2, y2), 
                    arrowprops=dict(arrowstyle='<->', color=color, lw=1.5))
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        if text_on_arrow:
            ax.text(mid_x, mid_y, text, color=color, 
                    ha='center', va='center', fontsize=9, rotation=rotation,
                    bbox=bbox_style)
        else:
            ax.text(mid_x + text_offset[0], mid_y + text_offset[1], text, color=color, 
                    ha=ha, va=va, fontsize=9, rotation=rotation,
                    bbox=dict(boxstyle='square,pad=0', fc='white', ec='none', alpha=0.7))

    # 1. 53.5 mm Label - MOVED TO Z=-5
    # Points to Window Bottom (53.5)
    add_dim(-5, 0, -5, Y_WINDOW_BOTTOM, "53.5 mm", color='black', text_on_arrow=True, rotation=90)

    # 2. 51.65 mm Label
    add_dim(60, 0, 60, Y_PIPE_TOP_RIGHT, "51.65 mm", color='black', text_on_arrow=True, rotation=90)

    # 3. Red X + 1.85 mm Gap
    # Gap between 51.65 and 53.5
    mid_gap_y = (Y_PIPE_TOP_LEFT + Y_WINDOW_BOTTOM) / 2
    x_mark_z = 0.0
    ax.text(x_mark_z, mid_gap_y, "X", color='red', fontsize=10, ha='center', va='center', fontweight='bold', zorder=20)
    ax.annotate('1.85 mm', xy=(x_mark_z, mid_gap_y), xytext=(10, 57.5), 
                arrowprops=dict(arrowstyle='-', color='red', lw=0.5), color='red', fontsize=8, va='center', ha='left')

    # 4. Window Height (119.8 mm)
    add_dim(-8, Y_WINDOW_BOTTOM, -8, Y_WINDOW_TOP, "119.8 mm", color='black', text_on_arrow=True, rotation=90)

    # 5. 1.525 mm (Pipe to Box)
    mid_gap_box = (Y_PIPE_TOP_RIGHT + Y_BOX_BOTTOM) / 2
    x_gap_box = Z_BOX_FRONT + 5
    ax.text(x_gap_box, mid_gap_box, "X", color='red', fontsize=10, ha='center', va='center', fontweight='bold', zorder=20)
    ax.annotate('1.525 mm', xy=(x_gap_box+2, mid_gap_box), xytext=(x_gap_box+15, 57.5), 
                 arrowprops=dict(arrowstyle='-', color='red', lw=0.5), color='red', fontsize=8, va='center')

    # 6. 75.05 mm
    # add_dim(Z_CHIP1_START - 20, 0, Z_CHIP1_START - 20, Y_CHIP_BOTTOM, "75.05 mm", color='black', text_on_arrow=True, rotation=90)
    add_dim(183, 0, 183, Y_CHIP_BOTTOM, "75.05 mm", color='black', text_on_arrow=True, rotation=90)

    # 7. 21.88 mm (Horizontal)
    z_dim_box = Z_CHIP1_START + 2 * CHIP_PITCH 
    add_dim(z_dim_box, Y_BOX_BOTTOM, z_dim_box, Y_CHIP_BOTTOM, "21.88 mm", color='black', text_on_arrow=True, rotation=0)

    # 8. Chip Height (29.94 mm - Horizontal)
    x_dim_chip_h = 183 #z_dim_box + 15
    add_dim(x_dim_chip_h, Y_CHIP_BOTTOM, x_dim_chip_h, Y_CHIP_TOP, "29.94 mm", color='black', text_on_arrow=True, rotation=0)

    # # 9. 114.3 mm
    # z_chip1_center = Z_CHIP1_START + (CHIP_Z_WIDTH / 2)
    # ax.plot([z_chip1_center, z_chip1_center], [Y_BOX_TOP, 185], 'k--', lw=0.8)
    # ax.plot([0, 0], [Y_PLOT_MAX, 185], 'k:', lw=0.8)
    # add_dim(0, 180, z_chip1_center, 180, "114.3 mm", color='black')
    
    # 9. 114.3 mm to the kapton window
    z_box_front = Z_BOX_FRONT
    ax.plot([z_box_front, z_box_front], [Y_BOX_TOP, 185], 'k--', lw=0.8) 
    ax.plot([0, 0], [Y_PLOT_MAX, 185], 'k:', lw=0.8)
    add_dim(0, 180, z_box_front, 180, "114.3 mm", color='black')


    # 10. 10.5 mm
    ax.plot([Z_BOX_FRONT, Z_BOX_FRONT], [Y_BOX_TOP, 125], 'k:', lw=0.5)
    ax.plot([Z_CHIP1_START, Z_CHIP1_START], [Y_BOX_TOP, 125], 'k:', lw=0.5)
    add_dim(Z_BOX_FRONT, 120, Z_CHIP1_START, 120, "10.5 mm", color='black', text_offset=(0, 2), ha='center', va='bottom')

    # 11. NEW: Chip Pitch Arrow (20 mm)
    # Between 3rd and 4th chip (index 2 and 3)
    z_chip1 = chips_z_starts[0]
    z_chip2 = chips_z_starts[1]
    y_pitch_arrow = Y_CHIP_CENTER_VERT
    # add_dim(z_chip1, y_pitch_arrow, z_chip2, y_pitch_arrow, "20 mm", color='black', text_on_arrow=True)
    add_dim(z_chip1, y_pitch_arrow, z_chip2, y_pitch_arrow, "20 mm", color='black', text_offset=(0, 2), ha='center', va='bottom')

    # 12. Dipole Exit Plane
    ax.annotate('', xy=(-60, 90), xytext=(0, 90), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(-30, 90, "To dipole exit plane\nz = -3032.155 mm", ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='none', alpha=0.9))

    # --- PLOT SETTINGS ---
    ax.set_aspect('equal')
    
    # Strictly enforce limits
    ax.set_xlim(X_PLOT_MIN, X_PLOT_MAX)
    ax.set_ylim(Y_PLOT_MIN, Y_PLOT_MAX)
    
    ax.set_xlabel(r"$z$ [mm]", fontsize=14)
    ax.set_ylabel(r"$y$ [mm]", fontsize=14)
    # ax.set_title("Detector Ara Layout", fontsize=14)
    
    # Grid & Ticks
    major_locator_x = MultipleLocator(20) # 20mm steps for X might look cleaner with pitch 20
    major_locator_y = MultipleLocator(10)
    # minor_locator = MultipleLocator(1) # 1mm minor
    
    ax.xaxis.set_major_locator(major_locator_x)
    # ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_major_locator(major_locator_y)
    # ax.yaxis.set_minor_locator(minor_locator)
    
    minor_locator_x = MultipleLocator(2)
    minor_locator_y = MultipleLocator(2)
    
    ax.xaxis.set_minor_locator(minor_locator_x)
    ax.yaxis.set_minor_locator(minor_locator_y)
    
    ax.grid(which='major', color='#CCCCCC', linestyle='-', linewidth=1.2)
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.8)
    
    # Ensure ticks on all sides
    ax.tick_params(axis='both', which='both', direction='out', top=True, right=True, labelbottom=True)

    ax.legend(loc='upper right', framealpha=1.0, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"detector_area_sketch.pdf")
    plt.show()

if __name__ == "__main__":
    draw_detector_final_v9()