import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def plot_smooth_ordered_pairs(list1, list2, labels=None, colors=None, line_styles=None, title="Smooth Ordered Pairs Plot"):
    """
    Plot two lists of ordered pairs with smooth interpolated lines.
    
    Parameters:
    - list1, list2: Lists of (x, y) tuples
    - labels: Tuple of legend labels (label1, label2)
    - colors: Tuple of line colors (color1, color2)
    - line_styles: Tuple of line styles (style1, style2)
    - title: Plot title
    """
    # Default styling
    if labels is None:
        labels = ('List 1', 'List 2')
    if colors is None:
        colors = ('#1f77b4', '#ff7f0e')  # Matplotlib default blue and orange
    if line_styles is None:
        line_styles = ('-', '--')
    
    # Convert to numpy arrays and sort by x values
    arr1 = np.array(sorted(list1, key=lambda x: x[0]))
    arr2 = np.array(sorted(list2, key=lambda x: x[0]))
    
    # Create figure with nice styling
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.gca()
    
    # Plot each list with smooth interpolation
    for arr, label, color, line_style in zip([arr1, arr2], labels, colors, line_styles):
        x, y = arr[:, 0], arr[:, 1]
        
        # Create smooth curve using cubic spline interpolation
        if len(x) > 3:  # Need at least 4 points for cubic spline
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spl = make_interp_spline(x, y, k=3)  # Cubic spline
            y_smooth = spl(x_smooth)
        else:  # Fall back to linear interpolation if not enough points
            x_smooth = np.linspace(x.min(), x.max(), 100)
            y_smooth = np.interp(x_smooth, x, y)
        
        # Plot with markers for original points
        plt.plot(x_smooth, y_smooth, color=color, linestyle=line_style, 
                linewidth=2.5, alpha=0.8, label=label)
        plt.scatter(x, y, color=color, s=60, edgecolors='white', zorder=3)
    
    # Styling
    ax.set_facecolor('#f8f9fa')  # Light gray background
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d1d1d1')
    ax.spines['bottom'].set_color('#d1d1d1')
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(frameon=True, framealpha=1, facecolor='white')
    
    plt.tight_layout()
    plt.show()


#UOT
list1 = [(50, .2), (200, 0.478), (500, 0.591)]
#OT
list2 = [(50, .260), (200, .598), (500, 0.733)]
    
# Plot with custom styling
plot_smooth_ordered_pairs(
        list1, list2,
        labels=('UOT', 'OT'),
        colors=('#4e79a7', '#e15759'),
        line_styles=('-', '-'),
        title="Comparison of OT and UOT with k =2, geom = 0.001, entropy = .1 and reg_m = .1"
)