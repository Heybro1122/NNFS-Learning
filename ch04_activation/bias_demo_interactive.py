"""
Interactive Bias Demo with Sliders
Experiment with weights and biases to see how they affect the graph
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Activation function
def relu(x):
    return np.maximum(0, x)

# The network function: 2 layers of 1 neuron each
def forward_pass(x, w1, b1, w2, b2):
    # Neuron 1
    layer1_out = relu(w1 * x + b1)
    
    # Neuron 2 (takes output of neuron 1 as input)
    # Note: The book example often uses linear activation for the last neuron 
    # to see the full effect, but let's stick to ReLU if that's what we're testing.
    # If we want to strictly follow the "up/down" shift logic for the *final* graph 
    # even if it goes negative, the last layer usually doesn't have ReLU in regression.
    # However, based on the user's context of "ReLU activation functions" (plural),
    # I will apply ReLU to both, but keep in mind ReLU clips negative values to 0.
    
    layer2_out = relu(w2 * layer1_out + b2)
    return layer2_out

# Setup the plot
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(left=0.1, bottom=0.35) # Make room for sliders

x = np.linspace(-5, 5, 500)
w1_init, b1_init = 1.0, 0.0
w2_init, b2_init = 1.0, 0.0

# Initial plot
y = forward_pass(x, w1_init, b1_init, w2_init, b2_init)
line, = ax.plot(x, y, lw=2, color='blue')

# Visual guides
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title("Network: y = ReLU(w2 * ReLU(w1*x + b1) + b2)")
ax.set_ylim(-2, 5)
ax.set_xlim(-5, 5)
ax.grid(True)

# Add Sliders
axcolor = 'lightgoldenrodyellow'

# Slider positions [left, bottom, width, height]
ax_w1 = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_b1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_w2 = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_b2 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

s_w1 = Slider(ax_w1, 'Weight 1 (Slope)', -5.0, 5.0, valinit=w1_init)
s_b1 = Slider(ax_b1, 'Bias 1 (Left/Right)', -5.0, 5.0, valinit=b1_init)
s_w2 = Slider(ax_w2, 'Weight 2', -5.0, 5.0, valinit=w2_init)
s_b2 = Slider(ax_b2, 'Bias 2 (Up/Down)', -5.0, 5.0, valinit=b2_init)

# Update function
def update(val):
    w1 = s_w1.val
    b1 = s_b1.val
    w2 = s_w2.val
    b2 = s_b2.val
    
    # Recalculate y
    y_new = forward_pass(x, w1, b1, w2, b2)
    line.set_ydata(y_new)
    fig.canvas.draw_idle()

# Attach update function to sliders
s_w1.on_changed(update)
s_b1.on_changed(update)
s_w2.on_changed(update)
s_b2.on_changed(update)

# Reset button
resetax = plt.axes([0.8, 0.01, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    s_w1.reset()
    s_b1.reset()
    s_w2.reset()
    s_b2.reset()
button.on_clicked(reset)

plt.show()
