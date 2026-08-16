# GraphBot mathematics

This guide explains the models currently implemented in GraphBot's browser UI and the experimental research planners retained in `GraphBot.py`. It is a companion to the showcase [README](../README.md), not a guarantee that every generated expression is ideal for every Graphwar map.

## Coordinates and expressions

The captured field is mapped linearly to the approximate Graphwar rectangle:

$$x\in[-25,25],\qquad y\in[-15,15].$$

For a field image of width $W$ and height $H$, pixel $(p_x,p_y)$ becomes

$$
x=-25+50\frac{p_x}{W},\qquad y=15-30\frac{p_y}{H}.
$$

GraphBot emits direct-function expressions. In Graphwar's normal-function mode, the game vertically translates the entered function to pass through the active soldier, so a constant offset does not change the shot. Review [GAME_RULES.md](../GAME_RULES.md) for parser syntax, differential-equation modes, and game-level limits.

## Piecewise paths from absolute values

Click Mode and linear Draw Mode use the same building block. Between two points,

$$
d=-\frac{y_1-y_2}{2(x_2-x_1)},\qquad
s(x)=d\bigl(|x-x_1|-|x-x_2|\bigr).
$$

The sum of these V-shaped pieces creates a route through ordered waypoints. If a route turns back in $x$, the UI creates a near-vertical transition instead of requiring a true vertical graph, which cannot be represented as a single-valued $y=f(x)$.

## Draw-mode data

A freehand stroke is resampled to a training set. Draw Mode can prevent backward drawing, which preserves the one-$y$-per-$x$ assumption used by every current formula exporter. The selected method is evaluated against these sampled points; the UI displays mean squared error (MSE) for comparison.

## Sigmoid network

The shallow sigmoid model is

$$
f(x)=b+\sum_{i=1}^{N}\frac{w_i}{1+\exp(-k(x-x_i))}.
$$

$x_i$ are step locations, $w_i$ are learned step heights, and $k$ controls steepness. This is useful for accumulating smooth changes. The UI exposes the number of neurons, $k$, epochs, learning rate, and initialization choices.

## Taylor / polynomial features

Before polynomial features are evaluated, GraphBot normalizes position:

$$t=\frac{x-c}{s},\qquad \phi(t)=[1,t,t^2,\ldots,t^n].$$

With no hidden layers, the model is a linear combination of these features—a polynomial in $t$. With one or more hidden layers, the feature vector is passed through the feature MLP described below. “Taylor” is a useful UI name for this power-basis model, but it is not a symbolic Taylor-series derivation around a known analytic function.

## Fourier features

The Fourier model uses a normalized coordinate $t$ and basis

$$
\phi(t)=[1,\cos(\pi t),\sin(\pi t),\ldots,\cos(K\pi t),\sin(K\pi t)].
$$

With zero hidden layers the model is a harmonic linear combination. Hidden layers make this a feature MLP rather than a strict Fourier series. More harmonics increase its ability to describe fine oscillation, but can also create unnecessarily complicated expressions.

## Feature MLPs and activations

Taylor and Fourier share the same small fully connected network:

$$
\phi(x)\rightarrow W_1\phi(x)+b_1\rightarrow a(\cdot)\rightarrow\cdots\rightarrow y.
$$

The implemented hidden-layer activations are:

- `tanh`
- logistic sigmoid
- ReLU
- Leaky ReLU ($\alpha=0.01$)
- Softplus
- Swish / SiLU
- GELU approximation
- Mish

The UI trains these networks in JavaScript with gradient descent. ReLU and Leaky ReLU use He-style initialization; other choices use a Xavier-style range. The browser exporter turns network layers into expression text. Formula size grows quickly with depth and width, so a numerically good model may still be inconvenient to paste into the game.

### Exporting activations without `max`

Graphwar expressions work well with arithmetic and `abs`, so ReLU has the exact identity

$$
\operatorname{ReLU}(z)=\frac{z+|z|}{2}.
$$

Leaky ReLU is exported by combining positive and negative absolute-value parts. Softplus, Swish, GELU, Mish, sigmoid, and `tanh` are emitted in their corresponding expression forms by the current UI exporter. Always inspect a copied formula if your Graphwar build has different parser support; [GAME_RULES.md](../GAME_RULES.md) is the project reference for the game syntax used here.

## Cubic spline and B-spline

The cubic spline interpolator creates a cubic on each interval:

$$
S_i(x)=a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3.
$$

Natural boundary conditions set endpoint second derivatives to zero. The UI's clamped option uses zero endpoint first derivatives. The exporter expresses the piecewise cubic through truncated-power terms based on `abs`, avoiding a conditional expression.

The B-spline fit uses cubic basis functions:

$$S(x)=\sum_i c_iB_i^{(3)}(x).$$

It fits coefficients by a regularized least-squares solve. The control-point count controls flexibility; the smoothing parameter adds diagonal regularization. The exporter first converts the fitted spline into piecewise cubic intervals, then serializes those intervals for Graphwar.

## Trajectory Search: evolutionary solver

The UI's Trajectory Search (formerly Dot Mode) evolves a population of paths. A genome contains $y$ values at fixed, increasing $x$ knots:

$$g=[y_0,y_1,\ldots,y_{m-1}],\qquad y_0=y_{\mathrm{start}}.$$

The initial population combines noisy target-guided paths with random walks. New generations retain elites, use tournament parent selection, blend parents with crossover, then mutate individual genes. Mutation cools as generations advance and occasionally uses broad restarts when the search stagnates.

Each genome is rendered as piecewise linear or as a sampled natural cubic spline. Paths are ranked lexicographically rather than by one opaque scalar:

1. in-field trajectories beat out-of-bounds trajectories;
2. lower constraint penalty wins, including a forbidden-mask collision penalty when enabled;
3. more target hits win;
4. lower summed miss distance wins;
5. lower edge-strip penalty, then lower excess length, wins.

Targets are hit when their distance to the path polyline is within the configured hit radius. Search currently moves only right, so targets to the left of the start are marked unreachable.

## Forbidden region handling

On field capture, Python thresholds very dark pixels, removes detected player regions, applies morphology, filters tiny components, and dilates the result by a safety margin. The mask is downsampled to a run-length-encoded occupancy grid sent to the browser. Trajectory Search samples its candidate trajectory against this grid and adds the resulting penalty to its ranking.

This makes the grid the collision source of truth for the UI search. It is still image-based: calibration and unusual map graphics can affect it.

## Legacy research planners

`GraphBot.py` is not the primary UI, but it contains experimental alternatives: a right-moving A* waypoint chain, a ridge-regularized polynomial search, and a symbolic genetic algorithm over expression trees. They use player and obstacle detection from the Python pipeline and should be treated as research tools, not as the standard workflow.
