# Graphwar Rules and Tutorial

Source project: [catabriga/graphwar](https://github.com/catabriga/graphwar)

Graphwar is an artillery game in which you must hit your enemies using mathematical functions.
The trajectory of your shot is determined by the function you write.
Your goal is to avoid obstacles and teammates and hit enemies.
The game takes place in a Cartesian plane.

## Game Modes

### Normal Function

The Normal Function mode is the most basic mode.
In this mode, the shot is exactly the function you typed, but with one important adjustment:
the function is translated vertically so it passes through your soldier.

So if you enter:

`y = f(x)`

the actual trajectory used by the game is:

`y = f(x) + c`

where `c` is the constant needed to pass through the soldier position.

### First-Order Differential Equation

In this mode, you enter a first-order differential equation instead of a direct function.

Examples:

- `y' = 3*sin(x)+2`
- `y' = -y/3`
- `y' = 1/(x+y)`

No translation constant is added in this mode.
Your soldier position is used as the initial condition, and the fired curve is the resulting solution.

### Second-Order Differential Equation

This mode is similar to first-order mode, but with second-order equations.

Examples:

- `y'' = -y + y' + 2*x - 1`
- `y'' = 4*sin(x) + 2^x`
- `y'' = 1.04^(-(x+y)^2)`

To get a unique solution, this mode uses two initial conditions:

- soldier position;
- firing angle.

You can change the firing angle with the up/down keys.
This is the only game mode where firing angle affects the curve.

## Common Pitfalls

### Constants in normal mode do not matter

Because the function is translated by `+c`, constants in your entered function become irrelevant.
These produce the same trajectory in-game:

- `y = 2*x + 3`
- `y = 2*x - 8`
- `y = 2*x`

### Field limits can make functions look wrong

Graphwar axis limits are approximately:

- `x: -25 .. 25`
- `y: -15 .. 15`

Functions can grow very fast.
For example, `y = x^2` is `100` at `x = 10`, so it can hit the top quickly and look almost vertical.
Scaling helps, for example:

- `y = (x^2)/50`

### Your side usually has negative x

Your team is on the left side of the plane, so your soldier often starts at negative `x`.
That is why `sqrt(x)` often explodes immediately.
Safer form:

- `y = sqrt(abs(x))`

### "Explosion" usually means invalid or too long curve

A function can explode if:

- value is undefined (for example `log(x)` at negative `x`);
- denominator goes to zero;
- square root/log argument is invalid;
- curve becomes too long (for example high-frequency oscillations like `sin(100*x)`).

## Function Syntax

### Variables

- `x`
- `y`
- `y'`

### Operators

- `+`
- `-`
- `/`
- `*`
- `^`

### Functions

- `sqrt()`
- `log()`
- `ln()`
- `abs()`
- `sin()`
- `cos()`
- `tan()`
- `exp()`

### Other Examples

- `y = ((x-3)^2)/20`
- `y = ln(abs(x))`
- `y = sin(x/20)*5`
- `y' = 1.2^x`
- `y'' = (1.2^(-(x+3)^2))*(20*(-y))`

Use parentheses generously to avoid misinterpretation.
For example:

- `1/x+2` means `(1/x) + 2`
- use `1/(x+2)` if that is what you want

## Chat Commands

- `-skip` — if everyone uses this command, the current map is skipped.
- `-sayfunc` — show other players' functions in chat.
- `-stopsayfunc` — stop showing functions after `-sayfunc`.
- `-shownext` — highlight each player's next soldier.
- `-stopshownext` — stop showing next soldiers.

Type commands directly in the game chat.

## FAQ

### Is my soldier at the origin?

No.
Your soldier is not at the origin.
In normal mode, the function is translated in `y` so it passes through your soldier at that soldier's `x`.

### My function looks like a vertical line. Why?

At your current soldier `x`, the function may have a very large value.
After translation, the visible part can become extremely steep.
Try scaling down (for example divide by `10` or `50`).

### My function explodes immediately or randomly. Why?

Most common reasons:

- undefined value at some point (`log(x)` for negative `x`, invalid square root, division by zero);
- function length limit reached (high-frequency curves can be too long).

### I change angle but nothing happens. Why?

Angle only affects second-order differential equation mode.
In other modes, changing angle does nothing.

### People cannot join games I host. Why?

Most likely your router blocks incoming connections.
You may need to open/forward the required port in router settings.

### Any other questions?

Original project contact from tutorial:

- `graphwar.contact@gmail.com`

