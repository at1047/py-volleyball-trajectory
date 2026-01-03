# Volleyball Setting Trajectory Solver

The Volleyball Trajectory Simulator is an interactive tool for modeling and visualizing volleyball set trajectories under realistic flight dynamics. The project uses my preferred way of defining a volleyball set:start position, target position, and desired airtime, rather than ambiguous descriptors such as “higher” or “faster.”

The simulator solves an inverse trajectory problem to compute feasible launch parameters. Candidate solutions are evaluated by solving the resulting initial value problem (IVP), numerically integrating the ball’s dynamics forward in time using odeint, and set quality is measured by the time the trajectory spends within a hitter-defined hitting window.



To build:
```
docker buildx build --platform linux/amd64,linux/arm64 --push -t at1047/volleyball-trajectory 
```
