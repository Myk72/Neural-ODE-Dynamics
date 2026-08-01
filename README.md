# Neural ODE-Based Continuous-Time Modeling System

A deep learning framework for learning and predicting continuous-time dynamical
systems with Neural Ordinary Differential Equations (Neural ODEs).

## Overview

The project learns how a system changes over time by representing its instantaneous
rate of change with a neural network:

$$
\frac{d\mathbf{x}(t)}{dt}=f_\theta(t,\mathbf{x}(t)).
$$

Instead of directly predicting only the next sampled value, the model learns a
continuous vector field. A differentiable ODE solver integrates this field from an
initial state to produce a complete trajectory at requested times.

## Goal

The goal is to recover useful system dynamics from observed trajectories and use
them for continuous-time interpolation, forecasting, and phase-space analysis. The
initial focus is on low-dimensional systems such as damped oscillators and
interacting populations, where predictions can be compared with known behaviour.

## How It Works

1. Time-series trajectories provide timestamps, observed states, and initial
   conditions.
2. A neural network estimates the derivative of the state at each point in time.
3. An ODE solver integrates those derivatives to generate predicted trajectories.
4. Prediction errors are propagated through the solver to update the neural network.
5. The learned dynamics are evaluated using rollout accuracy, extrapolation,
   numerical cost, and phase-space behaviour.

## Key Features

- Neural-network parameterization of continuous-time dynamics
- Differentiable Euler, Runge--Kutta, and adaptive ODE solvers
- Prediction at arbitrary timestamps
- Continuous-time interpolation and forecasting
- Reproducible experiment and trajectory-data contracts
- Visualization and evaluation of learned dynamics
