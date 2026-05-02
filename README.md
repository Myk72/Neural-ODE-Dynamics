# Neural ODE-Based Continuous-Time Modeling System

A deep learning framework for modeling continuous-time dynamical systems using Neural Ordinary Differential Equations (Neural ODEs).

## Overview

This project learns the underlying dynamics of time-series data by parameterizing a differential equation with a neural network:

dx/dt = fθ(x, t)

Instead of discrete predictions, the model evolves system states continuously over time using differentiable ODE solvers.

## Key Features

* Neural network parameterization of differential equations
* Differentiable ODE solvers (Euler / Runge-Kutta)
* Backpropagation through time using adjoint methods
* Continuous-time prediction and interpolation
* Visualization of learned dynamics