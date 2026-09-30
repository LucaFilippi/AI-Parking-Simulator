# Parking AI Simulator

A Python simulator where a neural network learns to park a car using evolutionary training.

## Features

* 2D car physics and collision system
* Configurable parking spots, obstacles, and other cars
* Visual track editor
* Distance sensors
* Neural network with configurable architecture
* Evolutionary training
* Headless training for faster simulations
* Fitness and collision statistics
* Save and load trained AI models
* Save and load tracks

## How It Works

The AI receives information such as:

* Car speed and orientation
* Distance and angle to the parking spot
* Parking alignment errors
* Distance sensor readings

It controls:

* Steering
* Throttle
* Brake
* Handbrake

Each generation evaluates a population of neural networks. The best networks are selected, mutated, and used to create the next generation.

Collisions, especially with other cars and obstacles, heavily reduce fitness.

## Project Structure

```text
parking-ai/
├── main.py
├── ai.py
├── requirements.txt
├── models/
└── tracks/
```

## Technologies

* Python
* Pygame
* NumPy

## Goal

Teach an AI to reliably park a car in different environments without using visual input.

> **Parking AI — teaching a machine to park, one generation at a time.**
