# Quantum-sensor-simulation (julia)
Simulate the interaction between a two particle system and a chain of qubits using julia.

## Set up

1. Start a Julia session in the directory of the project.
2. Activate the Pkg manager by typing `]`
3. Run `activate .` and then `instantiate`

Now all the dependencies of the environment have been installed and you can use it.

## Usage

Modify the parameters of the simulation using `config.yml` and run

```
$ julia --threads [#THREADS] --project main.jl config.yml
```