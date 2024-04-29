#!/bin/bash

#SBATCH -J test               # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 2                  # Total number of nodes requested
#SBATCH -n 8                  # Total number of mpi tasks requested
#SBATCH -t 01:30:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p mi1004x            # Desired partition

# Launch an MPI-based executable

prun ./a.out

# salloc -N 1 -n 1 -p devel -t 00:30:00