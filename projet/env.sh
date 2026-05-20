salloc -t 4:00:00 --account=r250123 --constraint="armgpu" --cores=48 -N 4 --mem=10G --gpus=4 --exclusive
srun --pty bash
romeo_load_armgpu_env
spack load gcc@11.4.1 /nrzc6ai
spack load openmpi@4.1.7 /nkokjyt
spack load cuda@12.6.2 /3mzltpz
