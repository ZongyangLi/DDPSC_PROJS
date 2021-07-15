#PBS -S /bin/bash
#PBS -m bae
#PBS -M zongyangli86@gmail.com
#PBS -N lad_06_29
#PBS -l nodes=1:ppn=10
#PBS -l walltime=48:00:00

module purge
module load gdal-stack-2.7.10 nco

source /home/zongyang/angleDistribution/codes/venv/bin/activate

bash /home/zongyang/angleDistribution/codes/angle_distribution.sh 6 29 29
