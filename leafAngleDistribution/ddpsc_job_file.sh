####################
#
# Example Vanilla Universe Job
# Multi-CPU HTCondor submit description file
#
####################

universe         = vanilla
getenv           = true
accounting_group = $ENV(CONDOR_GROUP)
request_cpus     = 40

log              = leafAngle.log
output           = leafAngle.out
error            = leafAngle.error

executable       = /home/zli/pyWorkSpace/leafAngleDistribution/ddpsc_leaf_angle.sh
arguments        = 
queue