#!/bin/bash

#set -x

sh_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

src_dir="$(realpath "${sh_dir}/src")"

data_dir="$(realpath "${sh_dir}/data")"

if (( $# != 7 )); then
    echo "Usage: run_model.s model_name pii alpha chi lambda rhoseed run"
    exit 1
fi


model=$1
pii=$(python3 -c "print('{:.3f}'.format($2))")
alpha=$(python3 -c "print('{:.2f}'.format($3))")
chi=$(python3 -c "print('{:.2f}'.format($4))")
lambda=$(python3 -c "print('{:.1f}'.format($5))")
rhoseed=$(python3 -c "print('{:.2f}'.format($6))")
run=$(python3 -c "print('{:d}'.format($7))")

invgamma0=40.0 #40.0
zeta0=0.1
p0=1.5
KQ=10.0
K=1
T=8 #20
n_steps=8e+4
dt_dump=0.1
rho_in=150
rhoisoend=400
rhonemend=700 
rrhoend=150
ncluster=16 
rcluster=3.2
rhocluster=180
rhobg=15
mx=100
my=100
dx=1
dy=1

save_dir="${sh_dir}/data/cluster/$model/ncluster_${ncluster}_rhocluster_${rhocluster}_rhobg_${rhobg}/invgamma0_${invgamma0}_rhoseed_${rhoseed}_KQ_${KQ}/pii_${pii}_alpha_${alpha}_chi_${chi}_lambda_${lambda}/run_${run}"


if [ ! -d $save_dir ]; then
    mkdir -p $save_dir
fi

params_file="${save_dir}/parameters.json"

echo \
"
{
    "\"run\"" : $run,
    "\"T\"" : $T,
    "\"n_steps\"" : $n_steps,
    "\"dt_dump\"" : $dt_dump,
    "\"K\"" : $K,
    "\"invgamma0\"" : $invgamma0,
    "\"alpha\"" : $alpha,
    "\"zeta0\"" : $zeta0,
    "\"chi\"": $chi,
    "\"lambda\"": $lambda,
    "\"KQ\"": $KQ,
    "\"pii\"": $pii,
    "\"p0\"": $p0,
    "\"rhoseed\"" : $rhoseed,
    "\"rho_in\"" : $rho_in,
    "\"rhoisoend\"" : $rhoisoend,
    "\"rhonemend\"" : $rhonemend,
    "\"rrhoend\"" : $rrhoend,
    "\"ncluster\"" : $ncluster,
    "\"rcluster\"" : $rcluster,
    "\"rhocluster\"" : $rhocluster,
    "\"rhobg\"" : $rhobg,
    "\"mx\"" : $mx,
    "\"my\"" : $my,
    "\"dx\"" : $dx,
    "\"dy\"" : $dy
}
" > $params_file

python3 -m models.$model -s $save_dir

python3 -m src.analysis.create_avgs -s $save_dir

python3 -m src.analysis.create_videos_rho -s $save_dir