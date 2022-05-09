#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_CMSAK8`
DATADIR=${DATADIR_CMSAK8}
[[ -z $DATADIR ]] && DATADIR='/mldata/licq/deepjetak8'

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

epochs=50
samples_per_epoch=$((10000 * 1024 / $NGPUS))
samples_per_epoch_val=$((10000 * 128))
dataopts="--num-workers 12 --fetch-step 0.01"

# PN, PFN, PCNN, ParT
model=$1
[[ -z ${model} ]] && model="ParT"
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformerTagger.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
# elif [[ "$model" == "PN" ]]; then
#     modelopts="networks/example_ParticleNet.py"
#     batchopts="--batch-size 512 --start-lr 1e-2"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin", "kinpid", "full"
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="MD"

if ! [[ "${FEATURE_TYPE}" =~ ^(Full|MD)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

$CMD \
    --data-train \
    "BulkGravitonToHHTo4Q_HWW:${DATADIR}/20220301_ak8_UL17_v3/BulkGravitonToHHTo4Q_MX-600to6000_MH-15to250_HWW_JHUGen_v3/*/*.root" \
    "QCD_Pt_170to300_TuneCP5_13TeV_pythia8:${DATADIR}/20200601_ak8_UL17_flat/QCD_Pt_170to300_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_300to470_TuneCP5_13TeV_pythia8:${DATADIR}/20200601_ak8_UL17_flat/QCD_Pt_300to470_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_470to600_TuneCP5_13TeV_pythia8:${DATADIR}/20200601_ak8_UL17_flat/QCD_Pt_470to600_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_600to800_TuneCP5_13TeV_pythia8:${DATADIR}/20200601_ak8_UL17_flat/QCD_Pt_600to800_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_800to1000_TuneCP5_13TeV_pythia8:${DATADIR}/20200601_ak8_UL17_flat/QCD_Pt_800to1000_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8:${DATADIR}/20200601_ak8_UL17_flat/QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8:${DATADIR}/20200601_ak8_UL17_flat/QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8:${DATADIR}/20200601_ak8_UL17_flat/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    "QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8:${DATADIR}/20200601_ak8_UL17_flat/QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8/*/*/*/output_*.root" \
    --data-config data/CMSAK8/CMSHWWAK8_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/CMSHWWAK8/${FEATURE_TYPE}/${model}/{auto}${suffix}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --gpus 0 \
    --optimizer ranger --log logs/CMSHWWAK8_${FEATURE_TYPE}_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard CMSHWWAK8_${FEATURE_TYPE}_${model}${suffix} \
    "${@:3}"
