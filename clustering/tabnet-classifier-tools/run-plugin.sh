#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
inp_dir=/data/inputs

# Output paths
outDir=/data/output

#Additional args
filePattern=".*.csv"
testSize=0.2
nD=8
nA=8
nSteps=3
gamma=1.3
catEmbDim=1
nIndependent=2
nShared=2
epsilon=1e-15
seed=0
momentum=0.02
clipValue=""
lambdaSparse=1e-3
optimizerFn="Adam" 
lr=0.02
schedulerFn="StepLR"
stepSize=10
deviceName="auto"
maskType="entmax"
groupedFeatures=""
nSharedDecoder=1
nIndepDecoder=1
evalMetric="auc"
maxEpochs=200
patience=10
weights=0
lossFn="L1Loss"
batchSize=1024
virtualBatchSize=128
numWorkers=0
dropLast="false"
warmStart="false"
computeImportance="true"
classifier="TabNetClassifier"
targetVar="income"


# Show the help options
# docker run polusai/pytorch-tabnet-tool:${version}

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/pytorch-tabnet-tool:${version} \
            --inpdir ${inpDir} \
            --filePattern ${filePattern} \
            --testSize ${testSize} \
            --nD ${nD} \
            --nA ${nA} \
            --nSteps ${nSteps} \
            --gamma ${gamma} \
            --catEmbDim ${catEmbDim} \
            --nIndependent ${nIndependent} \
            --nShared ${nShared} \
            --epsilon ${epsilon} \
            --seed ${seed} \
            --momentum ${momentum} \
            --clipValue ${clipValue} \
            --lambdaSparse ${lambdaSparse} \
            --optimizerFn ${optimizerFn} \
            --lr ${lr} \
            --schedulerFn ${schedulerFn} \
            --stepSize ${stepSize} \
            --deviceName ${deviceName} \
            --maskType ${maskType} \
            --groupedFeatures ${groupedFeatures} \
            --nSharedDecoder ${nSharedDecoder} \
            --nIndepDecoder ${nIndepDecoder} \
            --evalMetric ${evalMetric} \
            --maxEpochs ${maxEpochs} \
            --patience ${patience} \
            --weights ${weights} \
            --lossFn ${lossFn} \
            --batchSize ${batchSize} \
            --virtualBatchSize ${virtualBatchSize} \
            --numWorkers ${numWorkers} \
            --dropLast ${dropLast} \
            --warmStart ${warmStart} \
            --computeImportance ${computeImportance} \
            --targetVar ${targetVar} \
            --classifier ${classifier} \
            --outdir ${outDir} \
