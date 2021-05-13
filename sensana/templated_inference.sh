#!/bin/bash

mkdir -p satest_yamls
mkdir -p satest_logs

echo -n "" > inference_jobs.txt

for n in {40..59}
do

  export specfile="satest_yamls/satest.inference.${n}.yaml"
  cp satest.inference.template.yaml ${specfile}
  sed -i "s/__RNG_SEED__/${n}/g" ${specfile}
  echo "pf inference --spec ${specfile} |& tee satest_logs/satest.inference.${n}.log" >> inference_jobs.txt

done

parallel --jobs 5 < inference_jobs.txt
