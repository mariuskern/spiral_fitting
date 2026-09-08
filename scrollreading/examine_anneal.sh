#!/bin/sh

for N in {2..2}; do

  cp d:/annealRuns/PHerc0139_sip_20k/annealState_out_$N.csv d:/pipelineOutput/manualBadPatch.csv
  ./simpaper10 v
  ./simpaper10 h
  ./simpaper10 f 30 <<HEREDOC
g
q
HEREDOC
  
  ./render_from_zarr6 d:/zarrs/PHerc0139/volume/2 d:/pipelineOutput/patch_0.bin - -c d:/pipelineOutput/patch_0_colours.csv

  cp d:/pipelineOutput/patch_0.tif d:/annealRuns/PHerc0139_sip_20k/patch_0_$N.tif


done