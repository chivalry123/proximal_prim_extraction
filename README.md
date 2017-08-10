# proximal_prim_extraction

## added implementation of one more method called --AlgoPartialSnapShotOfPoscar 1
Just add --AlgoPartialSnapShotOfPoscar 1 to the following lines to activate it

## generate multiple ouput
python extract_small_confgs_from_GMC_batch.py --MAXSIZE 64 -pr example_1/original/PRIM -po example_1/original/POSCAR.ideal --WriteToPoscar example_1/extracted/output_poscar --BatchSize 10 --TrialMultiplier 10 


## if you have large supercell, it may be useful to do save and load 
python extract_small_confgs_from_GMC_batch.py --MAXSIZE 64 -pr example_1/original/PRIM -po example_1/original/POSCAR.ideal --WriteToPoscar example_1/extracted/output_poscar --BatchSize 10 --TrialMultiplier 10 --SaveErrorVectorObjFile test001

python extract_small_confgs_from_GMC_batch.py --MAXSIZE 64 -pr example_1/original/PRIM -po example_1/original/POSCAR.ideal --WriteToPoscar example_1/extracted/output_poscar --BatchSize 10 --TrialMultiplier 10 --LoadErrorVectorObjFile test001


## Just generate one output (not recommended, outdated , deprecated, I will not update this part anymore)
python extract_small_confgs_from_GMC.py --MAXSIZE 64 -pr example_1/PRIM -po example_1/POSCAR.ideal --WriteToPoscar example_1/POSCAR_OUTPUT

