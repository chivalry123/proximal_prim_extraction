# proximal_prim_extraction

## generate multiple ouput
python extract_small_confgs_from_GMC_batch.py --MAXSIZE 64 -pr example_1/PRIM -po example_1/POSCAR.ideal --WriteToPoscar example_1/extracted/output_poscar --BatchSize 10 --TrialMultiplier 10

## Just generate one output (not recommended)
python extract_small_confgs_from_GMC.py --MAXSIZE 64 -pr example_1/PRIM -po example_1/POSCAR.ideal --WriteToPoscar example_1/POSCAR_OUTPUT

