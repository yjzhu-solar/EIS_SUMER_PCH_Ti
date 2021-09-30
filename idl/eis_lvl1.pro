pro eis_lvl1

;filenames = file_search('../raw_data/north_pole/20071116/eis_l0_*.fits')
;output_dir = "../raw_data/north_pole/20071116/level1/"

filenames = file_search('../raw_data/linewidth_calib/eis_l0_*.fits')
output_dir = "../raw_data/linewidth_calib/"

for i = 0, sizeof(filenames) - 1 do begin
    eis_prep, filenames[i],outdir=output_dir, /default, /quiet, /save
end
print,"end"
end