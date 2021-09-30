pro sumer_image_check
filename_338 = file_search('../raw_data/sumer/20071116/level1/sum_r_20071116_09*0338_l1.sav')

for ii = 0, 2 do begin
    restore, filename_338[ii],/ver
    window,ii,xs=200,ys=600
    plot_image,image_lvl1[192-22:192+14,*]
end
end