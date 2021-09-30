pro sumer_wvl_cal_607

filenames = file_search('../raw_data/sumer/20071116/level1/sum_r_20071116_09*0607_l1.sav')

image_aver = fltarr(1024,360)
for ii = 0, 1  do begin
    restore, filenames[ii],/ver
    image_aver = image_aver + image_lvl1
end

print,ref_pix,wvl_ref
image_aver = image_aver/3.0
window,0,xs=1000,ys=600
plot_image,alog10(image_aver)
window,1,xs=600,ys=400
image_yaver = mean(image_aver,dimension=1)
print,image_yaver[0:40]
print,image_yaver[299:330]
plot,indgen(360),image_yaver
oplot,fltarr(2)+299,!y.crange

spectrum_line_id = mean(image_aver[*,200:299],dimension=2)
;window,2,xs=600,ys=400
;plot,indgen(1024),spectrum_line_id
spec_gauss_widget,indgen(1024),spectrum_line_id,fltarr(1024)+0.01
end