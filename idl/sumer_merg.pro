pro sumer_merg
    int_merg = dblarr(4096,300) 
    filename_338 = file_search('../raw_data/sumer/20071116/level1/sum_r_20071116_09*0338_l1.sav')
    image_aver = dblarr(1024,360)
    for ii = 0, 2 do begin
        restore, filename_338[ii],/ver
        image_aver = image_aver + image_lvl1
    end
    image_aver = image_aver/3.0
    int_merg[0:1023,*] = congrid(image_aver[*,6:294],1024,300)

    filename_375 = file_search('../raw_data/sumer/20071116/level1/sum_r_20071116_09*0375_l1.sav')
    image_aver = dblarr(1024,360)
    for ii = 0, 2 do begin
        restore, filename_375[ii],/ver
        image_aver = image_aver + image_lvl1
    end
    image_aver = image_aver/3.0
    int_merg[1024:2047,*] = congrid(image_aver[*,6:299],1024,300)

    filename_509 = file_search('../raw_data/sumer/20071116/level1/sum_r_20071116_09*0509_l1.sav')
    image_aver = dblarr(1024,360)
    for ii = 0, 2 do begin
        restore, filename_509[ii],/ver
        image_aver = image_aver + image_lvl1
    end
    image_aver = image_aver/3.0
    int_merg[2048:3071,*] = congrid(image_aver[*,31:312],1024,300)

    filename_607 = file_search('../raw_data/sumer/20071116/level1/sum_r_20071116_09*0607_l1.sav')
    image_aver = dblarr(1024,360)
    for ii = 0, 2 do begin
        restore, filename_607[ii],/ver
        image_aver = image_aver + image_lvl1
    end
    image_aver = image_aver/3.0
    int_merg[3072:4095,*] = congrid(image_aver[*,29:310],1024,300)

    ;spec_gauss_widget,indgen(4096),int_merg[*,-10],fltarr(4096)+0.1/300.0
    ;spec_gauss_widget,indgen(4096),mean(int_merg[*,-42:-10],dimension=2),fltarr(4096)+0.1/300.0
    ;spec_gauss_widget,indgen(4096),mean(int_merg[*,-74:-43],dimension=2),fltarr(4096)+0.1/300.0
    print,slit_number
    save,filename = "../save/sumer_merg.sav",int_merg
end