pro correct_tilt_test
    restore,"../src/final_data_all.save",/ver
    data = reform(NEW_DATA_OFFSET[1150:1190,*,120])
    
    c1 = 0.1415e-3
    c2 = 0.0205e-6

    for i = 0,533 do begin
        wvl_cor = c1*i + c2*(i^2)
        data[*,i] = interpol(reform(data[*,i]),wvl[1150:1190]-wvl_cor,wvl[1150:1190])
    end

    window,1,xs=500,ys=500
    plot_image,alog(data),scale=[3,1]

end