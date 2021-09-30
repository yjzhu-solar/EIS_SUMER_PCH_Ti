pro sav_slit_width
    ypix_start = 370 ;256
    slit_length = 384; 512
    slit_width = dblarr(slit_length) 
    for ii = 0,slit_length - 1 do begin
      slit_width[ii] = eis_slit_width(ii+ypix_start,slit_ind=2)
    end
    save,filename="../save/slit_width_384_370.sav",ypix_start,slit_width
end