pro save_eis_recalibrate_intensity

    wvl_sw = findgen(401)/10 + 170
    wvl_lw = findgen(441)/10 + 246
    
    response_hpw_sw = dblarr(401)
    response_hpw_lw = dblarr(441)
    response_gdz_sw = dblarr(401)
    response_gdz_lw = dblarr(441)

    for ii = 0, 400 do begin
        response_hpw_sw[ii] = eis_recalibrate_intensity('16-NOV-2007',wvl_sw[ii],1)
        response_gdz_sw[ii] = eis_recalibrate_intensity('16-NOV-2007',wvl_sw[ii],1,/gdz)
    end

    for ii = 0, 440 do begin
        response_hpw_lw[ii] = eis_recalibrate_intensity('16-NOV-2007',wvl_lw[ii],1)
        response_gdz_lw[ii] = eis_recalibrate_intensity('16-NOV-2007',wvl_lw[ii],1,/gdz)
    end

    save,filename="../save/eis_recalibrate_intensity_table.sav",response_hpw_sw,response_gdz_sw,response_hpw_lw,response_gdz_lw,wvl_sw,wvl_lw



end