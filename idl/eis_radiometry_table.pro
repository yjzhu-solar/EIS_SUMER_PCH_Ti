pro eis_radiometry_table
    wvl_short = findgen(81)*0.5d + 170
    wvl_long = findgen(85)*0.5d + 247

    wvl = [wvl_short,wvl_long]
    
    time = '16-NOV-2007'
    filename = "../save/eis_recalib_radio_table.sav"

    hpw_int = eis_recalibrate_intensity(time,wvl,1)    
    gdz_int = eis_recalibrate_intensity(time,wvl,1,/gdz)


    save,filename = filename,wvl,hpw_int,gdz_int
end