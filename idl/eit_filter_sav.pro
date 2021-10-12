pro eit_filter_sav
    filename = "../raw_data/soho_eit_filter/eit_195_al1.txt"
    wvl = findgen(80)*0.5 + 175
    Aeff = eit_parms(wvl,195,"al1",text)

    openw,lun,filename,/get_lun
    printf,lun,"*SOHO/EIT 195 Effective Area"
    printf,lun,"*Filter: Al+1"
    printf,lun,text
    printf,lun,"Wavelength        Aeff"
    for ii = 0, n_elements(wvl) - 1 do begin
        printf,lun,FORMAT='(F7.2,G16.5)',wvl[ii],Aeff[ii]
    end
    free_lun,lun
end