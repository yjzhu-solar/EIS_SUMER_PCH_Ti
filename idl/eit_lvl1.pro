pro eit_lvl1
    filenames = file_search('../raw_data/eit_20071116/efz*')
    for ii = 0, sizeof(filenames) - 1 do begin
        time_exp = strmid(filenames[ii],5,6,/reverse_offset)
        ;print,time_exp
        ;filename_save = "../sav/eit_20071116/"+strmid(filenames[ii],14,15,/reverse_offset)+"_l1.sav"
        ;
        eit_prep, filenames[ii], hdr, img, outdir="../save/eit_20071116_l1/",/verbose
        ;save,filename = filename_save, hdr, img
    end
end