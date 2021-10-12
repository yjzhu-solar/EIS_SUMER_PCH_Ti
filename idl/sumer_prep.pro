pro sumer_prep

filenames = file_search('../raw_data/sumer/20071116/sum*')
;filenames = file_search('../raw_data/sumer/20071116/sum_r_20071116_09364575.05099_02')
ff_filename = "../raw_data/sumer/20071116/ff_b_20070317_1101d_r.rst"

;restore flatfield matrix
restore,ff_filename,/ver
ff_matrix = image
ff_header = header_data

for ii = 0, sizeof(filenames) - 1 do begin
    restore,filenames[ii],/ver
    print,filenames[ii]
    decomp5,image_data,header_data,image_decomp ;decompress the IDL restore file
    image_decomp = reverse(image_decomp) ;reverse the image

    ;reference pixel after reversion
    ref_pix = 1023 - pixpos(header_data) + 1
    wvl_ref = wavel(header_data)
    xcen = suny(header_data)
    ycen = -sunz(header_data)
    xbin = biny(header_data)
    ybin = binz(header_data)
    date_obs = expsta(header_data)
    expt = exptim(header_data)
    magnification,wvl_ref*2,1,mag_a,d_lam_a,mag_b,d_lam_b
    if detector(header_data) eq 1 then dlamb = d_lam_a
    if detector(header_data) eq 2 then dlamb = d_lam_b

    print,"Reference pixel number before reversion: ",pixpos(header_data)
    print,"Reference pixel number after reversion: ",ref_pix
    print,"Wavelength of reference pixel: ",wvl_ref
    print,"Solar-X center",xcen
    print,"Solar-Y center",ycen
    print,"Spatial binning X",xbin
    print,"Spatial binning Y",ybin
    print,"Wavelength per pixel",dlamb

    ;determine the detector
    if detector(header_data) eq 1 then detector_name = "A"
    if detector(header_data) eq 2 then detector_name = "B"

    print,"Detector name: ",detector_name

    ;determine the slit
    slit_number = slitnum(header_data)

    print,"Slit number: ", slit_number

    p_start = 350
    p_end = 410
    y_start = 300
    y_end = 310
    ;window,0,xs=600,ys=400
    ;plot,indgen(60),mean(image_decomp[p_start:p_end,y_start:y_end],dimension=2) 
    ;counts --> count rates
    image_decomp = image_decomp/expt[0]
    ;deadtime correction
    deadtime_corr,detector_name,image_decomp,image_deadcorr,yevent(header_data)
    
    ;window,1,xs=600,ys=400
    ;plot,indgen(60),mean(image_deadcorr[p_start:p_end,y_start:y_end],dimension=2) 

    ;normal flat field (no odd even pattern correction)
    image_ff = sum_flatfield(image_deadcorr,slit_number,ref_pix,ff_matrix)
    ;window,2,xs=600,ys=400
    ;plot,indgen(60),mean(image_ff[p_start:p_end,y_start:y_end],dimension=2) 

    ;local gain correction
    local_gain_corr,detector_name,image_ff,image_lgcorr
    ;correct geometric distortion

    ;window,3,xs=600,ys=400
    ;plot,indgen(60),mean(image_lgcorr[p_start:p_end,y_start:y_end],dimension=2) 

    image_distcorr = destr_bilin(image_lgcorr,slit_number,ref_pix,detector_name)
    ;plot_image,alog10(image_distcorr)
    
    image_lvl1 = image_distcorr

    ;window,4,xs=600,ys=400
    ;plot,indgen(60),mean(image_distcorr[p_start:p_end,y_start:y_end],dimension=2) 
    
    filename_save = "../raw_data/sumer/20071116/level1/"+strmid(filenames[ii],32,29,/reverse_offset)+"_l1.sav"
    save,filename = filename_save,image_lvl1,ref_pix,wvl_ref,xcen, $
        ycen,xbin,ybin,date_obs,expt,slit_number,detector_name,dlamb


end
end