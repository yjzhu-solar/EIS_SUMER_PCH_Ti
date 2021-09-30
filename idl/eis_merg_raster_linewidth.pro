pro eis_merg_raster_linewidth

n_window = 4
n_raster = 12
ypix_start = 256
n_ypix = 512
c1 = 0.1415e-3
c2 = 0.0205e-6
wvl_cor = c1*(findgen(n_ypix)+ypix_start) + c2*(findgen(n_ypix)^2+ypix_start)

;window_wvl_min_ccd_index = [604,680,766,826,885,954,1036,1148,1252,1534,1575,1655,2495,2704,2792,2860,3127,3348,3409,3484,3546,3590,3744]
window_start_index = intarr(n_window)
window_end_index = intarr(n_window)
filenames = file_search('../raw_data/linewidth_calib/eis_l1_*.fits')

;get wavelength points
data = obj_new('eis_data',filenames[0])
n_wvl = total(data->getxw())
print,"total wavelength points",n_wvl

index_start = 0
for ii = 0, sizeof(filenames) - 1 do begin
    print,filenames[ii]
    new_data_offset = dblarr(n_wvl,n_ypix+22,n_raster)
    new_err_offset = dblarr(n_wvl,n_ypix+22,n_raster)
    wvl = dblarr(n_wvl)
    raw_image = dblarr(n_wvl,n_ypix,n_raster)
    raw_image_err = dblarr(n_wvl,n_ypix,n_raster)
    ;print,eis_get_wininfo(filenames[ii],nwin=23,/list)
    ;p1=eis_getwindata(filenames[ii],0)
    ;help,p1,/str
    ;calculate the offset
    data = obj_new('eis_data',filenames[ii])
    yws = data->getinfo('YWS')
    print,yws
    current_index = 0
    for jj = 0, n_window - 1 do begin
        wvl_seg = data->getlam(jj)
        data_seg = data->getvar(jj)
        err_seg = data->geterr(jj)
        start = current_index
        ends = current_index + (n_elements(wvl_seg))[0]-1
        window_start_index[jj] = start
        window_end_index[jj] = ends
        wvl[start:ends] = wvl_seg
        ;print,size(data_seg)
        for kk = 0,n_raster-1 do begin
            for ll = 0,n_ypix-1 do begin
                data_seg[*,ll,kk] = interpol(reform(data_seg[*,ll,kk]),wvl_seg-wvl_cor[ll],wvl_seg,/spline)
                err_seg[*,ll,kk] = interpol(reform(err_seg[*,ll,kk]),wvl_seg-wvl_cor[ll],wvl_seg,/spline)
            end 
        end
        raw_image[start:ends,*,*] = data_seg
        raw_image_err[start:ends,*,*] = err_seg
        current_index = ends + 1
    end
    ;print,current_index

    wvl_offset = eis_ccd_offset(wvl)
    offset_start = wvl_offset[0]
    wvl_offset = wvl_offset - offset_start
    pixel_offset = round(-wvl_offset)

;    if (data->getycen(256,/raster) lt -1220) then begin
;        index_end = index_start + 2
    for kk = 0, n_wvl-1 do begin
        y_start = pixel_offset[kk]
        y_end = pixel_offset[kk] + n_ypix - 1
        ;print,kk,y_end
        new_data_offset[kk, y_start:y_end,*] = raw_image[kk, * ,*]
        new_err_offset[kk, y_start:y_end,*] = raw_image_err[kk, * ,*]
    end
;        index_start = index_start + 3
;        print,data->getexp()
;    endif
    save,filename = "../save/linewidth_calibration_tilt_cor"+strmid(filenames[ii],23,18,/reverse_offset)+".sav",wvl,new_data_offset,new_err_offset, $
    window_start_index,window_end_index
end

end