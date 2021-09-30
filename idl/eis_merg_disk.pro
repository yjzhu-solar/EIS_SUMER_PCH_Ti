pro eis_merg_disk

filenames = file_search('../level0/lvl1_retain/eis_l1_*.fits')
new_data_offset = dblarr(4096,534,3)
wvl = dblarr(4096)

;calculate the offset
data = obj_new('eis_data',filenames[0])
for ii =0, 3 do begin
    start = ii * 1024
    ends = ii * 1024 + 1023
    wvl[start:ends] = data->getlam(ii)
end

wvl_offset = eis_ccd_offset(wvl)
offset_start = wvl_offset[0]
wvl_offset = wvl_offset - offset_start
pixel_offset = round(-wvl_offset)

print,filenames
index_start = 0
for ii = 0, sizeof(filenames) - 1 do begin
    data = obj_new('eis_data',filenames[ii])
    print,data->getycen(256,/raster)

    if (data->getycen(256,/raster) gt -830) then begin
        wvl_ii = data->getlam(0)
        print,wvl_ii[0]
        index_end = index_start + 2
        raw_image = dblarr(4096,512,3)
        raw_image[0:1023,*,*] = data->getvar(0)
        raw_image[1024:2047,*,*] = data->getvar(1)
        raw_image[2048:3071,*,*] = data->getvar(2)
        raw_image[3072:4095,*,*] = data->getvar(3)
        for kk = 0, 4095 do begin
            y_start = pixel_offset[kk]
            y_end = pixel_offset[kk] + 511
            new_data_offset[kk, y_start:y_end,index_start:index_end] = raw_image[kk, * ,*]
        end
        index_start = index_start + 3
        print,data->getexp()
    endif
    
end
save,filename = "../save/new_lvl1_offset_disk_retain.sav",wvl,new_data_offset
end
