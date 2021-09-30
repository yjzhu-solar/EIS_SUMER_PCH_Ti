pro eis_line_width_fit
    restore,"../save/new_lvl1_offset_limb_1116_northpole_tilt_cor_l1_20071116_07262.sav",/ver
    eis_all = mean(new_data_offset,dimension=3)
    eis_all_err = sqrt(total(new_err_offset^2.,3))/7.

    ;eis_fit = mean(eis_all[*,391:422],dimension=2)
    ;eis_fit_err = sqrt(total(eis_all_err[*,391:422]^2.,2))/16.

    eis_fit = mean(eis_all[*,359:390],dimension=2)
    eis_fit_err = sqrt(total(eis_all_err[*,359:390]^2.,2))/16.

    spec_gauss_widget,wvl,eis_fit,eis_fit_err,/angpix



end