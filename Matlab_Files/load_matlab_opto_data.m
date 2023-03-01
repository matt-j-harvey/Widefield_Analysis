function load_matlab_opto_data(base_directory, fsm_filename)

    % Load FSM Data
    fsm_full_filepath = join([base_directory, '/', fsm_filename])
    fsm_data = load(fsm_full_filepath);
    fsm_data = fsm_data.fsm;

    % Extract Laser Power Data
    laser_powers = fsm_data.laserpower;

    % Extract Stimuli Identities
    session_lightbox = fsm_data.optoImages
    image_ids = session_lightbox.tbtImageID

    % Save These Images
    laser_power_filename = join([base_directory, "Stimuli_Onsets", "laser_powers.mat"], '/')
    opto_id_filename = join([base_directory,"Stimuli_Onsets", "opto_stim_ids.mat"], '/')
    save(laser_power_filename, "laser_powers")
    save(opto_id_filename, "image_ids")
    

end
 