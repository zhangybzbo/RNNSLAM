name=~/projects/ColonData/sequences/
num=031
python python/cvt_colon_to_tumrgbd.py \
	--depth_dir $name$num'/reproduce/depth_refined_145000/' \
	--image_dir $name$num'/image/' \
	--cameras_file_path $name$num'/reproduce/kf_pose_result.txt' \
	--output_dir $name$num'/reproduce/tum_refined_145000' \
	--intrinsic $name'calib_270_216.txt' \
	--repeat 1 --high_intensity_threshold 250 --low_intensity_threshold 70 --rescale_w 320 --rescale_h 256
